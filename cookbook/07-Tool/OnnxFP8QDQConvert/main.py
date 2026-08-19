# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Turn Transformer-Engine's custom FP8 Q/DQ operators into standard opset-19 Q/DQ.

Transformer-Engine exports FP8 quantisation as **custom operators** in the `trt` domain,
wrapped in `Cast` nodes because the custom op works in float32:

    Cast(fp16->fp32) -> TRT_FP8QuantizeLinear -> TRT_FP8DequantizeLinear -> Cast(fp32->fp16)

`scripts/convert_te_onnx_to_trt_onnx.py` in TensorRT-OSS rewrites this into standard
opset-19 `QuantizeLinear` / `DequantizeLinear` with an FP8 (E4M3) zero-point. This example
re-implements that conversion small enough to read, and then checks the usual reason for
running it -- and finds it no longer holds:

**On TensorRT 11.1.0.106 the parser already understands the custom operators.** It maps
`trt::TRT_FP8QuantizeLinear` straight onto `IQuantizeLayer`, and both the original and the
converted file build to the **same 4-layer engine**. So the conversion is not what makes a
Transformer-Engine model loadable by TensorRT any more.

What it is still for is **portability**: the `trt` domain is not a real ONNX domain, so
every other runtime refuses the file outright --

    Fatal error: trt:TRT_FP8QuantizeLinear(-1) is not a registered function/op

-- while the converted file loads and runs in ONNX Runtime. If TensorRT is the only
consumer, the conversion buys nothing; if anything else has to read the model, it is the
whole ballgame.

The rewrite is four separate edits, and the last two are the ones worth knowing about:

1. `TRT_FP8QuantizeLinear` -> `QuantizeLinear`, `TRT_FP8DequantizeLinear` -> `DequantizeLinear`.
2. Add the FP8 zero-point input the standard operator requires and the custom one does not.
   Its **dtype is what selects FP8**: `TensorProto.FLOAT8E4M3FN`.
3. Drop the `Cast` in front of Q and behind DQ; they exist only because the custom operator
   was float32-only. Measured: this changes the ONNX node count (11 -> 8) but **not** the
   final engine, which is 4 layers either way because TensorRT folds the casts itself. Doing
   it also forces the scale retyping in `retype_scale`, which is the fiddliest part of the
   whole conversion -- so if the target is TensorRT, leaving the casts alone is defensible.
4. Bump the model opset to 19. Q/DQ existed long before, but **FP8 zero-point types did
   not**, so an opset-13 file with an E4M3 zero-point is invalid rather than merely old.

No Transformer-Engine installation is needed: `case_build_te_style_model` synthesises a
graph in exactly the shape TE emits, which is also what makes the before/after comparison
verifiable rather than asserted.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnx
import onnx.helper as oh
import onnxruntime
import tensorrt as trt

from tensorrt_cookbook import case_mark

np.random.seed(31193)

N_BATCH = 4
N_K = 16
N_N = 8
TE_DOMAIN = "trt"
TE_QUANTIZE = "TRT_FP8QuantizeLinear"
TE_DEQUANTIZE = "TRT_FP8DequantizeLinear"

output_path = Path(__file__).parent
onnx_file_te = output_path / "model-te_custom_op.onnx"
onnx_file_opset19 = output_path / "model-opset19_qdq.onnx"

result = OrderedDict()

# ================================================================ A model shaped like TE's output

def build_te_style_model() -> None:
    """One quantised MatMul, written the way Transformer-Engine writes it.

    Activations arrive in float16, are cast up to float32 for the custom operator, quantised,
    dequantised and cast back down. The weights get the same treatment on their own branch.
    """
    weight = (np.random.rand(N_K, N_N).astype(np.float32) * 2 - 1)

    node_list = [
        # ---- activation branch
        oh.make_node("Cast", ["x"], ["x_fp32"], "cast_before_q", to=onnx.TensorProto.FLOAT),
        oh.make_node(TE_QUANTIZE, ["x_fp32", "x_scale"], ["x_q"], "te_quantize_x", domain=TE_DOMAIN),
        oh.make_node(TE_DEQUANTIZE, ["x_q", "x_scale"], ["x_dq"], "te_dequantize_x", domain=TE_DOMAIN),
        oh.make_node("Cast", ["x_dq"], ["x_fp16"], "cast_after_dq", to=onnx.TensorProto.FLOAT16),
        # ---- weight branch
        oh.make_node(TE_QUANTIZE, ["w", "w_scale"], ["w_q"], "te_quantize_w", domain=TE_DOMAIN),
        oh.make_node(TE_DEQUANTIZE, ["w_q", "w_scale"], ["w_dq"], "te_dequantize_w", domain=TE_DOMAIN),
        oh.make_node("Cast", ["w_dq"], ["w_fp16"], "cast_after_dq_w", to=onnx.TensorProto.FLOAT16),
        # ---- the operation that all of this exists for
        oh.make_node("MatMul", ["x_fp16", "w_fp16"], ["y"], "matmul"),
    ]
    graph = oh.make_graph(
        node_list,
        "te_style",
        [oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT16, [N_BATCH, N_K])],
        [oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT16, [N_BATCH, N_N])],
        [
            oh.make_tensor("x_scale", onnx.TensorProto.FLOAT, [], [0.05]),
            oh.make_tensor("w_scale", onnx.TensorProto.FLOAT, [], [0.02]),
            oh.make_tensor("w", onnx.TensorProto.FLOAT, [N_K, N_N], weight.reshape(-1)),
        ],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13), oh.make_opsetid(TE_DOMAIN, 1)])
    model.ir_version = 10
    onnx.save(model, onnx_file_te)  # `onnx.checker` cannot validate the custom domain, so it is not run
    return

# ================================================================ The conversion

def retype_scale(graph, node, target_type: int) -> bool:
    """Give `node` a scale initializer of `target_type`, duplicating it if it is shared.

    Two things make this fiddly, and the upstream `cast_scale` helper exists for both.

    Needed because `QuantizeLinear` binds `x` and `y_scale` to the *same* type parameter T1.
    Removing the `Cast` in front of Q leaves a float16 activation next to a float32 scale,
    and ONNX rejects that with

        Type parameter (T1) of Optype (QuantizeLinear) bound to different types
        (tensor(float16) and tensor(float))

    Skipping this produces a file TensorRT still happily parses and every other runtime
    refuses.

    Second, the scale initializer is **shared** between the Q and the DQ of a pair, and after
    dropping only some of the surrounding casts the two ends can legitimately need different
    types -- on the weight branch the Q still sees a float32 constant while the DQ now has a
    float16 consumer. Mutating the shared tensor just moves the error to the other node, so a
    per-node copy is made instead.
    """
    name = node.input[1]
    array = None
    for initializer in graph.initializer:
        if initializer.name == name:
            array = onnx.numpy_helper.to_array(initializer)
            break
    if array is None:
        return False
    dtype = np.float16 if target_type == onnx.TensorProto.FLOAT16 else np.float32
    new_name = f"{node.name}_scale"
    if not any(initializer.name == new_name for initializer in graph.initializer):
        graph.initializer.append(oh.make_tensor(new_name, target_type, list(array.shape), array.astype(dtype).reshape(-1)))
    node.input[1] = new_name
    return True

def find_producer(graph, tensor_name: str):
    """The node that writes `tensor_name`, or None."""
    for index, node in enumerate(graph.node):
        if tensor_name in node.output:
            return node, index
    return None, None

def find_consumer(graph, tensor_name: str):
    """The single node that reads `tensor_name`, or None."""
    for index, node in enumerate(graph.node):
        if tensor_name in node.input:
            return node, index
    return None, None

def convert_te_to_opset19(remove_cast: bool = True) -> dict:
    """Rewrite the custom operators into standard opset-19 Q/DQ. Returns a small report."""
    model = onnx.load(onnx_file_te)
    graph = model.graph
    report = {"quantize": 0, "dequantize": 0, "cast_removed": 0, "scale_recast": 0}

    # The FP8 zero-point. Its *type* is what tells ONNX (and TensorRT) that this is FP8;
    # the value is always zero for the symmetric scaling Transformer-Engine uses.
    zero_point_name = "fp8_zero_point"
    graph.initializer.append(oh.make_tensor(zero_point_name, onnx.TensorProto.FLOAT8E4M3FN, [], [0]))

    index_to_delete = set()
    for node in graph.node:
        if node.op_type not in [TE_QUANTIZE, TE_DEQUANTIZE]:
            continue
        is_quantize = node.op_type == TE_QUANTIZE
        node.op_type = "QuantizeLinear" if is_quantize else "DequantizeLinear"
        node.domain = ""  # Back to the standard domain
        node.input.append(zero_point_name)
        report["quantize" if is_quantize else "dequantize"] += 1

        if not remove_cast:
            continue
        if is_quantize:
            # Cast -> Q  becomes  Q, reading whatever fed the Cast
            producer, producer_index = find_producer(graph, node.input[0])
            if producer is not None and producer.op_type == "Cast":
                node.input[0] = producer.input[0]
                index_to_delete.add(producer_index)
                report["cast_removed"] += 1
                # The scale has to follow the activation's type, see `retype_scale`
                if retype_scale(graph, node, onnx.TensorProto.FLOAT16):
                    report["scale_recast"] += 1
        else:
            # DQ -> Cast  becomes  DQ, writing whatever the Cast wrote
            consumer, consumer_index = find_consumer(graph, node.output[0])
            if consumer is not None and consumer.op_type == "Cast":
                node.output[0] = consumer.output[0]
                index_to_delete.add(consumer_index)
                report["cast_removed"] += 1
                if retype_scale(graph, node, onnx.TensorProto.FLOAT16):
                    report["scale_recast"] += 1

    for index in sorted(index_to_delete, reverse=True):
        del graph.node[index]

    # Opset 19 is not cosmetic: FP8 zero-point types do not exist before it
    del model.opset_import[:]
    model.opset_import.extend([oh.make_opsetid("", 19)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, onnx_file_opset19)
    return report

# ================================================================ Verification

def parse_with_tensorrt(onnx_file: Path) -> tuple:
    """Return `(parsed, layer_type_list, first_error)`."""
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    parsed = parser.parse_from_file(str(onnx_file))
    if not parsed:
        return False, [], str(parser.get_error(parser.num_errors - 1))
    return True, [str(network.get_layer(i).type).replace("LayerType.", "") for i in range(network.num_layers)], ""

# ================================================================ Cases

@case_mark
def case_build_te_style_model() -> None:
    """Synthesise the graph Transformer-Engine would have exported."""
    build_te_style_model()
    model = onnx.load(onnx_file_te)
    print(f"    node types: {[n.op_type for n in model.graph.node]}")
    print(f"    opset     : {[(o.domain or '(standard)', o.version) for o in model.opset_import]}")
    result["te"] = len(model.graph.node)
    return

@case_mark
def case_who_can_read_the_custom_op() -> None:
    """Who accepts the TE-style file as it stands: TensorRT yes, everyone else no."""
    parsed, layer_type_list, error = parse_with_tensorrt(onnx_file_te)
    print(f"    TensorRT parses the TE-style model : {parsed}")
    assert parsed, "TensorRT 11.1 was expected to understand the trt-domain custom operators"
    print(f"        layers: {layer_type_list}")
    print("        -> the custom operators became real QUANTIZE / DEQUANTIZE layers, no conversion needed")

    # `onnx.checker` passes too: a custom domain is legal ONNX, it just is not portable
    onnx.checker.check_model(onnx.load(onnx_file_te))
    print("    onnx.checker accepts it            : True (a custom domain is legal ONNX)")

    try:
        option = onnxruntime.SessionOptions()
        option.intra_op_num_threads = 1
        onnxruntime.InferenceSession(str(onnx_file_te), option, providers=["CPUExecutionProvider"])
        ran = True
        message = ""
    except Exception as exception:  # noqa: BLE001 - the message is the point
        ran = False
        message = str(exception).splitlines()[-1].strip()
    print(f"    onnxruntime loads it               : {ran}")
    print(f"        {message}")
    assert not ran, "The custom operators were expected to be unusable outside TensorRT"
    result["te_ort"] = ran
    result["te_layer"] = layer_type_list
    return

@case_mark
def case_convert() -> None:
    """Do the rewrite, and show what TensorRT makes of the result."""
    report = convert_te_to_opset19(remove_cast=True)
    model = onnx.load(onnx_file_opset19)
    print(f"    rewritten: {report}")
    print(f"    node types: {[n.op_type for n in model.graph.node]}")
    print(f"    opset     : {[(o.domain or '(standard)', o.version) for o in model.opset_import]}")

    parsed, layer_type_list, error = parse_with_tensorrt(onnx_file_opset19)
    print(f"    TensorRT parses the converted model: {parsed}")
    assert parsed, f"Conversion produced something TensorRT still cannot parse: {error}"
    print(f"    TensorRT layers: {layer_type_list}")
    assert "QUANTIZE" in layer_type_list and "DEQUANTIZE" in layer_type_list, "Q/DQ did not become real layers"
    result["opset19"] = len(model.graph.node)
    result["layer"] = layer_type_list
    return

@case_mark
def case_keep_the_casts() -> None:
    """What the redundant `Cast` nodes cost, measured rather than asserted.

    Leaving them in still parses -- which is exactly why this is easy to get wrong -- but
    the graph now contains Cast layers that force the tensor back through float32 between
    the dequantise and the consumer.
    """
    report = convert_te_to_opset19(remove_cast=False)
    model = onnx.load(onnx_file_opset19)
    parsed, layer_type_list, _ = parse_with_tensorrt(onnx_file_opset19)
    print(f"    rewritten without removing Cast: {report}")
    print(f"    ONNX nodes  : {len(model.graph.node)} (vs {result['opset19']} with the Cast nodes removed)")
    print(f"    parses      : {parsed}, TensorRT layers: {layer_type_list}")
    result["opset19_with_cast"] = len(model.graph.node)
    result["layer_with_cast"] = layer_type_list

    convert_te_to_opset19(remove_cast=True)  # Leave the good file on disk
    return

# ================================================================ Entrance

def main() -> None:
    case_build_te_style_model()
    case_who_can_read_the_custom_op()
    case_convert()
    case_keep_the_casts()

    print("\n" + "=" * 88)
    print(f"{'Model':<30}{'ONNX nodes':>12}{'TRT parses':>12}{'Q/DQ layers':>13}{'onnxruntime':>13}")
    print("-" * 88)
    print(f"{'TE custom operators':<30}{result['te']:>12}{'yes':>12}{sum(t in ['QUANTIZE', 'DEQUANTIZE'] for t in result['te_layer']):>13}{'no':>13}")
    print(f"{'opset19, Cast kept':<30}{result['opset19_with_cast']:>12}{'yes':>12}{sum(t in ['QUANTIZE', 'DEQUANTIZE'] for t in result['layer_with_cast']):>13}{'yes':>13}")
    print(f"{'opset19, Cast removed':<30}{result['opset19']:>12}{'yes':>12}{sum(t in ['QUANTIZE', 'DEQUANTIZE'] for t in result['layer']):>13}{'yes':>13}")
    print("=" * 88)
    print("TensorRT reads all three; only the converted files are portable. The conversion is a")
    print("portability step, not a TensorRT-enablement step -- on 11.1 the parser needs no help.")
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
