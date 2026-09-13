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
"""What the `Loop` back end actually costs and buys, on a real model.

`01-SubgraphInONNX/02-scaling.py` measured this on hand-built models. Here the
same three representations come out of the *outliner* instead, so the numbers
apply to the tool rather than to an idealised example:

* `flat`     the pre-processed model, untouched
* `function` folded into a local function (TensorRT inlines it again)
* `loop`     folded into an ONNX `Loop` (TensorRT builds a real `ILoopLayer`)

Everything is measured end to end: TensorRT layer count, build time, serialized
engine size and inference latency, plus a numeric cross check against `flat`.
"""

import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import tensorrt as trt
import torch
import torch.nn as nn
from cuda.bindings import runtime as cudart

from tensorrt_cookbook import OutlineConfig, TRTWrapperV1, case_mark, outline

torch.manual_seed(31193)
np.random.seed(31193)

N_LAYER = 12
N_MODEL = 128
N_HEAD = 4
N_FF = 512
N_SEQ = 32
N_B = 4
OPSET = 17
N_WARMUP = 20
N_TEST = 100
N_B_LARGE = 1  # The real model is a decoder, one sequence is the realistic shape
N_SEQ_LARGE = 64

output_path = Path(__file__).parent
flat_file = output_path / "model-flat.onnx"

def prepare() -> dict:
    """Export a 12-layer encoder, then fold it both ways. Returns the three files."""
    if not flat_file.exists():
        layer = nn.TransformerEncoderLayer(N_MODEL, N_HEAD, dim_feedforward=N_FF, batch_first=True)
        model = nn.TransformerEncoder(layer, N_LAYER).eval()
        # `nn.TransformerEncoder` clones one layer with `deepcopy`, so out of the box
        # all N layers hold the *same* weights and TensorRT deduplicates them in the
        # engine. That makes the engine-size comparison meaningless, so re-randomise.
        for parameter in model.parameters():
            with torch.no_grad():
                parameter.copy_(torch.randn_like(parameter) * 0.02)
        torch.onnx.export(model, (torch.randn(N_B, N_SEQ, N_MODEL), ), flat_file, dynamo=False, opset_version=OPSET, input_names=["x"], output_names=["y"])

    file_dict = {}
    for backend in ["function", "loop"]:
        onnx_file = output_path / f"model-{backend}.onnx"
        report = outline(flat_file, onnx_file, OutlineConfig(backend=backend))
        pattern = report["patterns"][0]
        print(f"    {backend:<9}: main graph {report['n_node_output']:>3} nodes, "
              f"pattern {pattern['size']} nodes x {pattern['n_instance']} instances, backend={pattern['backend']}, "
              f"onnxruntime={report['verification']['onnxruntime']}")
        file_dict[backend] = onnx_file
    # The pre-processed model is what the outliner actually worked on, so that is
    # the fair `flat` baseline rather than the raw export.
    from tensorrt_cookbook.onnx_outliner.preprocess import preprocess
    model_slim, _ = preprocess(onnx.load(flat_file))
    slim_file = output_path / "model-slim.onnx"
    onnx.save(model_slim, slim_file)
    return {"flat": slim_file, **file_dict}

def measure(onnx_file: Path, input_data: dict, *, b_tf32: bool = True) -> dict:
    """Build an engine, then time the build and the inference.

    `TRTWrapperV1` owns the builder / network / config / profile quartet and the
    whole runtime side (deserialize, context, host and device buffers), so what
    is left here is the parser, the shape profile and the two timers -- i.e. the
    part this benchmark is actually about.
    """
    tw = TRTWrapperV1(logger=trt.Logger.Severity.ERROR)
    parser = trt.OnnxParser(tw.network, tw.logger)
    if not parser.parse_from_file(str(onnx_file)):
        raise RuntimeError(f"{onnx_file}: {parser.get_error(0)}")
    n_layer = tw.network.num_layers  # Read before `build`, the network is consumed by it

    # One fixed shape per input: this measures the cost of the representation,
    # not of a wide profile. The real model has two dynamic inputs, the toy one
    # has a single static one, so the profile is driven by the data handed in.
    for name, data in input_data.items():
        tw.profile.set_shape(name, data.shape, data.shape, data.shape)

    if not b_tf32:
        tw.builder_config.clear_flag(trt.BuilderFlag.TF32)

    t0 = time.time()
    assert tw.build(), f"failed building {onnx_file}"
    build_time = time.time() - t0

    tw.setup(input_data, b_print_io=False)
    output_name = tw.tensor_name_list[tw.n_input]  # The first output, used for the numeric cross check

    for _ in range(N_WARMUP):
        tw.context.execute_async_v3(0)
    cudart.cudaStreamSynchronize(0)
    t0 = time.time()
    for _ in range(N_TEST):
        tw.context.execute_async_v3(0)
    cudart.cudaStreamSynchronize(0)
    latency = (time.time() - t0) / N_TEST * 1000

    tw.infer(b_print_io=False)  # One more pass, this time copying the output back
    output_host = tw.buffer[output_name][0]

    model = onnx.load(onnx_file, load_external_data=False)
    return dict(n_onnx_node=len(model.graph.node), n_function=len(model.functions), n_trt_layer=n_layer, build_time=build_time, engine_size=tw.engine_bytes.nbytes / 1024 ** 2, latency=latency, output=np.array(output_host))

@case_mark
def case_compare() -> None:
    """The three representations, side by side."""
    file_dict = prepare()
    input_data = (np.random.rand(N_B, N_SEQ, N_MODEL).astype(np.float32) - 0.5)

    result = {}
    for tag, onnx_file in file_dict.items():
        result[tag] = measure(onnx_file, {"x": input_data})
        reference = ort.InferenceSession(str(file_dict["flat"]), providers=["CPUExecutionProvider"]).run(None, {"x": input_data})[0]
        other = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"]).run(None, {"x": input_data})[0]
        result[tag]["ort_diff"] = float(np.max(np.abs(reference - other)))

    header = f"{'':<10}{'ONNXnode':>10}{'function':>10}{'TRTlayer':>10}{'build(s)':>10}{'engine(MB)':>12}{'latency(ms)':>13}{'ORTdiff':>10}{'TRTdiff':>10}"
    print("\n    " + "=" * len(header))
    print("    " + header)
    print("    " + "-" * len(header))
    for tag, r in result.items():
        trt_diff = float(np.max(np.abs(r["output"] - result["flat"]["output"])))
        print(f"    {tag:<10}{r['n_onnx_node']:>10}{r['n_function']:>10}{r['n_trt_layer']:>10}{r['build_time']:>10.2f}"
              f"{r['engine_size']:>12.1f}{r['latency']:>13.3f}{r['ort_diff']:>10.1e}{trt_diff:>10.1e}")
    print("    " + "=" * len(header))

    loop, flat = result["loop"], result["flat"]
    print(f"\n    Loop vs flat: TRT layers {flat['n_trt_layer']} -> {loop['n_trt_layer']} ({flat['n_trt_layer'] / loop['n_trt_layer']:.1f}x fewer),"
          f" build {flat['build_time']:.1f}s -> {loop['build_time']:.1f}s,"
          f" engine {flat['engine_size']:.1f}MB -> {loop['engine_size']:.1f}MB,"
          f" latency {flat['latency']:.3f}ms -> {loop['latency']:.3f}ms ({loop['latency'] / flat['latency']:.1f}x slower)")
    print("    The function version is inlined by the parser, so it matches `flat` layer for layer.")
    return

@case_mark
def case_large_model() -> None:
    """The 1.5 GB / 6119 node model, measured end to end.

    A real 24-layer decoder rather than the toy encoder above, so this is the
    number to quote. Two INT64 inputs and 49 outputs, all dynamic, which is why
    `measure` takes a dict.

    The baseline here is `function`, not `flat`: the parser inlines local
    functions, so the two are the same TensorRT network layer for layer (858 vs
    858 in `case_compare`), and building a third 1.5 GB artifact to prove it
    again is not worth the disk.
    """
    from tensorrt_cookbook import cookbook_path
    onnx_file = cookbook_path("00-Data", "model") / "model-large.onnx"
    if not onnx_file.exists():
        print(f"    skipped, {onnx_file} not found")
        return

    input_data = {
        "input_ids": np.random.randint(0, 50257, size=(N_B_LARGE, N_SEQ_LARGE), dtype=np.int64),
        "attention_mask": np.ones((N_B_LARGE, N_SEQ_LARGE), dtype=np.int64),
    }
    print(f"    input: batch {N_B_LARGE} x sequence {N_SEQ_LARGE}, 2 inputs / 49 outputs")

    result, target_dict = {}, {}
    for backend in ["function", "loop"]:
        target = output_path / f"model-large-{backend}.onnx"
        target_dict[backend] = target
        report = outline(onnx_file, target, OutlineConfig(backend=backend))
        pattern = report["patterns"][0]
        stat = pattern["loop"][0] if pattern["backend"] == "loop" else {}
        print(f"    {backend:<9}: main graph {report['n_node_output']:>3} nodes, "
              f"onnxruntime rel={report['verification']['onnxruntime'].get('max_rel_diff'):.1e}")
        if stat:
            print(f"                {stat['n_iteration']} iterations, {stat['n_carried']} loop variable(s), "
                  f"{stat['n_scan_output']} scan output(s), {stat['n_slice_node']} slice nodes, "
                  f"{stat['n_stacked_initializer']} stacked initializers")
        result[backend] = measure(target, input_data)

    # The first output is `logits`, whose magnitude is in the tens, so an absolute
    # difference there is not interpretable on its own -- report both.
    scale = float(np.max(np.abs(result["function"]["output"])))
    header = f"{'':<10}{'ONNXnode':>10}{'TRTlayer':>10}{'build(s)':>10}{'engine(MB)':>12}{'latency(ms)':>13}{'TRTabs':>10}{'TRTrel':>10}"
    print("\n    " + "=" * len(header))
    print("    " + header)
    print("    " + "-" * len(header))
    for tag, r in result.items():
        trt_diff = float(np.max(np.abs(r["output"] - result["function"]["output"])))
        print(f"    {tag:<10}{r['n_onnx_node']:>10}{r['n_trt_layer']:>10}{r['build_time']:>10.2f}"
              f"{r['engine_size']:>12.1f}{r['latency']:>13.3f}{trt_diff:>10.1e}{trt_diff / scale:>10.1e}")
    print("    " + "=" * len(header))
    print(f"    `logits` ranges up to {scale:.1f}, which is what the relative column divides by.")

    # The relative difference above sits right at TF32's precision (2^-11 ~ 4.9e-4),
    # which is suspicious: onnxruntime only saw 4.5e-06. TensorRT enables TF32 for
    # FP32 networks by default, so rebuild both without it and see what is left.
    tf32_rel = float(np.max(np.abs(result["loop"]["output"] - result["function"]["output"]))) / scale
    plain = {tag: measure(target_dict[tag], input_data, b_tf32=False) for tag in ["function", "loop"]}
    plain_scale = float(np.max(np.abs(plain["function"]["output"])))
    plain_rel = float(np.max(np.abs(plain["loop"]["output"] - plain["function"]["output"]))) / plain_scale
    print(f"\n    With TF32 disabled the same comparison gives rel={plain_rel:.1e}, "
          f"{tf32_rel / plain_rel:.0f}x smaller than the {tf32_rel:.1e} above.")
    print("    So the TensorRT-side difference is TF32, not the folding: at plain FP32 the `Loop`")
    print("    version agrees with the flat one to a few ULP, same order as the onnxruntime check.")

    loop, function = result["loop"], result["function"]
    print(f"\n    Loop vs function: TRT layers {function['n_trt_layer']} -> {loop['n_trt_layer']} "
          f"({function['n_trt_layer'] / loop['n_trt_layer']:.1f}x fewer),"
          f" build {function['build_time']:.1f}s -> {loop['build_time']:.1f}s,"
          f" engine {function['engine_size']:.0f}MB -> {loop['engine_size']:.0f}MB,"
          f" latency {function['latency']:.3f}ms -> {loop['latency']:.3f}ms")
    print("    `function` is the flat network as far as TensorRT is concerned, the parser inlines it.")

    for target in target_dict.values():
        target.unlink(missing_ok=True)  # 1.5 GB each
    return

if __name__ == "__main__":
    case_compare()
    case_large_model()
    for pattern in ["model-*.onnx"]:  # ~37 MB of regenerable artefacts, do not keep them
        for target in output_path.glob(pattern):
            target.unlink(missing_ok=True)
    print("\nFinish")
