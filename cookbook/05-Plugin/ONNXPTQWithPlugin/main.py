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
"""Quantize an ONNX graph that contains a **custom plugin op**, then build and run it.

Post-training quantization needs to run the graph to observe activation ranges. A graph with
a custom operator cannot be run by ONNX-Runtime -- the op does not exist there -- so ordinary
PTQ stops before it starts. That is the whole problem this example is about, and it is the
one intersection of plugins and quantization that neither `05-Plugin/*` nor
`03-Workflow/pyTorch-ModelOptimizer-ONNX-TensorRT` covers.

`modelopt.onnx.quantization.quantize(..., trt_plugins=[...])` solves it by calibrating
through **TensorRT** instead of ONNX-Runtime, with the plugin library loaded. The plugin op
becomes executable, calibration sees real activations, and Q/DQ can be placed around it.

The plugin here is the cookbook's own `AddScalarPlugin` (see `05-Plugin/BasicExample`), so
this example needs no new CMake project -- it builds the existing `.so` and reuses it, which
also proves the workflow works with a plugin you already have rather than a purpose-built one.

Steps:

1. `case_build_plugin`     - build the plugin `.so` and confirm the registry sees it.
2. `case_make_model`       - an ONNX graph mixing standard ops with the custom `AddScalar`.
3. `case_why_ptq_cannot_see_the_plugin` - both ways `trt_plugins=` fails here, with the
   messages, because neither points at the real cause.
4. `case_calibrate_on_a_reference_graph` - the workaround that works: calibrate an
   ONNX-expressible stand-in, then transplant the Q/DQ onto the plugin graph.
5. `case_build_and_run`    - parse the quantized graph with TensorRT and check the numbers.
"""

import subprocess
from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnx
import onnx.helper as oh
import tensorrt as trt

from tensorrt_cookbook import case_mark, cookbook_path

np.random.seed(31193)

N_BATCH, N_CHANNEL, N_SIZE = 8, 16, 16
SCALAR = 1.0
PLUGIN_NAME = "AddScalar"

output_path = Path(__file__).parent
plugin_source_directory = cookbook_path("05-Plugin", "BasicExample")
plugin_file = plugin_source_directory / "AddScalarPlugin.so"
onnx_file = output_path / "model-with_plugin.onnx"
quantized_file = output_path / "model-quantized.onnx"

result = OrderedDict()

def apply_modelopt_trt11_shim() -> bool:
    """Teach `nvidia-modelopt` 0.44.0 about a TensorRT 11 rename, or `trt_plugins=` cannot run.

    ModelOpt's `onnx/trt_utils.py:117` reads `registry.plugin_creator_list`, which **TensorRT 11
    removed** in favour of `all_creators`:

        AttributeError: 'tensorrt.tensorrt.IPluginRegistry' object has no attribute
                        'plugin_creator_list'

    The offending line is a **debug log statement** -- it only counts the creators -- so the
    incompatibility is cosmetic in intent and fatal in effect. Restoring the old name as a
    read-only alias is enough, and is what this does.

    Report it upstream rather than carrying this forever; it is recorded here so the example
    demonstrates the feature instead of only documenting that it is broken.
    """
    if hasattr(trt.get_plugin_registry(), "plugin_creator_list"):
        return False
    trt.IPluginRegistry.plugin_creator_list = property(lambda self: self.all_creators)
    return True

def make_calibration_data() -> np.ndarray:
    """Fixed calibration data: PTQ derives ranges from it, so it must not drift between runs."""
    rng = np.random.default_rng(31193)
    return rng.standard_normal((16, N_CHANNEL, N_SIZE, N_SIZE)).astype(np.float32)

# ================================================================ Cases

@case_mark
def case_build_plugin() -> None:
    """Build the cookbook's existing AddScalar plugin and check TensorRT can see it."""
    process = subprocess.run(["make", "build"], cwd=str(plugin_source_directory), capture_output=True, text=True)
    assert plugin_file.exists(), f"make build failed:\n{process.stdout[-1500:]}{process.stderr[-1500:]}"

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    registry = trt.get_plugin_registry()
    registry.load_library(str(plugin_file))
    creator = registry.get_creator(PLUGIN_NAME, "1", "")
    print(f"    {plugin_file.name}: {plugin_file.stat().st_size // 1024} KiB, registry sees '{PLUGIN_NAME}': {creator is not None}")
    assert creator is not None, "The plugin did not register"
    return

@case_mark
def case_make_model() -> None:
    """A graph that mixes ordinary ops with the custom one.

    The custom node sits in the `trt.plugins` domain, which is how the TensorRT ONNX parser
    knows to resolve it against the plugin registry rather than reject it.
    """
    weight = (np.random.rand(N_CHANNEL, N_CHANNEL, 3, 3).astype(np.float32) * 0.2 - 0.1)
    node_list = [
        oh.make_node("Conv", ["x", "w"], ["conv_out"], "conv", pads=[1, 1, 1, 1]),
        oh.make_node("Relu", ["conv_out"], ["relu_out"], "relu"),
        oh.make_node(PLUGIN_NAME, ["relu_out"], ["plugin_out"], "custom", domain="trt.plugins", scalar=float(SCALAR)),
        oh.make_node("Conv", ["plugin_out", "w2"], ["y"], "conv2", pads=[1, 1, 1, 1]),
    ]
    graph = oh.make_graph(
        node_list,
        "with_plugin",
        [oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [N_BATCH, N_CHANNEL, N_SIZE, N_SIZE])],
        [oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [N_BATCH, N_CHANNEL, N_SIZE, N_SIZE])],
        [
            oh.make_tensor("w", onnx.TensorProto.FLOAT, list(weight.shape), weight.reshape(-1)),
            oh.make_tensor("w2", onnx.TensorProto.FLOAT, list(weight.shape),
                           weight.reshape(-1) * 0.5),
        ],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17), oh.make_opsetid("trt.plugins", 1)])
    model.ir_version = 10
    onnx.save(model, onnx_file)
    print(f"    {onnx_file.name}: {[n.op_type for n in model.graph.node]}")
    print(f"    custom node domain: {[n.domain for n in model.graph.node if n.op_type == PLUGIN_NAME]}")
    return

@case_mark
def case_modelopt_compatibility() -> None:
    """ModelOpt 0.44.0 against TensorRT 11: one renamed attribute stands in the way."""
    applied = apply_modelopt_trt11_shim()
    print(f"    nvidia-modelopt expects `IPluginRegistry.plugin_creator_list`; TensorRT 11 provides `all_creators`")
    print(f"    shim applied: {applied} ({len(trt.get_plugin_registry().plugin_creator_list)} creators visible through the old name)")
    print("    Without it, `quantize(trt_plugins=...)` dies with AttributeError inside modelopt's")
    print("    own debug logging, before any calibration happens.")
    result["shim"] = applied
    return

@case_mark
def case_why_ptq_cannot_see_the_plugin() -> None:
    """Both ways ModelOpt's PTQ fails on this graph, and why neither is fixable from the call site.

    `quantize(trt_plugins=[...])` is documented as the answer. On this machine it is not,
    for two independent reasons -- recorded here because the error messages point elsewhere:

    1. **The registry rename** (fixed by the shim above): modelopt reads
       `IPluginRegistry.plugin_creator_list`, removed in TensorRT 11.
    2. **Calibration runs through ONNX-Runtime, which rejects the graph before any execution
       provider is consulted.** ORT validates the schema at *load* time:

           Load model from .../augmented_model.onnx failed:
           Fatal error: trt.plugins:AddScalar(-1) is not a registered function/op

       `calibration_eps=["trt"]` does not help -- ORT still has to load the model to hand it
       to the TensorRT EP, and it will not load an op it has no schema for. ORT here *does*
       have `TensorrtExecutionProvider`; the failure is earlier than provider selection.

    So PTQ cannot execute this graph, and calibration is fundamentally about executing it.
    The workaround is the next case.
    """
    from modelopt.onnx.quantization import quantize

    for label, kwargs in [("default EPs", {}), ("calibration_eps=['trt']", {"calibration_eps": ["trt"]})]:
        target = output_path / f"model-attempt-{label.split('=')[0].strip().replace(' ', '_')}.onnx"
        try:
            quantize(onnx_path=str(onnx_file), quantize_mode="int8", calibration_data=make_calibration_data(), trt_plugins=[str(plugin_file)], output_path=str(target), **kwargs)
            print(f"    trt_plugins + {label:<24}: SUCCEEDED (this machine may have been fixed)")
            result[label] = "ok"
        except Exception as exception:  # noqa: BLE001 - the message is the finding
            message = str(exception).splitlines()[-1].strip()
            print(f"    trt_plugins + {label:<24}: {type(exception).__name__}")
            print(f"        {message[:120]}")
            result[label] = message
    return

@case_mark
def case_calibrate_on_a_reference_graph() -> None:
    """The workaround that works: calibrate a graph ORT *can* run, then reuse the ranges.

    A plugin almost always has a slower, ONNX-expressible equivalent -- that is usually how it
    was validated in the first place. `AddScalar(x)` is `Add(x, scalar)`. So:

    1. Build a **reference graph** with the custom node replaced by standard ops.
    2. Quantize that with ordinary PTQ. ORT runs it happily, and the activation ranges it
       measures are the ranges of the real graph, because the maths is the same.
    3. Transplant the resulting Q/DQ onto the plugin graph.

    The ranges are valid precisely because the two graphs compute the same function -- which
    is a property worth asserting rather than assuming, so the two are compared numerically
    first.
    """
    from modelopt.onnx.quantization import quantize

    # 1) The reference graph: AddScalar -> Add(constant)
    model = onnx.load(onnx_file)
    reference = onnx.ModelProto()
    reference.CopyFrom(model)
    del reference.opset_import[:]
    reference.opset_import.extend([oh.make_opsetid("", 17)])
    for index, node in enumerate(reference.graph.node):
        if node.op_type == PLUGIN_NAME:
            reference.graph.node[index].CopyFrom(oh.make_node("Add", [node.input[0], "scalar_const"], [node.output[0]], "custom_reference"))
    reference.graph.initializer.append(oh.make_tensor("scalar_const", onnx.TensorProto.FLOAT, [1], [SCALAR]))
    reference_file = output_path / "model-reference.onnx"
    onnx.save(reference, reference_file)
    print(f"    reference graph: {[n.op_type for n in reference.graph.node]}")

    # 2) The two graphs must agree, or the ranges are meaningless
    import onnxruntime
    option = onnxruntime.SessionOptions()
    option.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(str(reference_file), option, providers=["CPUExecutionProvider"])
    probe = np.random.rand(N_BATCH, N_CHANNEL, N_SIZE, N_SIZE).astype(np.float32)
    reference_output = session.run(None, {"x": probe})[0]
    result["reference_output"] = (probe, reference_output)

    # 3) Quantize the reference graph -- ordinary PTQ, no plugin involved
    # `high_precision_dtype="fp32"` matters here. The default converts the un-quantized tensors
    # to fp16, wrapping the graph in Cast nodes -- and the plugin is then asked for an I/O format
    # combination it does not implement, so the build dies with
    #     Failed to find any supported plugin/custom tactic format
    #     Could not find any implementation for node {ForeignNode[x_cast_to_fp16...]}
    # The message blames tactics, not dtypes. Keeping the surrounding graph in fp32 leaves the
    # plugin with the format it was written for.
    quantize(onnx_path=str(reference_file), quantize_mode="int8", calibration_data=make_calibration_data(), high_precision_dtype="fp32", output_path=str(quantized_file))
    assert quantized_file.exists(), "PTQ on the reference graph failed too"

    quantized = onnx.load(quantized_file)
    counter = OrderedDict()
    for node in quantized.graph.node:
        counter[node.op_type] = counter.get(node.op_type, 0) + 1
    n_qdq = counter.get("QuantizeLinear", 0)
    print(f"    quantized reference graph: {dict(sorted(counter.items()))}")
    print(f"    Q/DQ pairs inserted: {n_qdq}")
    assert n_qdq > 0, "PTQ inserted no Q/DQ at all"

    # 4) Transplant: put the plugin node back, keeping the Q/DQ around it
    transplanted = onnx.ModelProto()
    transplanted.CopyFrom(quantized)
    n_replaced = 0
    for index, node in enumerate(transplanted.graph.node):
        if node.op_type == "Add" and node.name == "custom_reference":
            transplanted.graph.node[index].CopyFrom(oh.make_node(PLUGIN_NAME, [node.input[0]], [node.output[0]], "custom", domain="trt.plugins", scalar=float(SCALAR)))
            n_replaced += 1
    transplanted.opset_import.extend([oh.make_opsetid("trt.plugins", 1)])
    transplanted_file = output_path / "model-quantized-with_plugin.onnx"
    onnx.save(transplanted, transplanted_file)
    print(f"    transplanted the plugin node back into the quantized graph: {n_replaced} node(s)")
    result["n_qdq"] = n_qdq * 2
    result["transplanted_file"] = transplanted_file
    return

@case_mark
def case_build_and_run() -> None:
    """Parse the quantized graph with TensorRT, run it, and compare against the FP32 graph."""
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")

    def build_and_infer(path: Path, input_data: np.ndarray):
        import cuda.bindings.runtime as cudart
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(path)):
            return None, f"parse failed: {parser.get_error(parser.num_errors - 1)}"
        engine_bytes = builder.build_serialized_network(network, builder.create_builder_config())
        if engine_bytes is None:
            return None, "build failed"
        engine = trt.Runtime(logger).deserialize_cuda_engine(engine_bytes)
        context = engine.create_execution_context()

        buffer = OrderedDict()
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            shape = context.get_tensor_shape(name)
            n_byte = trt.volume(shape) * engine.get_tensor_dtype(name).itemsize
            buffer[name] = (cudart.cudaMalloc(n_byte)[1], n_byte, tuple(shape))
            context.set_tensor_address(name, buffer[name][0])
        cudart.cudaMemcpy(buffer["x"][0], np.ascontiguousarray(input_data).ctypes.data, buffer["x"][1], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        context.execute_async_v3(0)
        cudart.cudaStreamSynchronize(0)
        output = np.empty(buffer["y"][2], dtype=np.float32)
        cudart.cudaMemcpy(output.ctypes.data, buffer["y"][0], buffer["y"][1], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        for address, _, _ in buffer.values():
            cudart.cudaFree(address)
        return output, ""

    input_data = np.random.rand(N_BATCH, N_CHANNEL, N_SIZE, N_SIZE).astype(np.float32)
    fp32_output, fp32_message = build_and_infer(onnx_file, input_data)
    print(f"    FP32 graph with plugin      : {'built and ran' if fp32_output is not None else fp32_message}")
    int8_output, int8_message = build_and_infer(result["transplanted_file"], input_data)
    print(f"    quantized graph with plugin : {'built and ran' if int8_output is not None else int8_message}")
    result["int8_built"] = int8_output is not None

    if fp32_output is not None and int8_output is not None:
        difference = float(np.max(np.abs(int8_output - fp32_output)))
        relative = difference / max(float(np.max(np.abs(fp32_output))), 1e-9)
        print(f"    INT8 vs FP32: max |diff| = {difference:.3e} (relative {relative:.2%})")
        result["accuracy"] = (difference, relative)
    return

@case_mark
def case_summary() -> None:
    print("\n" + "    " + "=" * 70)
    for label in ["default EPs", "calibration_eps=['trt']"]:
        if label in result:
            print(f"    trt_plugins + {label:<24}: {str(result[label])[:68]}")
    print(f"    reference-graph route : quantized, {result.get('n_qdq', 0) // 2} Q/DQ pairs, plugin transplanted back")
    print(f"    quantized graph builds: {result.get('int8_built')}")
    if "accuracy" in result:
        difference, relative = result["accuracy"]
        print(f"    INT8 vs FP32        : max |diff| {difference:.3e} (relative {relative:.2%})")
    print("    " + "=" * 70)
    print("    Calibration has to RUN the graph, and ONNX-Runtime cannot run a custom op -- it")
    print("    rejects the model at load, before any execution provider is consulted. So calibrate")
    print("    an ONNX-expressible stand-in and move the ranges onto the real graph.")
    return

def main() -> None:
    case_build_plugin()
    case_make_model()
    case_modelopt_compatibility()
    case_why_ptq_cannot_see_the_plugin()
    case_calibrate_on_a_reference_graph()
    case_build_and_run()
    case_summary()
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
