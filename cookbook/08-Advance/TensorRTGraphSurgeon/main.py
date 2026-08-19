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
"""Edit a parsed network at the `INetworkDefinition` level, then build and run it.

ONNX-GraphSurgeon edits the ONNX *before* the parser; this edits what the parser *produced*. That
matters when the thing to change only exists after parsing (a layer the parser synthesised, a
tactic-visible structure), or when the ONNX is not yours to modify.

The toolkit is smaller than it looks. `INetworkDefinition` offers `get_layer` / `num_layers` to
walk, `mark_output` / `unmark_output` to move the boundary, and `ILayer.set_input` to rewire. There
is **no `remove_layer`**: a layer is deleted by making nothing consume it and letting the builder
drop it, which case 3 does.

The last case is the one worth the example: an ONNX that TensorRT already runs perfectly is made to
use our own plugin instead, and the price of that is measured rather than assumed.
"""

import subprocess
import time
from pathlib import Path

import numpy as np
import onnx
import tensorrt as trt
from cuda.bindings import runtime as cudart
from tensorrt_cookbook import TRTWrapperV1, case_mark, cookbook_path, get_plugin, load_plugin_files

SHAPE = [8, 512, 512]  # Large enough that the kernel time in case 4 means something
SCALAR = 1.0
onnx_file = Path(__file__).parent / "model-add-relu-mul.onnx"
plugin_file = cookbook_path("05-Plugin", "BasicExample") / "AddScalarPlugin.so"

input_data = {"x": (np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE) % 7 - 3)}
reference = np.maximum(input_data["x"] + SCALAR, 0) * 2.0

def build_onnx_file():
    """`x -> Add(1.0) -> Relu -> Mul(2.0) -> y`, generated so the target layer is unambiguous.

    A model from `00-Data/model/` would do as well, but its layer names are not obviously tied to
    the operation being replaced, and every case below starts by *finding* a layer by name.
    """
    node_list = [
        onnx.helper.make_node("Add", ["x", "one"], ["a"], "node_add"),
        onnx.helper.make_node("Relu", ["a"], ["b"], "node_relu"),
        onnx.helper.make_node("Mul", ["b", "two"], ["y"], "node_mul"),
    ]
    initializer_list = [
        onnx.numpy_helper.from_array(np.array([SCALAR], dtype=np.float32), "one"),
        onnx.numpy_helper.from_array(np.array([2.0], dtype=np.float32), "two"),
    ]
    graph = onnx.helper.make_graph(
        node_list,
        "add-relu-mul",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, SHAPE)],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, SHAPE)],
        initializer_list,
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, onnx_file)
    return

def parse_onnx_file() -> TRTWrapperV1:
    """A wrapper whose network is the freshly parsed ONNX, ready to be edited."""
    tw = TRTWrapperV1(logger=trt.Logger.Severity.ERROR)
    parser = trt.OnnxParser(tw.network, tw.logger)
    if not parser.parse_from_file(str(onnx_file)):
        raise RuntimeError(f"Failed to parse {onnx_file}: {parser.get_error(0)}")
    return tw

def find_layer(network, name: str):
    """Locate a layer by name. The ONNX node names survive parsing, which is what makes this work."""
    for i in range(network.num_layers):
        if network.get_layer(i).name == name:
            return network.get_layer(i)
    raise KeyError(f"{name} not among {[network.get_layer(i).name for i in range(network.num_layers)]}")

def build_and_run(tw: TRTWrapperV1, tag: str):
    """Build the edited network, run it once, and report the engine's layer count."""
    tw.build()  # The parser already marked the outputs, so nothing is passed here
    tw.setup(input_data, b_print_io=False)
    tw.infer(b_print_io=False)
    output_name = tw.network.get_output(0).name
    output = tw.buffer[output_name][0]
    print(f"    {tag}: engine layers={tw.engine.num_layers}, {output_name}[0, 0, :4]={output.reshape(-1)[:4]}")
    return output

@case_mark
def case_walk_the_parsed_network():
    """What the parser actually produced, which is never quite the ONNX graph."""
    tw = parse_onnx_file()
    print(f"    3 ONNX nodes became {tw.network.num_layers} TensorRT layers:")
    for i in range(tw.network.num_layers):
        layer = tw.network.get_layer(i)
        input_name = [layer.get_input(j).name for j in range(layer.num_inputs)]
        output_name = [layer.get_output(j).name for j in range(layer.num_outputs)]
        print(f"        {i}: {layer.name:22s} {str(layer.type).split('.')[-1]:12s} {input_name} -> {output_name}")
    print(f"    network outputs: {[tw.network.get_output(i).name for i in range(tw.network.num_outputs)]}")
    print("    -> the ONNX node names survive (`node_add`, `node_relu`, `node_mul`), which is how a layer is")
    print("       found later; the extra CONSTANT / SHUFFLE layers are the parser broadcasting the scalars")
    return

@case_mark
def case_append_a_layer():
    """Add a layer after the last one: unmark the old output, build on it, mark the new one."""
    tw = parse_onnx_file()
    old_output = tw.network.get_output(0)
    tw.network.unmark_output(old_output)  # Without this the network would have two outputs

    layer = tw.network.add_unary(old_output, trt.UnaryOperation.NEG)
    layer.name = "appended_neg"
    layer.get_output(0).name = "y_negative"
    tw.network.mark_output(layer.get_output(0))

    output = build_and_run(tw, "with an appended NEG")
    assert np.allclose(output, -reference), "the appended layer did not do what it should"
    print("    -> `unmark_output` + `mark_output` move the boundary; the old output tensor stays a normal tensor")
    return

@case_mark
def case_bypass_a_layer():
    """Delete the `Relu`, which `INetworkDefinition` has no API for.

    The layer object cannot be removed. What is removable is its *use*: point the consumer at the
    producer's input instead, and the builder drops what nothing reads. The network still reports
    the same `num_layers` afterwards -- the proof that it is gone is in the numbers.
    """
    tw = parse_onnx_file()
    n_layer_before = tw.network.num_layers
    add_output = find_layer(tw.network, "node_add").get_output(0)
    find_layer(tw.network, "node_mul").set_input(0, add_output)  # Was the Relu's output
    print(f"    network layers before {n_layer_before}, after rewiring {tw.network.num_layers} (`node_relu` is still there, unused)")

    output = build_and_run(tw, "with the Relu bypassed")
    assert np.allclose(output, (input_data["x"] + SCALAR) * 2.0), "the bypass did not remove the Relu"
    assert output.min() < 0, "a negative value proves the Relu no longer runs"
    print(f"    -> negative values ({output.min():.1f}) appear where the Relu used to clamp at 0")
    return

@case_mark
def case_replace_a_layer_with_a_plugin():
    """Replace the `Add` with our own plugin, on a model TensorRT already runs perfectly.

    This is the interesting direction: nothing is broken, the plugin is wanted for other reasons (a
    fused variant, a numerical convention, a kernel the team owns). Parsing first and swapping
    afterwards avoids touching the ONNX and avoids the custom-op dance of
    `../../05-Plugin/ONNXParserWithPlugin/`.

    The replacement is exact, and it costs the fusion: TensorRT had folded Add+Relu+Mul into one
    kernel, and a plugin cannot be fused into that.
    """
    if not plugin_file.exists():
        print(f"    building {plugin_file.name}")
        subprocess.run(["make", "-j"], cwd=plugin_file.parent, check=True, stdout=subprocess.DEVNULL)

    # Load the library and build the plugin object BEFORE any engine is built or released.
    # Measured on TensorRT 11.1.0.106: with the library loaded, building an engine and then letting
    # it go (rebinding the variable is enough) makes the next
    # `trt.get_plugin_registry().get_creator(...)` segfault, reproducibly and with no Python
    # traceback. Holding on to the wrappers and creating the plugin up front avoids the whole
    # question. `../GreenContext/` hits the mirror image of this: destroy the context first and
    # TensorRT's destructors crash at exit.
    load_plugin_files([plugin_file])
    plugin_info = dict(
        name="AddScalar",
        version="1",
        namespace="",
        argument_dict=dict(scalar=np.array([SCALAR], dtype=np.float32)),
        number_input_tensor=1,
        number_input_shape_tensor=0,
        plugin_api_version="3",
    )
    plugin = get_plugin(plugin_info)

    wrapper_list, result_list = [], []
    for tag, b_use_plugin in [("as parsed        ", False), ("Add -> AddScalar ", True)]:
        tw = parse_onnx_file()
        wrapper_list.append(tw)  # Kept alive on purpose, see the comment above
        if b_use_plugin:
            layer = tw.network.add_plugin_v3([tw.network.get_input(0)], [], plugin)
            layer.name = "AddScalarPluginLayer"
            layer.get_output(0).name = "a_from_plugin"
            # Rewire the consumer. `node_add` is now unused and the builder drops it.
            find_layer(tw.network, "node_relu").set_input(0, layer.get_output(0))

        output = build_and_run(tw, tag)
        # Time the kernels only: `TRTWrapperV1.infer` also copies host <-> device, and those copies
        # are ~150x the kernel time here, which would hide the effect completely.
        for _ in range(10):
            tw.context.execute_async_v3(0)
        cudart.cudaDeviceSynchronize()
        start_time = time.perf_counter()
        for _ in range(200):
            tw.context.execute_async_v3(0)
        cudart.cudaDeviceSynchronize()
        elapsed = (time.perf_counter() - start_time) * 1000 / 200
        print(f"    {tag}: kernel time {elapsed:.4f} ms")
        result_list.append((output, elapsed))

    assert np.array_equal(result_list[0][0], result_list[1][0]), "the plugin changed the numbers"
    print(f"    -> output is bit-identical, and the lost fusion costs "
          f"{result_list[1][1] / result_list[0][1]:.2f}x: one pass over memory became two")
    return

if __name__ == "__main__":
    build_onnx_file()

    case_walk_the_parsed_network()
    case_append_a_layer()
    case_bypass_a_layer()
    case_replace_a_layer_with_a_plugin()

    print("\nFinish")
