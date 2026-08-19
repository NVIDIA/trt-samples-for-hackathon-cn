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
"""Checking a TensorRT plugin against a CPU reference with `PluginRefRunner`.

`PluginRefRunner` walks an ONNX-GraphSurgeon graph and evaluates every node with
a NumPy function looked up in `OP_REGISTRY`. Two things about that sentence do
not match the name:

* it has nothing to do with plugins -- it never loads one, and the registry
  ships with three ops (`Identity`, `InstanceNormalization`,
  `MeanVarianceNormalization`);
* it runs the *whole* graph, so a single unregistered node -- `Mul` counts --
  stops it.

So the real workflow is: register a reference for your custom op, cut the plugin
node out into its own graph, and compare that against TensorRT. All three steps
are here, against the cookbook's `AddScalar` plugin from
`05-Plugin/ONNXParserWithPlugin/`.

The thing that will actually cost you an afternoon is
`case_loading_the_plugin_is_where_it_breaks`: Polygraphy's `LoadPlugins` calls
`ctypes.CDLL`, which does not register an `IPluginCreatorV3One`. It reports
success and the ONNX parser fails two layers later with a message about names
and versions.
"""

import subprocess

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
from polygraphy.backend.pluginref import PluginRefRunner
from polygraphy.backend.pluginref.references import OP_REGISTRY, register
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, LoadPlugins, NetworkFromOnnxPath, Profile, TrtRunner
from polygraphy.comparator import Comparator, CompareFunc
from polygraphy.logger import G_LOGGER

from tensorrt_cookbook import case_mark, cookbook_path

G_LOGGER.module_severity = G_LOGGER.ERROR

plugin_dir = cookbook_path("05-Plugin", "ONNXParserWithPlugin")
plugin_file = plugin_dir / "AddScalarPlugin.so"
addscalar_onnx = str(cookbook_path("00-Data", "model", "model-addscalar.onnx"))
mnist_onnx = str(cookbook_path("00-Data", "model", "model-trained.onnx"))
mixed_onnx = "model-mixed.onnx"
subgraph_onnx = "model-subgraph.onnx"
scalar5_onnx = "model-scalar5.onnx"

def ensure_plugin() -> None:
    """The .so is a build artifact of `05-Plugin/ONNXParserWithPlugin/`, not a checked-in file."""
    if not plugin_file.exists():
        subprocess.run(["make", "-C", str(plugin_dir), "build"], check=True, capture_output=True)

def load_addscalar_graph() -> gs.Graph:
    return gs.import_onnx(onnx.load(addscalar_onnx))

def build_mixed_graph() -> gs.Graph:
    """`y = (x * 2) + 1`, where the `+ 1` is the custom op and the `* 2` is not."""
    x = gs.Variable("x", np.float32, (4, ))
    scaled = gs.Variable("scaled", np.float32, (4, ))
    y = gs.Variable("y", np.float32, (4, ))
    two = gs.Constant("two", np.array([2.0], dtype=np.float32))
    nodes = [
        gs.Node("Mul", inputs=[x, two], outputs=[scaled]),
        gs.Node("AddScalar", attrs={"scalar": 1.0}, inputs=[scaled], outputs=[y]),
    ]
    return gs.Graph(nodes=nodes, inputs=[x], outputs=[y], opset=17)

def register_correct_reference() -> None:
    """`register` is *not* re-exported by `polygraphy.backend.pluginref`."""

    @register("AddScalar")
    def run_add_scalar(attrs, x):
        # The signature is (attrs, *inputs) and the return value is a *list*,
        # one entry per output of the node.
        return [x + attrs["scalar"]]

@case_mark
def case_the_registry_is_the_whole_story() -> None:
    """Three ops ship with it, and every node of the graph has to be in there.

    Neither half of the name survives contact: no plugin is loaded anywhere in
    this case, and `Conv` -- an ordinary ONNX op -- is as unsupported as a custom
    one. The runner is a small NumPy interpreter with a three-entry op table.
    """
    print(f"    ops with a reference implementation: {sorted(OP_REGISTRY)}")

    graph = load_addscalar_graph()
    print(f"    the AddScalar model is one node : {[node.op for node in graph.nodes]}")
    try:
        with PluginRefRunner(graph) as runner:
            runner.infer({"inputT0": np.arange(4, dtype=np.float32)})
    except Exception as e:
        print(f"    running it unregistered -> {type(e).__name__}: {str(e).splitlines()[0]}")

    mnist = gs.import_onnx(onnx.load(mnist_onnx))
    print(f"    a normal MNIST model uses       : {sorted({node.op for node in mnist.nodes})}")
    try:
        with PluginRefRunner(mnist) as runner:
            runner.infer({"x": np.zeros((1, 1, 28, 28), dtype=np.float32)})
    except Exception as e:
        print(f"    running that              -> {type(e).__name__}: {str(e).splitlines()[0]}")
    print("    so this is not a runner you point at a model -- it is one you point at a single op")
    return

@case_mark
def case_loading_the_plugin_is_where_it_breaks() -> None:
    """`LoadPlugins` uses `ctypes.CDLL`, which does not register a V3 creator.

    `ctypes.CDLL` was how plugins were loaded when `REGISTER_TENSORRT_PLUGIN`
    ran at static-initialisation time. An `IPluginCreatorV3One` is picked up by
    `IPluginRegistry::loadLibrary` instead, and `trt.get_plugin_registry()
    .load_library(path)` is the Python spelling of that.

    The failure mode is what makes it expensive: `LoadPlugins` logs
    "Loading plugin library: ..." and returns cleanly, and the complaint arrives
    two layers away from the cause, in the ONNX parser:

        Plugin not found, are the plugin name, version, and namespace correct?

    which sends you to check the name, version and namespace -- all of which are
    fine.
    """
    ensure_plugin()
    registry = trt.get_plugin_registry()
    creator_names = lambda: {creator.name for creator in registry.all_creators}

    print(f"    creators registered at startup            : {len(creator_names())}, AddScalar present: {'AddScalar' in creator_names()}")

    LoadPlugins(obj=None, plugins=[str(plugin_file)])()  # this is what Polygraphy does
    print(f"    after Polygraphy's LoadPlugins (ctypes)   : {len(creator_names())}, AddScalar present: {'AddScalar' in creator_names()}")

    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.ERROR), "")
    print(f"    after trt.init_libnvinfer_plugins          : {len(creator_names())}, AddScalar present: {'AddScalar' in creator_names()}")

    registry.load_library(str(plugin_file))
    print(f"    after registry.load_library                : {len(creator_names())}, AddScalar present: {'AddScalar' in creator_names()}")
    print("    only the third one works; the first two report success")

    # The CLI's `--plugins` flag is the same `ctypes.CDLL` path, so it cannot
    # load this plugin either -- from a fresh process, with the flag set.
    command = ["polygraphy", "run", addscalar_onnx, "--trt", "--plugins", str(plugin_file), "--input-shapes", "inputT0:[4]"]
    result = subprocess.run(command, capture_output=True, text=True)
    parser_error = [line for line in result.stderr.splitlines() + result.stdout.splitlines() if "Plugin not found" in line]
    print(f"    `polygraphy run --plugins ...` exit code {result.returncode}: {parser_error[0].strip() if parser_error else 'succeeded'}")
    return

@case_mark
def case_comparing_the_plugin_against_a_reference() -> None:
    """The workflow the runner exists for, end to end.

    `register` has to be imported from `polygraphy.backend.pluginref.references`
    -- it is not in `polygraphy.backend.pluginref`'s exports, so the one function
    you need to use this feature at all is the one that looks private.

    The decorated function receives `(attrs, *inputs)`, where `attrs` is the
    node's ONNX attributes and constant inputs arrive as NumPy arrays already.
    It must return a list with one entry per node output.
    """
    ensure_plugin()
    trt.get_plugin_registry().load_library(str(plugin_file))
    register_correct_reference()
    print(f"    registry after registering: {sorted(OP_REGISTRY)}")

    profile = Profile().add("inputT0", (1, ), (4, ), (8, ))
    build = EngineFromNetwork(NetworkFromOnnxPath(addscalar_onnx), config=CreateConfig(profiles=[profile]))
    feed = {"inputT0": np.arange(4, dtype=np.float32)}

    results = Comparator.run([TrtRunner(build), PluginRefRunner(load_addscalar_graph())], data_loader=[feed])
    for name, iterations in results.items():
        print(f"    {name.split('-N')[0]:<18}: {list(iterations[0].values())[0]}")
    print(f"    accuracy comparison passes: {bool(Comparator.compare_accuracy(results))}")
    return

@case_mark
def case_what_it_looks_like_when_they_disagree() -> None:
    """A passing comparison against a correct reference proves very little.

    The failure this catches most often is not arithmetic, it is an ignored
    attribute -- the plugin reads `scalar` from the ONNX node and the reference
    hard-codes it, or vice versa. Registering a reference that drops the
    attribute reproduces it exactly, and `CompareFunc.simple` reports the size of
    the disagreement.

    Note that re-registering an op silently replaces the previous entry:
    `OP_REGISTRY` is a plain dict and `register` does not check.
    """
    ensure_plugin()
    trt.get_plugin_registry().load_library(str(plugin_file))

    @register("AddScalar")
    def wrong_reference(attrs, x):
        return [x + 1.0]  # the model says scalar=1.0, so this one still agrees

    profile = Profile().add("inputT0", (1, ), (4, ), (8, ))
    build = EngineFromNetwork(NetworkFromOnnxPath(addscalar_onnx), config=CreateConfig(profiles=[profile]))
    feed = {"inputT0": np.arange(4, dtype=np.float32)}

    results = Comparator.run([TrtRunner(build), PluginRefRunner(load_addscalar_graph())], data_loader=[feed])
    print(f"    hard-coded 1.0, model attribute is 1.0 -> passes: {bool(Comparator.compare_accuracy(results))}")

    # Same reference, a model whose attribute is not 1.0. Nothing about the
    # reference changed; it was always wrong, and only this input reveals it.
    graph = load_addscalar_graph()
    graph.nodes[0].attrs["scalar"] = 5.0
    onnx.save(gs.export_onnx(graph), scalar5_onnx)

    build = EngineFromNetwork(NetworkFromOnnxPath(scalar5_onnx), config=CreateConfig(profiles=[profile]))
    results = Comparator.run([TrtRunner(build), PluginRefRunner(graph)], data_loader=[feed])
    passed = bool(Comparator.compare_accuracy(results, compare_func=CompareFunc.simple(atol=1e-6)))
    outputs = [list(iterations[0].values())[0] for iterations in results.values()]
    print(f"    same reference, model attribute is 5.0 -> passes: {passed}")
    print(f"    plugin says {outputs[0]}, reference says {outputs[1]}, max |diff| = {np.max(np.abs(outputs[0] - outputs[1])):.1f}")
    print("    the bug was in the reference the whole time -- the first comparison just could not see it")

    register_correct_reference()  # put the registry back
    return

@case_mark
def case_cutting_the_plugin_out_of_a_real_model() -> None:
    """Whole-graph evaluation means a real model needs a subgraph first.

    A model with one custom op among ordinary ones cannot be handed to
    `PluginRefRunner` -- `Mul` has no reference either. `onnx_graphsurgeon`
    reassigns `inputs`/`outputs` and `cleanup()` drops everything else, leaving a
    graph with just the plugin node, which both runners will accept.

    The trap in the `gs` part: `graph.copy()` produces *new* tensor objects, so
    the `Variable` you are holding from the original graph is not the one in the
    copy. Look the tensor up by name with `copy.tensors()`, or `cleanup()` raises
    "Encountered a node not in the graph".
    """
    ensure_plugin()
    trt.get_plugin_registry().load_library(str(plugin_file))
    register_correct_reference()

    graph = build_mixed_graph()
    feed = {"x": np.arange(4, dtype=np.float32)}
    print(f"    full graph nodes: {[node.op for node in graph.nodes]}")
    try:
        with PluginRefRunner(graph) as runner:
            runner.infer(feed)
    except Exception as e:
        print(f"    PluginRefRunner on the full graph -> {str(e).splitlines()[0]}")

    subgraph = graph.copy()
    tensors = subgraph.tensors()  # by name -- copy() made new objects
    subgraph.inputs = [tensors["scaled"]]
    subgraph.outputs = [tensors["y"]]
    subgraph.cleanup()
    onnx.save(gs.export_onnx(subgraph), subgraph_onnx)
    print(f"    subgraph nodes  : {[node.op for node in subgraph.nodes]}, inputs {[t.name for t in subgraph.inputs]}")

    subgraph_feed = {"scaled": feed["x"] * 2.0}
    build = EngineFromNetwork(NetworkFromOnnxPath(subgraph_onnx), config=CreateConfig())
    results = Comparator.run([TrtRunner(build), PluginRefRunner(subgraph)], data_loader=[subgraph_feed])
    print(f"    plugin vs reference on the isolated node: {bool(Comparator.compare_accuracy(results))}")

    onnx.save(gs.export_onnx(graph), mixed_onnx)
    with TrtRunner(EngineFromNetwork(NetworkFromOnnxPath(mixed_onnx), config=CreateConfig())) as runner:
        print(f"    and the full model still runs in TensorRT: {runner.infer(feed)['y']}")
    print("    the CLI spelling of all this is `polygraphy run model.onnx --trt --pluginref`")
    return

if __name__ == "__main__":
    case_the_registry_is_the_whole_story()
    case_loading_the_plugin_is_where_it_breaks()
    case_comparing_the_plugin_against_a_reference()
    case_what_it_looks_like_when_they_disagree()
    case_cutting_the_plugin_out_of_a_real_model()

    print("\nFinish")
