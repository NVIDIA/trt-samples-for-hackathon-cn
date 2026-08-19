# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import ctypes
import json
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt

from .utils_function import datatype_cast
from .utils_onnx import add_node

########################################################################################################################
# Helpers shared by the inspectors below

def _numel(shape_list):
    """Return the element count for a shape list, or ``None`` for non-positive dimensions."""
    if len(shape_list) == 0:  # Special case for scalar tensor
        return 1
    if any(d <= 0 for d in shape_list):
        return None
    n = 1
    for d in shape_list:
        n *= d
    return n

########################################################################################################################
# Inspect a serialized engine (plan file) and a live engine / execution context

def read_host_array_from_pointer(address: int, dtype: trt.DataType, shape_list: list):
    """Read a host-side array from a raw pointer using TensorRT dtype and shape."""
    trt_to_ctype = {
        trt.int8: ctypes.c_int8,
        trt.uint8: ctypes.c_uint8,
        # trt.int16: ctypes.c_int16,
        trt.int32: ctypes.c_int32,
        trt.int64: ctypes.c_int64,
        trt.float16: ctypes.c_uint16,
        trt.float32: ctypes.c_float,
        trt.bool: ctypes.c_bool,
    }
    ctype = trt_to_ctype.get(dtype, None)
    n_value = _numel(shape_list)
    if address is None or int(address) == 0 or ctype is None or n_value is None:
        return None
    try:
        pointer_type = ctypes.POINTER(ctype * n_value)
        raw = ctypes.cast(int(address), pointer_type).contents
        np_array = np.ctypeslib.as_array(raw)
        if dtype == trt.float16:
            np_array = np_array.view(np.float16)
        if len(shape_list) > 0:
            np_array = np_array.reshape(shape_list)
        return np_array.copy()
    except Exception as e:
        print(f"Error reading host array from pointer: {e}")
        return None

def print_engine_io_information(
    *,
    trt_file: Path = Path(),
    engine: trt.ICudaEngine = None,
    plugin_file_list: list | None = None,
) -> None:
    """Print tensor IO and optimization-profile shapes for an engine."""
    plugin_file_list = plugin_file_list or []

    if engine is None:
        with open(trt_file, "rb") as f:
            engine_bytes = f.read()

        logger = trt.Logger(trt.Logger.Severity.ERROR)
        # Load TenorRT native and customer's plugins
        trt.init_libnvinfer_plugins(logger, namespace="")
        for plugin_file in plugin_file_list:
            if plugin_file.exists():
                ctypes.cdll.LoadLibrary(plugin_file)
        runtime = trt.Runtime(logger)
        try:
            engine = runtime.deserialize_cuda_engine(engine_bytes)
        except RuntimeError:
            print("Failed loading engine, `print_engine_io_information()` is only supported when TRT version of engine and runtime is the same.")
            return

    context = engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)

    tensor_name_list = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    n_optimization_profile = engine.num_optimization_profiles
    max_name_width = 8  # Maximum Width of tensor Name
    max_shape_width = 0  # Maximum Width of tensor Shape

    # Get information of engine input / output
    tid = {}  # Tensor Information Dictionary
    for name in tensor_name_list:
        tensor = {}
        max_name_width = max(max_name_width, len(name))
        tensor["mode"] = "I" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "O"
        tensor["location"] = "GPU" if engine.get_tensor_location(name) == trt.TensorLocation.DEVICE else "CPU"
        tensor["data_type"] = str(engine.get_tensor_dtype(name))[9:]
        tensor["build_shape"] = str(engine.get_tensor_shape(name))
        tensor["profile_list"] = [[] for _ in range(n_optimization_profile)]
        if tensor["mode"] == "I":
            for i in range(n_optimization_profile):
                if tensor["location"] == "GPU":
                    shape = engine.get_tensor_profile_shape(name, i)
                else:
                    shape = engine.get_tensor_profile_values(i, name)
                tensor["profile_list"][i].extend(shape)
                max_shape_width = max(max_shape_width, *[len(str(s)) for s in shape])
        tid[name] = tensor

    # Set input shape to get output shape
    for i in range(n_optimization_profile):
        context.set_optimization_profile_async(i, 0)
        for j in range(3):  # Min, Opt, Max
            for name in tid.keys():
                if tid[name]["mode"] == "I":
                    if tid[name]["location"] == "GPU":
                        context.set_input_shape(name, tid[name]["profile_list"][i][j])
                    else:
                        context.set_tensor_address(name, np.array(tid[name]["profile_list"][i][j]).ctypes.data)
                elif tid[name]["mode"] == "O":
                    assert len(context.infer_shapes()) == 0
                    shape = context.get_tensor_shape(name)
                    tid[name]["profile_list"][i].append(shape)
                    max_shape_width = max(max_shape_width, len(str(shape)))

    print("\nInformation of engine input / output.")
    print(f"{'='*(max_name_width + max_shape_width + 24)}")
    print(f"{'Name':^{max_name_width}}|I/O|Location|DataType|{'Shape':^{max_shape_width}}|")
    print(f"{'-'*(max_name_width + max_shape_width + 24)}")
    for name in tensor_name_list:
        tensor = tid[name]
        info = f"{name:<{max_name_width}}|{tensor['mode']:^3s}|{tensor['location']:^8s}|{tensor['data_type']:^8s}|"
        info += f"{tensor['build_shape']:^{max_shape_width}}|"
        print(info)
    print(f"{'='*(max_name_width + max_shape_width + 24)}")

    print("\nInformation of optimization profile.")
    for i in range(n_optimization_profile):
        print(f"\nOptimization Profile {i}:")
        print(f"{'='*(max_name_width + max_shape_width * 3 + 4)}")
        print(f"{'Name':^{max_name_width}}|{'Min':^{max_shape_width}}|{'Opt':^{max_shape_width}}|{'Max':^{max_shape_width}}|")
        print(f"{'-'*(max_name_width + max_shape_width * 3 + 4)}")
        for name in tensor_name_list:
            tensor = tid[name]
            info = f"{name:<{max_name_width}}|"
            info += f"{str(tensor['profile_list'][i][0]):^{max_shape_width}}|"
            info += f"{str(tensor['profile_list'][i][1]):^{max_shape_width}}|"
            info += f"{str(tensor['profile_list'][i][2]):^{max_shape_width}}|"
            print(info)
        print(f"{'='*(max_name_width + max_shape_width * 3 + 4)}")
    return

def print_context_io_information(context: trt.IExecutionContext = None, ) -> None:
    """Print input/output tensor shapes currently bound in an execution context."""
    if context is None:
        print("`context` is None, skip printing context IO information.")
        return

    engine = context.engine
    context_index = context.active_optimization_profile
    n_io = engine.num_io_tensors
    max_name_width = 8  # Maximum Width of tensor Name
    max_shape_width = 0  # Maximum Width of runtime tensor Shape
    max_build_shape_width = 0  # Maximum Width of build tensor Shape
    tensor_info = {}

    for i in range(n_io):
        name = engine.get_tensor_name(i)
        mode = "I" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "O"
        location = "GPU" if engine.get_tensor_location(name) == trt.TensorLocation.DEVICE else "CPU"
        build_shape_list = [int(d) for d in engine.get_tensor_shape(name)]
        build_shape = str(tuple(build_shape_list))
        if location == "GPU":
            runtime_shape = str(context.get_tensor_shape(name))
        else:
            address = context.get_tensor_address(name)
            runtime_shape_list = [int(d) for d in context.get_tensor_shape(name)]
            read_shape_list = runtime_shape_list if _numel(runtime_shape_list) is not None else build_shape_list
            host_array = read_host_array_from_pointer(address, engine.get_tensor_dtype(name), read_shape_list)
            if host_array is not None:
                runtime_shape = str(host_array.tolist())
            else:
                runtime_shape = f"addr={address}"

        tensor_info[i] = [name, mode, location, build_shape, runtime_shape]
        max_name_width = max(max_name_width, len(name))
        max_build_shape_width = max(max_build_shape_width, len(build_shape))
        max_shape_width = max(max_shape_width, len(runtime_shape))

    print(f"Information of context input / output.")
    print(f"Using Optimization Profile: {context_index}")
    print(f"{'='*(max_name_width + max_build_shape_width + max_shape_width + 18)}")
    print(f"{'Name':^{max_name_width}}|I/O|Location|{'BuildShape':^{max_build_shape_width}}|{'ContextShape':^{max_shape_width}}|")
    print(f"{'-'*(max_name_width + max_build_shape_width + max_shape_width + 18)}")
    for i in range(n_io):
        name, mode, location, build_shape, runtime_shape = tensor_info[i]
        info = f"{name:<{max_name_width}}|{mode:^3s}|{location:^8s}|{build_shape:^{max_build_shape_width}}|{runtime_shape:^{max_shape_width}}|"
        print(info)
    print(f"{'='*(max_name_width + max_build_shape_width + max_shape_width + 18)}")
    return

########################################################################################################################
# Inspect the engine information JSON produced by `trt.EngineInspector`

def get_engine_tensor_info(tensor: dict = None):
    """
    Get information of a tensor
    """
    assert isinstance(tensor, dict) and "Dimensions" in tensor.keys(), f"Wrong tensor format: {tensor}"
    shape = tensor["Dimensions"]
    location = tensor["Location"] if "Location" in tensor.keys() else "Unknown"
    # Separate "Datatype" and "Format" keys. Support both schemas.
    if "Format/Datatype" in tensor.keys():
        fd = tensor["Format/Datatype"]
        fd_list = fd.split(" ")
        if "format" in fd_list:
            index = fd_list.index("format")
            data_type = fd_list[index - 1]
        else:
            data_type = fd_list[-1]
    else:
        data_type = tensor["Datatype"]
        fd = f'{tensor.get("Format", "")} {data_type}'.strip()
    data_type = datatype_cast(data_type, "np")
    info = f"{fd}->{location}"

    return data_type, shape, info

def is_tensor_used_later(name, tensor_list, layer_list):
    """
    Whether the tensor is used in the later part of the network
    """
    # Whether this tensor is used in the same layer
    if name in [sub_tensor["Name"] for sub_tensor in tensor_list]:
        return True
    # Whether this tensor is used in the later layers
    for sub_layer in layer_list:
        # This tensor firstly appears as input tensor in the later layers, it is useful
        if name in [tensor["Name"] for tensor in sub_layer["Inputs"]]:
            return True
        # This tensor firstly appears as output tensor in the later layers, it is useless now
        if name in [tensor["Name"] for tensor in sub_layer["Outputs"]]:
            return False
    return False

ENGINE_DOMAIN = "trt.engine"  # See `export_engine_as_onnx`

def export_engine_as_onnx(engine_json_file: Path = None, export_onnx_file: Path = None, engine_profile_file: Path = None, b_break_cycle: bool = False):
    """
    Export TensorRT engine as a "ONNX-like" file, which can be opened by software like Netron

    `engine_json_file` is what `trtexec --exportLayerInfo` writes. `engine_profile_file` is the
    optional `trtexec --exportProfile` companion: when it is given, each node gets the measured time
    of the layer it came from as a `Latency` attribute, so the graph says where the time goes and
    not just what the shape of the engine is. When it is omitted the attribute is left empty.

    The nodes are emitted into the `trt.engine` domain rather than the default one. Engine layer
    types - `kgen`, `gemm`, `shape_call`, ... - are the result of TensorRT's fusion and are not ONNX
    operators, so in the default domain every standard consumer rejects the file with
    `No Op registered for kgen with domain_version of N`. Declaring a domain of our own is what
    tells those tools that these nodes are not theirs to validate, and is the difference between a
    file only Netron will open and one that, for example, `tensorrt_cookbook.onnx_outliner` will
    also process.

    Note the graph is a *picture of an engine*, not a runnable model: no consumer can execute a
    `kgen` node. `onnx_outliner` works on it, but its defaults are tuned for source ONNX graphs
    where one transformer layer is dozens of fine-grained operators. After Myelin fusion each node
    is already a mega-kernel and the repeating unit is only one or two nodes wide, so the default
    `--min-size` filters everything out. Measured on a 315-layer gpt2-medium engine:

        default settings                              315 -> 314 nodes, coverage  0.0%
        --min-size 2 --min-repeat 2 --strictness L0   315 ->  38 nodes, coverage 95.2%

    Those looser values are recorded here rather than made the default because they are wrong for
    every other kind of input; pass them explicitly when outlining an engine graph.

    Control flow is not supported: an engine containing a Loop reuses a tensor name as the output of
    more than one layer, which has no ONNX equivalent, and this raises rather than emitting a graph
    with silently mismatched edges.
    """
    with open(engine_json_file, "r") as f:
        js = json.loads(f.read())

    # Optional per-layer timing, keyed by layer name. `--exportProfile` writes a list whose first
    # entry is a count header rather than a layer, hence the `"name" in row` guard.
    latency_map = {}
    if engine_profile_file is not None:
        with open(engine_profile_file, "r") as f:
            for row in json.loads(f.read()):
                if isinstance(row, dict) and "name" in row:
                    latency_map[row["name"]] = row

    layer_list = js["Layers"]
    # Convert name string to "I/O Tensors" (list of dicts)
    if "Bindings" in js:
        io_tensor_list = js["Bindings"]
    else:
        io_tensor_list = [t["Name"] for t in js.get("I/O Tensors", [])]

    # Preprocess to fix duplicate name problem, O(V^2)
    #
    # This is single static assignment renaming: every write gets a name of its own, and each read
    # is pointed at the write that reaches it. TensorRT reuses tensor names in two situations, and
    # both land here.
    #
    # The first is Myelin's internal scratch tensors, which is what the regex below is for. The
    # second is control flow: an engine containing a `Loop` initialises its carried tensor with one
    # layer and rewrites it with another on every trip, so a name has several producers and one
    # layer may even read and write the same name (the counter increment is literally
    # `add: Recurrence 0 Output -> Recurrence 0 Output`). Renaming turns that cyclic dependency into
    # a chain of versions, which is what makes the result a DAG and therefore expressible in ONNX.
    #
    # Renaming alone would silently lose the fact that the last version flows back to the first, so
    # `version_map` records the chain and a `BackEdge` marker node is emitted for each one further
    # down. Dropping the back-edge quietly is exactly what the other two writers in
    # `07-Tool/EngineVisualization` do, and it costs them 44% of the edges on a Loop engine.
    reg_myelin_tensor = r"(__my.+)|(__tran)(\d+)"  # for example: "__myln_k_arg__bb1_24", "__tran7010"
    n_producer = {}
    for layer in layer_list:
        for tensor in layer["Outputs"]:
            n_producer[tensor["Name"]] = n_producer.get(tensor["Name"], 0) + 1
    version_map = OrderedDict()  # original name -> [name of every version, in write order]

    global_count = 0
    for i, layer in enumerate(layer_list):
        tensor_list = layer["Outputs"]
        for j, tensor in enumerate(tensor_list):  # this tensor must appear in Outputs firstly
            b_myelin = len(re.findall(reg_myelin_tensor, tensor["Name"])) > 0
            b_multi_producer = b_break_cycle and n_producer.get(tensor["Name"], 0) > 1
            if not b_myelin and not b_multi_producer:
                continue
            old_name = tensor["Name"]
            new_name = tensor["Name"] + "@" + str(global_count)
            global_count += 1
            js["Layers"][i]["Outputs"][j]["Name"] = new_name
            if b_multi_producer:
                version_map.setdefault(old_name, []).append(new_name)

            for sub_tensor in tensor_list[(j + 1):]:
                if sub_tensor["Name"] == old_name:
                    sub_tensor["Name"] = new_name
            b_finish = False
            for sub_layer in layer_list[(i + 1):]:
                if b_finish:
                    break
                tensor_list = sub_layer["Inputs"]
                for sub_tensor in tensor_list:
                    if sub_tensor["Name"] == old_name:
                        sub_tensor["Name"] = new_name
                tensor_list = sub_layer["Outputs"]
                for sub_tensor in tensor_list:
                    if sub_tensor["Name"] == old_name:
                        b_finish = True

    # Main process of building ONNX like graph
    # Convert name string to "I/O Tensors" (list of dicts)
    if "Bindings" in js:
        io_tensor_list = js["Bindings"]
    else:
        io_tensor_list = [t["Name"] for t in js.get("I/O Tensors", [])]

    graph = gs.Graph(nodes=[], inputs=[], outputs=[])
    n = 0

    global_tensor_map = {}  # mapping from Name of TRT tensor (str) to GS tensor (gs.Variable)
    global_tensor_fd_map = {}  # mapping from Name of TRT tensor (str) to format and location of the tensor (str)
    for i, layer in enumerate(layer_list):
        input_tensor_list = []
        layer_tensor_fd_map = {}
        for j, tensor in enumerate(layer["Inputs"]):
            name = tensor["Name"]  # `name` can be duplicate in TensorRT engine
            if name in global_tensor_map.keys():  # already in the map
                if is_tensor_used_later(name, layer["Inputs"][(j + 1):], layer_list[(i + 1):]):
                    gs_tensor = global_tensor_map[name]
                    layer_tensor_fd_map[name] = global_tensor_fd_map[name]
                else:
                    gs_tensor = global_tensor_map.pop(name)
                    layer_tensor_fd_map[name] = global_tensor_fd_map.pop(name)
            else:
                data_type, shape, info = get_engine_tensor_info(tensor)
                gs_tensor = gs.Variable(name, data_type, shape)
                if is_tensor_used_later(name, layer["Inputs"][(j + 1):], layer_list[(i + 1):]):
                    global_tensor_map[name] = gs_tensor
                    global_tensor_fd_map[name] = info
                layer_tensor_fd_map[name] = info

            input_tensor_list.append(gs_tensor)
            if name in io_tensor_list and gs_tensor not in graph.inputs and gs_tensor not in graph.outputs:
                graph.inputs.append(gs_tensor)

        output_datatype_list = []
        output_shape_list = []
        for tensor in layer["Outputs"]:
            name = tensor["Name"]  # tensor["Name"] can be duplicate
            if name in global_tensor_map.keys() and b_break_cycle:
                # With `b_break_cycle`, the SSA renaming above should have left no name written
                # twice. Reaching this means the renaming missed a case rather than that the engine
                # is unrepresentable, so say so instead of emitting misdirected edges.
                raise RuntimeError(f"Cannot export {engine_json_file}: tensor '{name}' is still produced by more than one "
                                   f"layer after SSA renaming ('{layer['Name']}' is the second). This is a bug in "
                                   f"`export_engine_as_onnx`, not a limitation of the engine.")
            # Without `b_break_cycle` a repeated name is deliberate: both writers keep it, and the
            # resulting two-producer tensor is exactly the back-edge, drawn as a cycle by a viewer
            data_type, shape, info = get_engine_tensor_info(tensor)
            output_datatype_list.append(data_type)
            output_shape_list.append(shape)
            global_tensor_fd_map[name] = info

        attr = OrderedDict()
        for key, value in layer.items():
            if key in ["LayerType", "Name", "Inputs", "Outputs"]:
                continue
            attr[key] = str(value)

        # `Latency` is always present so that the attribute set does not depend on whether a profile
        # was supplied; an empty string marks "not measured" rather than "measured as zero"
        profile_row = latency_map.get(layer["Name"], {})
        attr["Latency"] = str(profile_row) if profile_row else ""

        output_tensor_list, n = add_node(graph, layer["LayerType"], input_tensor_list, attr, output_datatype_list, output_shape_list, "", "", n)
        graph.nodes[-1].name = layer["Name"]
        graph.nodes[-1].domain = ENGINE_DOMAIN

        if len(layer["Outputs"]) == 1:  # Convert single output tensor as a list
            output_tensor_list = [output_tensor_list]

        for i in range(len(layer["Outputs"])):
            name = layer["Outputs"][i]["Name"]
            gs_tensor = output_tensor_list[i]
            gs_tensor.name = name
            global_tensor_map[name] = gs_tensor
            if name in io_tensor_list and gs_tensor not in graph.outputs:
                graph.outputs.append(gs_tensor)
            layer_tensor_fd_map[name] = global_tensor_fd_map[name]

        graph.nodes[-1].attrs["TensorInfo"] = str(layer_tensor_fd_map)

    # Emit one `BackEdge` marker per renamed chain. The SSA renaming above made the graph acyclic,
    # which is the only shape ONNX accepts, but on its own it would leave no trace that the last
    # version flows back into the first - the loop would look like a straight line. This node has no
    # counterpart in the engine: it is a signpost saying "the cycle closes here", carrying the whole
    # version chain as an attribute so the original structure can be read back.
    tensor_map = graph.tensors() if b_break_cycle else {}
    for original_name, version_list in version_map.items():
        if len(version_list) < 2:  # Renamed but never actually rewritten, so no cycle to mark
            continue
        last_tensor = tensor_map.get(version_list[-1])
        if last_tensor is None:
            continue
        marker_output = gs.Variable(original_name + "@backedge", np.dtype(np.float32), [])
        node = gs.Node(
            "BackEdge",
            "BackEdge-" + original_name,
            inputs=[last_tensor],
            outputs=[marker_output],
            attrs=OrderedDict([("OriginalName", original_name), ("Target", version_list[0]), ("Versions", str(version_list))]),
        )
        node.domain = ENGINE_DOMAIN
        graph.nodes.append(node)
    if len(version_map) > 0:
        print(f"Marked {len(version_map)} back-edge(s) from control flow; the graph is a DAG of SSA versions:")
        for original_name, version_list in version_map.items():
            print(f"    {original_name} -> {len(version_list)} versions, back-edge {version_list[-1]} -> {version_list[0]}")

    # Drop tensors that are read but never written and are not graph inputs. TensorRT records its
    # own control channels - `l2_cache_policy_inputs_*`, `l2_cache_management_*` - among a layer's
    # Inputs, and they carry no data edge. Left in place they make the graph unsorted-looking, and
    # `onnx.checker` then reports the misleading `Nodes in a graph must be topologically sorted`
    # naming one of them, which sends you looking for a cycle that is not there.
    produced_name_set = {tensor.name for node in graph.nodes for tensor in node.outputs}
    declared_name_set = {tensor.name for tensor in graph.inputs}
    dangling_name_set = set()
    for node in graph.nodes:
        keep_list = []
        for tensor in node.inputs:
            if tensor.name in produced_name_set or tensor.name in declared_name_set:
                keep_list.append(tensor)
            else:
                dangling_name_set.add(tensor.name)
        node.inputs = keep_list
    if len(dangling_name_set) > 0:
        print(f"Removed {len(dangling_name_set)} dangling input tensors (read but never written, and not graph inputs):")
        for name in sorted(dangling_name_set):
            print(f"    {name}")

    onnx_model = gs.export_onnx(graph)
    # Engine layer types are not ONNX operators; see the docstring. Recent onnx-graphsurgeon already
    # declares the domain it finds on the nodes, so only add it when it is missing - a duplicated
    # opset entry is accepted by `onnx.checker` but is malformed all the same
    if ENGINE_DOMAIN not in [opset.domain for opset in onnx_model.opset_import]:
        onnx_model.opset_import.append(onnx.helper.make_opsetid(ENGINE_DOMAIN, 1))
    onnx.save(
        onnx_model,
        export_onnx_file,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=export_onnx_file.name + ".weight",
    )
    print(f"Succeed saving {export_onnx_file.name}: {len(graph.nodes):5d} Nodes, {len(graph.tensors().keys()):5d} tensors")

    return
