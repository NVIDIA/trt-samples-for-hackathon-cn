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
"""P5 -- turn a verified pattern into one shared `FunctionProto` plus call nodes.

The one hard constraint of the ONNX object model shows up here: `FunctionProto`
has no `initializer` field and a function is not a closure, so **weights have to
be function inputs** and the actual initializer is passed at the call site. This
is the same shape `torch.onnx.export(..., export_modules_as_functions=...)`
produces, see `01-SubgraphInONNX`.

Unlike `onnxscript.rewriter(as_function=True)`, which emits one `FunctionProto`
per match, this emits **one body for all instances**.
"""

from __future__ import annotations

import onnx
import onnx_graphsurgeon as gs

from .verify import VerifiedPattern

def outline_pattern(
    graph: gs.Graph,
    gs_node_list: list[gs.Node],
    pattern: VerifiedPattern,
    function_name: str,
    domain: str,
    root: gs.Graph | None = None,
) -> gs.Function:
    """Replace every instance of `pattern` with a call to one new local function.

    `graph` is where the nodes live, which may be a `Loop` or `If` body. `root` is
    where the function has to be *registered*: a `FunctionProto` is a property of
    the model, not of a graph, and so is the opset import that declares its
    domain. A sub-graph may call a model-local function perfectly well.
    """
    instance_list = pattern.instance_list
    reference = instance_list[0]

    # ---- Boundary tensors of every instance, in the canonical (offset, slot) order
    def boundary(instance) -> tuple[list, list]:
        """The real tensors this instance consumes from / exposes to the outside."""
        node_list = [gs_node_list[i] for i in instance.node_id_list]
        input_tensor = [node_list[offset].inputs[slot] for offset, slot in instance.external_input]
        output_tensor = [node_list[offset].outputs[slot] for offset, slot in instance.external_output]
        return input_tensor, output_tensor

    reference_input, reference_output = boundary(reference)
    boundary_list = [boundary(instance) for instance in instance_list]

    def common_type(getter) -> list:
        """dtype/shape shared by *every* instance, or (None, None) where they differ.

        A local function is a template instantiated once per call site, so
        pinning it to the reference instance's shapes is wrong the moment two
        instances run at different shapes: onnxruntime then reports
        `[ShapeInferenceError] Incompatible dimensions` and TensorRT refuses the
        model, even though the topology is fine. Only annotate what all
        instances agree on and let shape inference do the rest.
        """
        result = []
        for tensor_list in zip(*[getter(b) for b in boundary_list]):
            dtype = tensor_list[0].dtype if len({t.dtype for t in tensor_list}) == 1 else None
            shape_set = {tuple(t.shape) if t.shape is not None else None for t in tensor_list}
            result.append((dtype, tensor_list[0].shape if len(shape_set) == 1 else None))
        return result

    # ---- Build the function body out of a copy of the reference instance
    input_type = common_type(lambda b: b[0])
    output_type = common_type(lambda b: b[1])
    function_input = [gs.Variable(f"{function_name}_in_{k}", dtype=d, shape=s) for k, (d, s) in enumerate(input_type)]
    function_output = [gs.Variable(f"{function_name}_out_{k}", dtype=d, shape=s) for k, (d, s) in enumerate(output_type)]

    # Inputs are remapped by `(offset, slot)`, NOT by tensor identity. One tensor
    # may legitimately feed several slots of the instance (a shared weight, the
    # same activation used twice); an identity-keyed map would collapse them and
    # every body node reading that tensor would silently point at whichever
    # function input was registered last.
    input_index = {position: k for k, position in enumerate(reference.external_input)}
    # Outputs are safe to key by identity: a tensor has exactly one producer.
    remap = {id(t): v for t, v in zip(reference_output, function_output)}

    # Purely internal tensors get fresh objects too, so the body never shares a
    # tensor (or a name) with the main graph. Same rule as above: an internal
    # tensor only keeps its dtype/shape when every instance agrees on it.
    n_internal = 0
    for offset, node_id in enumerate(reference.node_id_list):
        for out_slot, tensor in enumerate(gs_node_list[node_id].outputs):
            if id(tensor) in remap:
                continue
            same_slot = [gs_node_list[i.node_id_list[offset]].outputs[out_slot] for i in instance_list]
            dtype = tensor.dtype if len({t.dtype for t in same_slot}) == 1 else None
            shape_set = {tuple(t.shape) if t.shape is not None else None for t in same_slot}
            remap[id(tensor)] = gs.Variable(f"{function_name}_t_{n_internal}", dtype=dtype, shape=tensor.shape if len(shape_set) == 1 else None)
            n_internal += 1

    body_node_list = []
    for offset, node_id in enumerate(reference.node_id_list):
        source = gs_node_list[node_id]
        body_node = source.copy()
        body_node.name = f"{function_name}/{offset:03d}_{source.name or source.op}"
        body_node.inputs = [function_input[input_index[(offset, slot)]] if (offset, slot) in input_index else remap.get(id(t), t) for slot, t in enumerate(source.inputs)]
        body_node.outputs = [remap.get(id(t), t) for t in source.outputs]
        body_node_list.append(body_node)

    # From level 1 on the body itself calls level-0 functions, so the custom
    # domain has to be declared both on the graph and on every function that
    # uses it, otherwise `onnx.checker` rejects the model.
    root = root if root is not None else graph
    if not any(o.domain == domain for o in root.import_domains):
        root.import_domains.append(onnx.helper.make_opsetid(domain, 1))

    function = gs.Function(
        function_name,
        domain=domain,
        nodes=body_node_list,
        inputs=function_input,
        outputs=function_output,
        opset=root.opset,
        import_domains=root.import_domains,
    )
    root.functions.append(function)

    # ---- Replace every instance with a single call node
    for k, instance in enumerate(instance_list):
        call_input, call_output = boundary(instance)
        call_node = gs.Node(
            op=function_name,
            domain=domain,
            name=f"{function_name}_call_{k}",
            inputs=list(call_input),
            outputs=list(call_output),
        )
        for node_id in instance.node_id_list:
            node = gs_node_list[node_id]
            node.inputs.clear()  # Detach, `graph.cleanup()` drops them afterwards
            node.outputs.clear()
        graph.nodes.append(call_node)

    return function
