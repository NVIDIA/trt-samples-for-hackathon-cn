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
"""The `Loop` back end -- the only folding that actually shrinks the TensorRT network.

A local function is inlined again by the TensorRT parser, so it costs nothing and
buys nothing downstream. An ONNX `Loop` becomes a real `ILoopLayer`, and the body
is instantiated **once** no matter how many iterations there are. Measured on a
12-layer encoder in `08-LoopBackend`:

* TensorRT layers 858 -> 115 and build time 12.2 s -> 8.1 s;
* the engine is the *same size*, folding does not save memory;
* inference is ~1.4x slower, because the weights become the result of a runtime
  `Gather` and can no longer be constant folded, kernel specialised or fused
  across iterations.

A deliberate trade, not an upgrade, so it is off by default.

How each `Loop` feature is used:

* **loop-carried variables** `v_1..v_N` -- every value an iteration hands to the
  next one. There can be several: a block with both a hidden state and a residual
  needs two, which is why the first version of this file, supporting exactly one,
  rejected most real models.
* **scan outputs** -- every value an iteration hands to the *outside*. ONNX
  stacks them into `[K, ...]`, and each original consumer is fed back its own
  slice with `Gather(scan, k)`.
* **enclosing-scope reads** -- inputs that are identical in every iteration. A
  sub-graph may read the outer scope; a function may not.
* **`Gather(W_stacked, iter)`** -- initializers that differ per iteration, stacked
  into a single `[K, ...]` constant.

`analyse` reports precisely what failed, so a rejection is never silent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx_graphsurgeon as gs

from .verify import VerifiedPattern

@dataclass
class LoopPlan:
    """How one chain of instances maps onto a `Loop`."""

    instance_list: list  # The instances of this chain, in iteration order
    carried_output: list  # External output indices that feed the next iteration
    scan_output: list  # External output indices that leave the chain
    role: list  # Per external input slot: ("carried", j) / ("invariant", ) / ("stack", )
    initial_tensor: list  # Per carried output, the value entering iteration 0
    final_tensor: list  # Per carried output, the value leaving the last iteration

def _boundary(gs_node_list: list, instance) -> tuple:
    """The real tensors one instance consumes from / exposes to the outside."""
    node_list = [gs_node_list[i] for i in instance.node_id_list]
    input_tensor = [node_list[offset].inputs[slot] for offset, slot in instance.external_input]
    output_tensor = [node_list[offset].outputs[slot] for offset, slot in instance.external_output]
    return input_tensor, output_tensor

def _split_into_chains(inside: list, output_tensor: list) -> tuple:
    """Group the instances into maximal chains of "my outputs feed that instance".

    A pattern is not always one chain. Six encoder layers arranged as
    `(3 layers + tanh) x 2` are two chains of three, and each becomes its own
    `Loop`. Returns `(chain_list, reason)`.
    """
    n_instance = len(output_tensor)
    successor = {}
    for i in range(n_instance):
        consumer = {id(node) for tensor in output_tensor[i] for node in tensor.outputs}
        target = [j for j in range(n_instance) if j != i and consumer & inside[j]]
        if len(target) > 1:
            return None, f"instance {i} feeds {len(target)} other instances of the same pattern, which is not a chain"
        if target:
            successor[i] = target[0]
    if len(set(successor.values())) != len(successor):
        return None, "two instances feed the same instance, which is not a chain"

    has_predecessor = set(successor.values())
    chain_list = []
    for i in range(n_instance):
        if i in has_predecessor:
            continue  # Not the head of a chain
        chain, cursor = [i], i
        while cursor in successor:
            cursor = successor[cursor]
            chain.append(cursor)
        chain_list.append(chain)
    return chain_list, ""

def _classify_outputs(inside: list, output_tensor: list, chain: list) -> tuple:
    """Split the external outputs into loop-carried ones and scan outputs.

    An output that reaches the next iteration is a loop-carried variable. One that
    leaves the chain is a scan output. An output can be **both**, when an
    intermediate iteration hands its value to the next iteration *and* to
    something outside.
    """
    in_chain = set().union(*[inside[i] for i in chain])
    carried, scan = [], []
    for j in range(len(output_tensor[0])):
        to_next, to_outside = [], []
        for position, i in enumerate(chain):
            consumer = {id(node) for node in output_tensor[i][j].outputs}
            next_instance = inside[chain[position + 1]] if position + 1 < len(chain) else set()
            if consumer & (in_chain - next_instance - inside[i]):
                return None, None, f"external output {j} is read by a non-adjacent iteration of the same chain"
            to_next.append(bool(consumer & next_instance))
            to_outside.append(bool(consumer - in_chain) or not consumer)

        # The last iteration has no successor, so it does not vote on whether this
        # output is loop-carried; everything it produces leaves the chain anyway.
        vote = to_next[:-1]
        if any(vote) and not all(vote):
            return None, None, f"external output {j} reaches the next iteration only sometimes"
        if vote and all(vote):
            carried.append(j)
            if any(to_outside[:-1]):
                scan.append(j)  # Also published outside partway through the chain
        else:
            scan.append(j)
    return carried, scan, ""

def _analyse_chain(instance_list: list, input_tensor: list, output_tensor: list, carried: list, scan: list) -> tuple:
    """Role of every external input slot for one chain. Returns (plan, reason)."""
    n_iteration = len(instance_list)
    produced_by = {}  # id(tensor) -> (iteration, carried output index)
    for i in range(n_iteration):
        for j in carried:
            produced_by[id(output_tensor[i][j])] = (i, j)

    role = []
    for slot in range(len(input_tensor[0])):
        column = [instance[slot] for instance in input_tensor]
        source = [produced_by.get(id(column[i])) for i in range(1, n_iteration)]
        if all(t is column[0] for t in column):
            role.append(("invariant", ))
        elif all(s is not None and s[0] == i and s[1] == source[0][1] for i, s in enumerate(source)):
            role.append(("carried", source[0][1]))
        elif all(isinstance(t, gs.Constant) for t in column) and len({(str(t.dtype), tuple(t.shape or ())) for t in column}) == 1:
            role.append(("stack", ))
        else:
            return None, f"input slot {slot} is neither loop-carried, invariant nor a stackable initializer"

    for j in carried:
        slot_list = [slot for slot, r in enumerate(role) if r == ("carried", j)]
        if not slot_list:
            return None, f"loop-carried output {j} is not read back by the next iteration"
        # Several slots may read the same loop variable -- a residual connection
        # feeds the block input to both the attention and the following Add.
        for i in range(n_iteration):
            if len({id(input_tensor[i][slot]) for slot in slot_list}) != 1:
                return None, f"loop variable {j} arrives as different tensors at different input slots"

    initial = [input_tensor[0][next(slot for slot, r in enumerate(role) if r == ("carried", j))] for j in carried]
    final = [output_tensor[-1][j] for j in carried]
    return LoopPlan(instance_list, carried, scan, role, initial, final), ""

def analyse(gs_node_list: list, pattern: VerifiedPattern) -> tuple:
    """Can this pattern become one `Loop` per chain? Returns (plan_list, reason).

    All-or-nothing on purpose: if even one instance ends up alone in a chain, the
    whole pattern falls back to a local function rather than producing a model
    that is half loop and half inline copies.
    """
    instance_list = pattern.instance_list
    if len(instance_list) < 2:
        return [], "fewer than two instances"

    boundary = [_boundary(gs_node_list, instance) for instance in instance_list]
    input_tensor = [b[0] for b in boundary]
    output_tensor = [b[1] for b in boundary]
    inside = [{id(gs_node_list[i]) for i in instance.node_id_list} for instance in instance_list]

    chain_list, reason = _split_into_chains(inside, output_tensor)
    if chain_list is None:
        return [], reason
    short = [c for c in chain_list if len(c) < 2]
    if short:
        return [], (f"the {len(instance_list)} instances break into {len(chain_list)} chains, "
                    f"{len(short)} of them a single instance, so they do not form loops")

    plan_list = []
    for chain in chain_list:
        carried, scan, reason = _classify_outputs(inside, output_tensor, chain)
        if carried is None:
            return [], reason
        if not carried:
            return [], "no output reaches the next iteration, the instances do not form a chain"
        plan, reason = _analyse_chain([instance_list[i] for i in chain], [input_tensor[i] for i in chain], [output_tensor[i] for i in chain], carried, scan)
        if plan is None:
            return [], reason
        plan_list.append(plan)
    return plan_list, ""

def outline_pattern_as_loop(graph: gs.Graph, gs_node_list: list, plan: LoopPlan, name: str, root: gs.Graph | None = None) -> dict:
    """Replace one chain with a single `Loop` node. Returns a small stat dict.

    `graph` is where the nodes live, `root` is where the opset comes from; they
    differ when the chain sits inside another `Loop` or `If` body.
    """
    root = root if root is not None else graph
    instance_list = plan.instance_list
    n_iteration = len(instance_list)
    reference = instance_list[0]
    reference_node = [gs_node_list[i] for i in reference.node_id_list]
    reference_input = _boundary(gs_node_list, reference)[0]

    # ---- Stack the per-iteration initializers into one `[K, ...]` constant
    stacked = {}
    for slot, kind in enumerate(plan.role):
        if kind[0] != "stack":
            continue
        value = np.stack([_boundary(gs_node_list, instance)[0][slot].values for instance in instance_list])
        stacked[slot] = gs.Constant(f"{name}_W{slot}", value)

    # ---- Body: (iter, cond_in, v_0_in, ...) -> (cond_out, v_0_out, ..., scan...)
    iteration = gs.Variable(f"{name}_iter", dtype=np.int64, shape=[])
    condition_in = gs.Variable(f"{name}_cond_in", dtype=bool, shape=[])
    condition_out = gs.Variable(f"{name}_cond_out", dtype=bool, shape=[])
    carried_in = {j: gs.Variable(f"{name}_v{j}_in", dtype=t.dtype, shape=t.shape) for j, t in zip(plan.carried_output, plan.initial_tensor)}

    body_node_list, slot_tensor = [], {}
    for slot, constant in stacked.items():
        gathered = gs.Variable(f"{name}_w{slot}", dtype=constant.dtype, shape=constant.shape[1:])
        body_node_list.append(gs.Node("Gather", f"{name}/Gather_{slot}", {"axis": 0}, inputs=[constant, iteration], outputs=[gathered]))
        slot_tensor[slot] = gathered

    input_index = {position: k for k, position in enumerate(reference.external_input)}
    remap, n_internal = {}, 0
    for node in reference_node:
        for tensor in node.outputs:
            if id(tensor) not in remap:
                remap[id(tensor)] = gs.Variable(f"{name}_t_{n_internal}", dtype=tensor.dtype, shape=tensor.shape)
                n_internal += 1

    def body_input(offset: int, slot: int, tensor):
        """What the body node at `offset` should read at input `slot`."""
        k = input_index.get((offset, slot))
        if k is None:
            return remap.get(id(tensor), tensor)  # Internal edge
        kind = plan.role[k]
        if kind[0] == "carried":
            return carried_in[kind[1]]
        if kind[0] == "stack":
            return slot_tensor[k]
        return reference_input[k]  # Invariant: a sub-graph may read the enclosing scope

    for offset, node in enumerate(reference_node):
        body_node = node.copy()
        body_node.name = f"{name}/{offset:03d}_{node.name or node.op}"
        body_node.inputs = [body_input(offset, slot, tensor) for slot, tensor in enumerate(node.inputs)]
        body_node.outputs = [remap[id(t)] for t in node.outputs]
        body_node_list.append(body_node)

    def body_output(j: int):
        """The body tensor carrying external output `j`."""
        offset, out_slot = reference.external_output[j]
        return remap[id(reference_node[offset].outputs[out_slot])]

    body_node_list.append(gs.Node("Identity", f"{name}/CondPassThrough", inputs=[condition_in], outputs=[condition_out]))
    carried_out = [body_output(j) for j in plan.carried_output]

    # A value that is both loop-carried and published outside needs two distinct
    # body outputs, because ONNX identifies them by position.
    scan_body_output = []
    for j in plan.scan_output:
        tensor = body_output(j)
        if any(tensor is t for t in carried_out):
            duplicate = gs.Variable(f"{name}_scan{j}", dtype=tensor.dtype, shape=tensor.shape)
            body_node_list.append(gs.Node("Identity", f"{name}/ScanDup_{j}", inputs=[tensor], outputs=[duplicate]))
            tensor = duplicate
        scan_body_output.append(tensor)

    body = gs.Graph(nodes=body_node_list, name=f"{name}_body", inputs=[iteration, condition_in] + list(carried_in.values()), outputs=[condition_out] + carried_out + scan_body_output, opset=root.opset)

    # ---- Main graph: one `Loop` node replacing the whole chain
    trip_count = gs.Constant(f"{name}_trip", np.array(n_iteration, dtype=np.int64))
    condition = gs.Constant(f"{name}_cond", np.array(True, dtype=bool))
    scan_tensor = [gs.Variable(f"{name}_scan_out{j}") for j in plan.scan_output]
    loop_node = gs.Node("Loop", name, {"body": body}, inputs=[trip_count, condition] + list(plan.initial_tensor), outputs=list(plan.final_tensor) + scan_tensor)

    # Hand every iteration's slice back to whoever used to read it. Re-using the
    # original tensor *objects* as the `Gather` outputs means the existing
    # consumers need no rewiring at all.
    slice_node_list = []
    for j, scan in zip(plan.scan_output, scan_tensor):
        for k, instance in enumerate(instance_list):
            if j in plan.carried_output and k == n_iteration - 1:
                continue  # The final value already comes out of the Loop directly
            original = _boundary(gs_node_list, instance)[1][j]
            index = gs.Constant(f"{name}_idx{j}_{k}", np.array(k, dtype=np.int64))
            slice_node_list.append(gs.Node("Gather", f"{name}/Slice_{j}_{k}", {"axis": 0}, inputs=[scan, index], outputs=[original]))

    for instance in instance_list:
        for node_id in instance.node_id_list:
            gs_node_list[node_id].inputs.clear()  # Detach, `graph.cleanup()` drops them
            gs_node_list[node_id].outputs.clear()
    graph.nodes.append(loop_node)
    graph.nodes.extend(slice_node_list)

    return {
        "n_iteration": n_iteration,
        "n_body_node": len(body_node_list),
        "n_carried": len(plan.carried_output),
        "n_scan_output": len(plan.scan_output),
        "n_stacked_initializer": len(stacked),
        "n_invariant_input": sum(1 for r in plan.role if r[0] == "invariant"),
        "n_slice_node": len(slice_node_list),
    }
