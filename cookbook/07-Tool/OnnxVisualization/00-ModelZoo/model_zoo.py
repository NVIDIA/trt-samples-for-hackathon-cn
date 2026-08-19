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
"""Every ONNX test model this project produced, and what the outliner should say
about each of them. This module is the single source of truth for both.

They were written one at a time while chasing a specific question, several of
them only ever lived inside a test function or a throwaway script. Each one
isolates one property of the graph, which is what makes them worth keeping:
together they are a small corpus for anything that has to reason about repeated
structure in ONNX.

The module is a library, not a script:

* `build_model_zoo.py`      writes every model to `model/` plus a `manifest.json`;
* `01-BasicUsage/test_outliner.py` imports the builders and `EXPECT` and asserts
  against them, so a model and its expected answer can never drift apart;
* `01-BasicUsage/main.py`     calls `ensure()` for the models its demo needs.

`EXPECT[name]` holds the outliner configuration the model is meant to be read
with plus the pattern list it must produce, as `(size, n_instance)` sorted by
decreasing gain -- the same order the report uses.
"""

import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx.helper as oh
import torch
import torch.nn as nn

np.random.seed(31193)
torch.manual_seed(31193)

output_path = Path(__file__).parent
model_path = output_path / "model"
OPSET = 17
FLOAT = onnx.TensorProto.FLOAT
N_C = 4

# Distinct unary float operators, used as the letters A, B, C, ... of the small graphs
UNARY = ["Relu", "Tanh", "Sigmoid", "Softplus", "Elu", "Selu", "Softsign"]

def make_model(node_list: list, output_name_list: list, name: str, *, input_name_list=None, initializer_list=None, n_channel: int = N_C) -> onnx.ModelProto:
    """Wrap nodes into a checked model with `[n_channel]` shaped float inputs."""
    input_name_list = input_name_list or ["x"]
    graph = oh.make_graph(
        node_list,
        name,
        [oh.make_tensor_value_info(n, FLOAT, [n_channel]) for n in input_name_list],
        [oh.make_tensor_value_info(n, FLOAT, [n_channel]) for n in output_name_list],
        initializer_list or [],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", OPSET)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model

def make_shaped_model(node_list: list, output_spec: list, name: str, *, input_shape: list, initializer_list=None) -> onnx.ModelProto:
    """`make_model` for graphs whose tensors are not all the same 1-D shape.

    Shape-sensitive operators change the rank or the extents, so every output
    needs its own declared shape. `output_spec` is a list of `(name, shape)`.
    """
    graph = oh.make_graph(
        node_list,
        name,
        [oh.make_tensor_value_info("x", FLOAT, input_shape)],
        [oh.make_tensor_value_info(n, FLOAT, s) for n, s in output_spec],
        initializer_list or [],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", OPSET)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model

# ================================================================ Topology of the repetition

def serial_chain(n_block: int = 4) -> onnx.ModelProto:
    """`A->B->C` repeated in series. The easy case every method handles."""
    node_list, previous = [], "x"
    for i in range(n_block):
        for op in ["Relu", "Tanh", "Sigmoid"]:
            node_list.append(oh.make_node(op, [previous], [f"{op}{i}"], f"{op}{i}"))
            previous = f"{op}{i}"
    return make_model(node_list, [previous], "serial_chain")

def serial_plus_parallel(n_block: int = 4, n_parallel: int = 6) -> onnx.ModelProto:
    """A serial chain next to an independent one. An arbitrary topological order
    interleaves them (`ADBDCD...`) and the pattern is missed entirely."""
    node_list, previous = [], "x"
    for i in range(n_block):
        for op in ["Relu", "Tanh", "Sigmoid"]:
            node_list.append(oh.make_node(op, [previous], [f"{op}{i}"], f"{op}{i}"))
            previous = f"{op}{i}"
    main_output, previous = previous, "x"
    for j in range(n_parallel):
        node_list.append(oh.make_node("Softplus", [previous], [f"D{j}"], f"D{j}"))
        previous = f"D{j}"
    return make_model(node_list, [main_output, previous], "serial_plus_parallel")

def two_tower(n_block: int = 4) -> onnx.ModelProto:
    """Two independent towers. An arbitrary topological order reports a *false*
    pattern `AEBFCG` that interleaves two unrelated towers."""
    node_list, output_list = [], []
    for op_set in [["Relu", "Tanh", "Sigmoid"], ["Elu", "Selu", "Softsign"]]:
        previous = "x"
        for i in range(n_block):
            for op in op_set:
                node_list.append(oh.make_node(op, [previous], [f"{op}{i}"], f"{op}{i}"))
                previous = f"{op}{i}"
        output_list.append(previous)
    return make_model(node_list, output_list, "two_tower")

def internal_branch(n_block: int = 4) -> onnx.ModelProto:
    """A block that forks into three branches and merges again. Parallelism
    *inside* a block is harmless: the block stays contiguous."""
    node_list, previous = [], "x"
    for i in range(n_block):
        node_list.append(oh.make_node("Relu", [previous], [f"S{i}"], f"S{i}"))
        for op in ["Sigmoid", "Tanh", "Softplus"]:
            node_list.append(oh.make_node(op, [f"S{i}"], [f"{op}{i}"], f"{op}{i}"))
        node_list.append(oh.make_node("Sum", [f"Sigmoid{i}", f"Tanh{i}", f"Softplus{i}"], [f"M{i}"], f"M{i}"))
        previous = f"M{i}"
    return make_model(node_list, [previous], "internal_branch")

def fan_out(n_branch: int = 5) -> onnx.ModelProto:
    """N identical branches side by side, merged by one `Sum`. Findable, but with
    no loop-carried dependency, so no `Loop` can express it."""
    node_list, output_list = [], []
    for i in range(n_branch):
        previous = "x"
        for op in ["Relu", "Tanh", "Sigmoid"]:
            node_list.append(oh.make_node(op, [previous], [f"{op}{i}"], f"{op}{i}"))
            previous = f"{op}{i}"
        output_list.append(previous)
    node_list.append(oh.make_node("Sum", output_list, ["y"], "Merge"))
    return make_model(node_list, ["y"], "fan_out")

def shared_hub(n_branch: int = 5) -> onnx.ModelProto:
    """Every branch reads one shared tensor in the middle."""
    node_list, output_list = [oh.make_node("Relu", ["x"], ["hub"], "HUB")], []
    for i in range(n_branch):
        node_list += [
            oh.make_node("Tanh", ["x"], [f"a{i}"], f"A{i}"),
            oh.make_node("Add", [f"a{i}", "hub"], [f"b{i}"], f"B{i}"),
            oh.make_node("Sigmoid", [f"b{i}"], [f"c{i}"], f"C{i}"),
        ]
        output_list.append(f"c{i}")
    node_list.append(oh.make_node("Sum", output_list, ["y"], "Merge"))
    return make_model(node_list, ["y"], "shared_hub")

def shared_accumulator(n_branch: int = 5) -> onnx.ModelProto:
    """Branches strung along one running accumulator."""
    node_list = [oh.make_node("Relu", ["x"], ["acc0"], "ACC0")]
    for i in range(n_branch):
        node_list += [
            oh.make_node("Tanh", ["x"], [f"a{i}"], f"A{i}"),
            oh.make_node("Add", [f"a{i}", f"acc{i}"], [f"b{i}"], f"B{i}"),
            oh.make_node("Sigmoid", [f"b{i}"], [f"acc{i + 1}"], f"C{i}"),
        ]
    node_list.append(oh.make_node("Identity", [f"acc{n_branch}"], ["y"], "OUT"))
    return make_model(node_list, ["y"], "shared_accumulator")

def nested_two_level(n_inner: int = 3, n_outer: int = 2) -> onnx.ModelProto:
    """`n_outer` groups of `n_inner` blocks, each group closed by a `Softplus`.

    A genuinely two-level structure: the inner block is more frequent than the
    group, so a second folding level has something to find. Selection in P4 is
    gain driven and always takes the biggest win first:
        gain(inner) = (3-1) * (6-1) = 10   <- wins at level 0
        gain(outer) = (10-1) * (2-1) = 9
    After level 0 the graph reads `B B B Softplus B B B Softplus`, and level 1
    picks up `B B B Softplus` twice.
    """
    node_list, previous, index = [], "x", 0
    for _ in range(n_outer):
        for _ in range(n_inner):
            for op in ["Relu", "Tanh", "Sigmoid"]:
                node_list.append(oh.make_node(op, [previous], [f"t{index}"], f"{op}{index}"))
                previous = f"t{index}"
                index += 1
        node_list.append(oh.make_node("Softplus", [previous], [f"t{index}"], f"Softplus{index}"))
        previous = f"t{index}"
        index += 1
    return make_model(node_list, [previous], "nested_two_level")

# ================================================================ Traps for the rewriter

def shared_input(n_block: int = 4) -> onnx.ModelProto:
    """The *reference* instance shares a tensor across three input slots while the
    others do not.

    Every block is `m = Mul(a, b); n = Add(m, a); o = Relu(n)`, so it has three
    external input slots. In block 0 all three are fed by the same tensor `p`; in
    the other blocks they are three different tensors. Remapping a function body
    by tensor *identity* collapses block 0's three slots into whichever function
    input was registered last, so every other call site silently loses two of its
    three arguments. `onnx.checker` accepts the result, only the numeric cross
    check notices. Keyed by `(offset, slot)` the three slots stay distinct.

    It has to be the *reference* instance that shares the tensor: if every
    instance shared it too, the collapse would be harmless and the bug would hide.
    """
    node_list = [oh.make_node("Relu", ["x"], ["p"], "P")]
    previous, side_op = "p", ["Sigmoid", "Tanh", "Softplus", "Elu", "Selu"]
    for i in range(n_block):
        if i == 0:
            a = b = "p"  # The reference instance, and the only one that shares
        else:
            node_list.append(oh.make_node(side_op[i - 1], ["x"], [f"s{i}"], f"S{i}"))
            a, b = previous, f"s{i}"
        node_list += [
            oh.make_node("Mul", [a, b], [f"m{i}"], f"Mul{i}"),
            oh.make_node("Add", [f"m{i}", a], [f"n{i}"], f"Add{i}"),
            oh.make_node("Relu", [f"n{i}"], [f"o{i}"], f"Relu{i}"),
        ]
        previous = f"o{i}"
    return make_model(node_list, [previous], "shared_input")

def constant_attribute(n_block: int = 4) -> onnx.ModelProto:
    """Blocks holding `Constant` nodes of the same dtype and shape but different
    values.

    A `Constant`'s value lives in an *attribute*, which gets baked into the
    function body, unlike an initializer input which is passed per call site.
    Comparing such attributes by dtype and shape alone merges blocks that are not
    interchangeable and hands every call site the reference's constant. Each block
    here adds a different scalar, so a wrong merge changes the result while still
    producing a structurally valid model. The correct answer is *no pattern*.

    Only visible with `preprocess=False`: onnxslim turns the `Constant` nodes into
    initializers, after which they are per-call-site inputs and the trap is gone.
    """
    node_list, previous = [], "x"
    for i in range(n_block):
        value = oh.make_tensor(f"c{i}", FLOAT, [1], [float(i + 1)])
        node_list += [
            oh.make_node("Constant", [], [f"k{i}"], f"Const{i}", value=value),
            oh.make_node("Add", [previous, f"k{i}"], [f"a{i}"], f"Add{i}"),
            oh.make_node("Relu", [f"a{i}"], [f"r{i}"], f"Relu{i}"),
        ]
        previous = f"r{i}"
    return make_model(node_list, [previous], "constant_attribute")

def ambiguous_sibling(n_block: int = 4) -> onnx.ModelProto:
    """A block that hands the same value to two nodes of the **same operator**.

    `(direction, out_slot, in_slot, neighbour label)` cannot tell those two edges
    apart, and refusing to grow through an ambiguous edge leaves the side branch
    behind -- which cost about a third of the planted blocks in the stress test.
    Pairing same-key edges by the neighbour's own edge signature recovers the
    whole 4-node block instead of the 3-node trunk.
    """
    node_list, previous, side = [], "x", []
    for i in range(n_block):
        node_list += [
            oh.make_node("Relu", [previous], [f"a{i}"], f"A{i}"),
            oh.make_node("Tanh", [f"a{i}"], [f"b{i}"], f"B{i}"),
            oh.make_node("Tanh", [f"a{i}"], [f"c{i}"], f"C{i}"),
            oh.make_node("Sigmoid", [f"b{i}"], [f"d{i}"], f"D{i}"),
        ]
        side.append(f"c{i}")
        previous = f"d{i}"
    node_list.append(oh.make_node("Sum", side + [previous], ["y"], "Merge"))
    return make_model(node_list, ["y"], "ambiguous_sibling")

def planted_block_with_answer(onnx_file: Path, seed: int = 31193 + 4) -> tuple:
    """One draw of the random planted-block generator, with the planted answer.

    Returns `(onnx_file, planted)` where `planted` lists the node names of every
    instance that was planted, so a test can check what was recovered. On this
    particular draw a foreign node lands in the middle of the planted 5-node
    block, its run of the topological order breaks, and `method="serial"` reports
    only 4 of the 5 nodes; lockstep growth seeded with that partial answer
    recovers the fifth.

    The full generator lives in `../03-ParallelRepeat/stress_test.py`, which
    which needs nothing but `tensorrt_cookbook`.
    """
    sys.path.insert(0, str(output_path.parent / "03-ParallelRepeat"))
    from stress_test import build_model
    rng = random.Random(seed)
    n_block_node, n_external_input, n_instance, n_filler = rng.randint(3, 8), rng.randint(1, 3), rng.randint(2, 6), rng.randint(0, 4)
    return build_model(rng, n_block_node, n_external_input, n_instance, n_filler, onnx_file)

def planted_block(seed: int = 31193 + 4) -> onnx.ModelProto:
    """`planted_block_with_answer` as a plain model, for the zoo."""
    onnx_file = model_path / "_tmp_planted.onnx"
    model_path.mkdir(exist_ok=True)
    planted_block_with_answer(onnx_file, seed)
    model = onnx.load(onnx_file)
    onnx_file.unlink(missing_ok=True)
    return model

# ================================================================ Shape-sensitive operators

# A cube, so that *any* `Transpose.perm` keeps the shape and the only thing that
# changes between blocks is the attribute under test.
N_CUBE = 4

def transpose_perm(perm_list: list = None) -> onnx.ModelProto:
    """Four `Relu -> Transpose -> Sigmoid` branches, two `perm` values, 2 + 2.

    `perm` is an **attribute**, so it is baked into the function body and blocks
    that disagree on it are not interchangeable. The correct answer is therefore
    *two* patterns of two instances, never one pattern of four.

    Merging them anyway is a silent error of the worst kind: `onnx.checker`
    passes and TensorRT parses the result happily, only the numeric cross check
    notices (`max_abs_diff` around 0.11). `--strictness L0` drops attributes from
    the node label and does exactly that -- see `EXPECT` and the regression.
    """
    perm_list = perm_list or [[0, 2, 1], [0, 2, 1], [2, 1, 0], [2, 1, 0]]
    node_list, output_spec = [], []
    for i, perm in enumerate(perm_list):
        node_list += [
            oh.make_node("Relu", ["x"], [f"a{i}"], f"A{i}"),
            oh.make_node("Transpose", [f"a{i}"], [f"b{i}"], f"B{i}", perm=perm),
            oh.make_node("Sigmoid", [f"b{i}"], [f"y{i}"], f"C{i}"),
        ]
        output_spec.append((f"y{i}", [N_CUBE] * 3))
    return make_shaped_model(node_list, output_spec, "transpose_perm", input_shape=[N_CUBE] * 3)

def concat_axis(axis_list: list = None) -> onnx.ModelProto:
    """Four `Relu -> Concat(t, t, axis) -> Sigmoid` branches, two `axis` values.

    Same trap as `transpose_perm`, but here the attribute changes the *output
    shape*, so a wrong merge does not even produce a runnable graph: the branches
    end up claiming incompatible shapes and onnxruntime refuses with a broadcast
    error. `onnx.checker` still passes, which is the point.
    """
    axis_list = axis_list or [0, 0, 1, 1]
    node_list, output_spec = [], []
    for i, axis in enumerate(axis_list):
        shape = [N_CUBE] * 3
        shape[axis] *= 2
        node_list += [
            oh.make_node("Relu", ["x"], [f"a{i}"], f"A{i}"),
            oh.make_node("Concat", [f"a{i}", f"a{i}"], [f"b{i}"], f"B{i}", axis=axis),
            oh.make_node("Sigmoid", [f"b{i}"], [f"y{i}"], f"C{i}"),
        ]
        output_spec.append((f"y{i}", shape))
    return make_shaped_model(node_list, output_spec, "concat_axis", input_shape=[N_CUBE] * 3)

def reshape_shape_input(shape_list: list = None) -> onnx.ModelProto:
    """The mirror image: four branches reshaping to four *different* targets.

    `Reshape` takes its target as an **input tensor**, not an attribute. Inputs
    are passed per call site, so blocks that reshape to different shapes really
    are interchangeable and **must** merge into a single pattern -- the opposite
    verdict from `transpose_perm`, for the opposite reason.

    Getting this wrong in the other direction (refusing to merge) costs nothing
    but coverage, so this is the case that guards against over-tightening the
    node label: a fix for `transpose_perm` that also splits these is too strict.
    """
    shape_list = shape_list or [[N_CUBE * N_CUBE, N_CUBE], [N_CUBE, N_CUBE * N_CUBE], [N_CUBE ** 3], [2, 2, N_CUBE * N_CUBE]]
    node_list, output_spec, initializer_list = [], [], []
    for i, target in enumerate(shape_list):
        initializer_list += [
            oh.make_tensor(f"s{i}", onnx.TensorProto.INT64, [len(target)], target),
            oh.make_tensor(f"back{i}", onnx.TensorProto.INT64, [3], [N_CUBE] * 3),
        ]
        node_list += [
            oh.make_node("Relu", ["x"], [f"a{i}"], f"A{i}"),
            oh.make_node("Reshape", [f"a{i}", f"s{i}"], [f"b{i}"], f"B{i}"),
            oh.make_node("Sigmoid", [f"b{i}"], [f"c{i}"], f"C{i}"),
            oh.make_node("Reshape", [f"c{i}", f"back{i}"], [f"y{i}"], f"D{i}"),
        ]
        output_spec.append((f"y{i}", [N_CUBE] * 3))
    return make_shaped_model(node_list, output_spec, "reshape_shape_input", input_shape=[N_CUBE] * 3, initializer_list=initializer_list)

# ================================================================ Loop / If shapes

def loop_scan_output(n_block: int = 4) -> onnx.ModelProto:
    """A block with two external outputs: one loop-carried, one leaving the chain.

    block i:  a = Relu(previous)   b = Tanh(a)   c = Sigmoid(a)
    `c` feeds block i+1 (a loop-carried variable) while every `b` is consumed
    outside the chain, so `b` has to become a scan output and each consumer gets
    its own `Gather(scan, k)` slice back.
    """
    node_list, previous, side = [], "x", []
    for i in range(n_block):
        node_list += [
            oh.make_node("Relu", [previous], [f"a{i}"], f"A{i}"),
            oh.make_node("Tanh", [f"a{i}"], [f"b{i}"], f"B{i}"),
            oh.make_node("Sigmoid", [f"a{i}"], [f"c{i}"], f"C{i}"),
        ]
        side.append(f"b{i}")
        previous = f"c{i}"
    node_list.append(oh.make_node("Sum", side + [previous], ["y"], "Merge"))
    return make_model(node_list, ["y"], "loop_scan_output")

def loop_two_carried(n_block: int = 4) -> onnx.ModelProto:
    """A block that hands **two** values to the next one, i.e. two loop-carried
    variables.

    block i:  p = Add(u, v)   q = Mul(p, u)   u' = Relu(p)   v' = Tanh(q)
    Supporting only a single loop-carried variable rejects this, which is what the
    first version of the `Loop` back end did and why it fitted almost no real model.
    """
    node_list = [oh.make_node("Relu", ["x"], ["u0"], "U0"), oh.make_node("Tanh", ["x"], ["v0"], "V0")]
    u, v = "u0", "v0"
    for i in range(n_block):
        node_list += [
            oh.make_node("Add", [u, v], [f"p{i}"], f"P{i}"),
            oh.make_node("Mul", [f"p{i}", u], [f"q{i}"], f"Q{i}"),
            oh.make_node("Relu", [f"p{i}"], [f"u{i + 1}"], f"UU{i}"),
            oh.make_node("Tanh", [f"q{i}"], [f"v{i + 1}"], f"VV{i}"),
        ]
        u, v = f"u{i + 1}", f"v{i + 1}"
    node_list.append(oh.make_node("Sum", [u, v], ["y"], "Merge"))
    return make_model(node_list, ["y"], "loop_two_carried")

def loop_with_repeats(n_block: int = 4, n_iteration: int = 3) -> onnx.ModelProto:
    """A `Loop` whose **body** holds repeated blocks, with nothing to fold outside.

    The main graph is a single `Loop` node, so a tool that only looks at the top
    level finds nothing at all. A `FunctionProto` belongs to the model rather than
    to a graph, so the body can call one just like the main graph can.
    """
    body_node_list, previous = [], "x_in"
    for i in range(n_block):
        for op in ["Relu", "Tanh", "Sigmoid"]:
            body_node_list.append(oh.make_node(op, [previous], [f"t{i}_{op}"], f"B{i}_{op}"))
            previous = f"t{i}_{op}"
    body_node_list += [
        oh.make_node("Identity", ["cond_in"], ["cond_out"], "Cond"),
        oh.make_node("Identity", [previous], ["x_out"], "Out"),
    ]
    body = oh.make_graph(
        body_node_list,
        "body",
        [oh.make_tensor_value_info("iter", onnx.TensorProto.INT64, []), oh.make_tensor_value_info("cond_in", onnx.TensorProto.BOOL, []), oh.make_tensor_value_info("x_in", FLOAT, [N_C])],
        [oh.make_tensor_value_info("cond_out", onnx.TensorProto.BOOL, []), oh.make_tensor_value_info("x_out", FLOAT, [N_C])],
    )
    initializer_list = [oh.make_tensor("trip", onnx.TensorProto.INT64, [], [n_iteration]), oh.make_tensor("cond", onnx.TensorProto.BOOL, [], [True])]
    node_list = [oh.make_node("Loop", ["trip", "cond", "x"], ["y"], "MyLoop", body=body)]
    return make_model(node_list, ["y"], "loop_with_repeats", initializer_list=initializer_list)

def if_with_repeats(n_block: int = 4) -> onnx.ModelProto:
    """An `If` whose two branches each hold repeated blocks."""

    def branch(tag: str, op_list: list):
        """One branch of the `If`, itself a chain of repeated blocks."""
        node_list, previous = [], "x"
        for i in range(n_block):
            for op in op_list:
                node_list.append(oh.make_node(op, [previous], [f"{tag}{i}_{op}"], f"{tag}{i}_{op}"))
                previous = f"{tag}{i}_{op}"
        node_list.append(oh.make_node("Identity", [previous], [f"{tag}_out"], f"{tag}_Out"))
        return oh.make_graph(node_list, f"{tag}_branch", [], [oh.make_tensor_value_info(f"{tag}_out", FLOAT, [N_C])])

    node_list = [oh.make_node("If", ["flag"], ["y"], "MyIf", then_branch=branch("T", ["Relu", "Tanh", "Sigmoid"]), else_branch=branch("E", ["Elu", "Selu", "Softsign"]))]
    return make_model(node_list, ["y"], "if_with_repeats", initializer_list=[oh.make_tensor("flag", onnx.TensorProto.BOOL, [], [True])])

# ================================================================ Exported from PyTorch

class Block(nn.Module):
    """`Linear -> ReLU -> Linear -> ReLU`, the repeated sub-module of `flat_mlp`."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(c, c)
        self.fc2 = nn.Linear(c, c)

    def forward(self, x):
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))

class MlpNet(nn.Module):
    """N blocks with independent weights, the classic node-explosion shape."""

    def __init__(self, c: int = 8, n: int = 4) -> None:
        super().__init__()
        self.block_list = nn.ModuleList([Block(c) for _ in range(n)])

    def forward(self, x):
        for block in self.block_list:
            x = block(x)
        return x

class BranchyNet(nn.Module):
    """Five identical encoder layers wired into a fork, a re-join and a skip:
    `A->B, A->C, C->E, h=B*C, D(h+A), f=D+E`.

    `A->D` is ambiguous, because a block takes exactly one tensor: `b_skip=True`
    feeds `D` the sum `h + A`, `b_skip=False` lets `D` read `h` alone and reach
    `A` only transitively. Both readings are worth testing, so both are in the zoo.
    """

    def __init__(self, b_skip: bool = True) -> None:
        super().__init__()
        self.b_skip = b_skip
        for name in "ABCDE":
            setattr(self, name, nn.TransformerEncoderLayer(64, 4, dim_feedforward=256, batch_first=True))

    def forward(self, x):
        a = self.A(x)
        b, c = self.B(a), self.C(a)
        e = self.E(c)
        h = b * c  # Element-wise product of the two parallel branches
        return self.D(h + a if self.b_skip else h) + e

class TwoStageNet(nn.Module):
    """`(3 encoder layers + tanh) x 2`, a genuinely two-level transformer.

    Six identical layers, but arranged as two stages. Level 0 folds the layer,
    because folding one layer beats folding a whole stage. Level 1 then sees
    `B B B Tanh B B B Tanh` and wraps each stage. The trailing `tanh` has to be on
    *both* stages: with a single separator between them the second group has no
    `tanh` and the two stages are not instances of one pattern at all.
    """

    def __init__(self) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(64, 4, dim_feedforward=256, batch_first=True)
        self.stage1, self.stage2 = nn.TransformerEncoder(layer, 3), nn.TransformerEncoder(layer, 3)

    def forward(self, x):
        return torch.tanh(self.stage2(torch.tanh(self.stage1(x))))

class MoENet(nn.Module):
    """Six expert branches summed together, the mixture-of-experts shape."""

    def __init__(self, n: int = 6, c: int = 32) -> None:
        super().__init__()
        self.expert = nn.ModuleList([nn.Sequential(nn.Linear(c, 4 * c), nn.ReLU(), nn.Linear(4 * c, c)) for _ in range(n)])

    def forward(self, x):
        return sum(expert(x) for expert in self.expert)

class LoopBodyNet(nn.Module):
    """A data-dependent `for` (kept as a real `Loop`) whose body contains a
    constant-trip `for` (unrolled at export), i.e. repeats inside a `Loop` body.

    The outer trip count has to be data dependent: a constant one is unrolled by
    the exporter and no `Loop` survives at all, see `../01-SubgraphInONNX`.
    """

    def forward(self, x, n):
        for _ in range(int(n)):
            for _ in range(4):
                x = torch.sigmoid(torch.tanh(torch.relu(x)))
        return x

class LoopSharedWeightNet(nn.Module):
    """One block applied a data-dependent number of times, so the export keeps a
    `Loop` whose trip count is a runtime input (a TensorRT *shape* input)."""

    def __init__(self, c: int = 8) -> None:
        super().__init__()
        self.block = Block(c)

    def forward(self, x, n):
        for _ in range(int(n)):
            x = self.block(x)
        return x

def export_torch(module, sample, onnx_file: Path, input_name_list: list, *, b_script: bool = False) -> onnx.ModelProto:
    """Export one PyTorch module and return the resulting model."""
    module = module.eval()
    if b_script:
        module = torch.jit.script(module)
    torch.onnx.export(module, sample, onnx_file, dynamo=False, opset_version=OPSET, input_names=input_name_list, output_names=["y"])
    return onnx.load(onnx_file)

def _export_as_function(onnx_file: Path) -> onnx.ModelProto:
    """`export_modules_as_functions` needs the module class, so it gets its own helper."""
    torch.onnx.export(MlpNet().eval(), (torch.randn(2, 8), ), onnx_file, dynamo=False, opset_version=OPSET, input_names=["x"], output_names=["y"], export_modules_as_functions={Block})
    return onnx.load(onnx_file)

def torch_model_list() -> list:
    """Every PyTorch-exported entry, as `(name, builder, description)`."""
    return [
        ("flat_mlp", lambda f: export_torch(MlpNet(), (torch.randn(2, 8), ), f, ["x"]), "4 MLP blocks with independent weights, exported flat"),
        ("flat_mlp_as_function", lambda f: _export_as_function(f), "the same 4 blocks exported with `export_modules_as_functions`, i.e. already outlined by PyTorch"),
        ("transformer_6layer", lambda f: export_torch(nn.TransformerEncoder(nn.TransformerEncoderLayer(64, 4, dim_feedforward=256, batch_first=True), 6), (torch.randn(2, 16, 64), ), f, ["x"]), "6 identical encoder layers, the main target of the whole project"),
        ("transformer_two_stage", lambda f: export_torch(TwoStageNet(), (torch.randn(2, 16, 64), ), f, ["x"]), "`(3 layers + tanh) x 2`, a two-level transformer"),
        ("transformer_branchy", lambda f: export_torch(BranchyNet(b_skip=True), (torch.randn(2, 16, 64), ), f, ["x"]), "5 encoder layers in a fork / re-join / skip DAG, `D` reads `h + A`"),
        ("transformer_branchy_transitive", lambda f: export_torch(BranchyNet(b_skip=False), (torch.randn(2, 16, 64), ), f, ["x"]), "the same DAG with `D` reading `h` alone, so `A` is reached only transitively"),
        ("moe_6expert", lambda f: export_torch(MoENet(), (torch.randn(2, 32), ), f, ["x"]), "6 parallel expert branches summed together"),
        ("loop_body_repeats", lambda f: export_torch(LoopBodyNet(), (torch.randn(8), torch.tensor(3)), f, ["x", "n"], b_script=True), "a real `Loop` whose body holds 4 repeated blocks"),
        ("loop_shared_weight", lambda f: export_torch(LoopSharedWeightNet(), (torch.randn(2, 8), torch.tensor(3)), f, ["x", "n"], b_script=True), "a `Loop` with a runtime trip count, which TensorRT sees as a shape input"),
    ]

SYNTHETIC = [
    (serial_chain, "4 blocks in series, the easy case"),
    (serial_plus_parallel, "a serial chain next to an independent one; an arbitrary topological order misses the pattern"),
    (two_tower, "two independent towers; an arbitrary topological order reports a false interleaved pattern"),
    (internal_branch, "parallelism *inside* a block, which is harmless"),
    (fan_out, "N branches side by side; findable but not expressible as a `Loop`"),
    (shared_hub, "every branch reads one shared tensor in the middle"),
    (shared_accumulator, "branches strung along a running accumulator"),
    (nested_two_level, "2 groups of 3 blocks, a genuinely two-level structure"),
    (shared_input, "the reference instance shares a tensor across slots, the others do not"),
    (constant_attribute, "same-shaped `Constant` attributes holding different values"),
    (ambiguous_sibling, "one value handed to two nodes of the same operator"),
    (transpose_perm, "two `Transpose.perm` values; merging them passes checker and TensorRT but is numerically wrong"),
    (concat_axis, "two `Concat.axis` values; merging them yields a graph onnxruntime cannot even run"),
    (reshape_shape_input, "four different `Reshape` targets, passed as inputs rather than attributes, so they *do* merge"),
    (loop_scan_output, "one loop-carried output plus one that leaves the chain"),
    (loop_two_carried, "two values handed to the next iteration, i.e. two loop variables"),
    (loop_with_repeats, "a `Loop` whose body holds the repeats, main graph has one node"),
    (if_with_repeats, "an `If` whose two branches each hold repeats"),
    (planted_block, "one draw of the random planted-block generator used by the stress test"),
]

# ================================================================ What the outliner must say

# `config` is the non-default part of `OutlineConfig`; `pattern` is the report's
# pattern list as `(size, n_instance)`, in the report's own order (decreasing gain).
#
# The synthetic graphs are unary chains that onnxslim would happily rewrite, and
# `constant_attribute` only holds its trap while the `Constant` nodes are still
# nodes, so they are read with `preprocess=False`. The PyTorch exports are real
# models and get the default preprocessing.
EXPECT = {
    # -- topology of the repetition
    "serial_chain": dict(config=dict(preprocess=False), pattern=[(3, 4)], note="the easy case: one 3-node block, 4 times"),
    "serial_plus_parallel": dict(config=dict(preprocess=False), pattern=[(3, 4), (3, 2)], note="both chains are found; an arbitrary topological order finds neither"),
    "two_tower": dict(config=dict(preprocess=False), pattern=[(3, 4), (3, 4)], note="one pattern per tower, no interleaved false pattern"),
    "internal_branch": dict(config=dict(preprocess=False), pattern=[(5, 4)], note="the whole 5-node block including its internal fork"),
    "fan_out": dict(config=dict(preprocess=False), pattern=[(3, 5)], note="found even though no `Loop` could express it"),
    "shared_hub": dict(config=dict(preprocess=False), pattern=[(3, 5)], note="the shared tensor becomes one more function input"),
    "shared_accumulator": dict(config=dict(preprocess=False), pattern=[(3, 5)], note="the accumulator is threaded through as an input, not a barrier"),
    "nested_two_level": dict(config=dict(preprocess=False), pattern=[(3, 6)], note="level 0 takes the inner block; `max_level=2` then adds `(4, 2)`"),
    # -- traps for the rewriter
    "shared_input": dict(config=dict(preprocess=False), pattern=[(3, 4)], note="correct only if body inputs are keyed by (offset, slot)"),
    "constant_attribute": dict(config=dict(preprocess=False), pattern=[], note="no pattern at all is the right answer; merging them changes the result"),
    "ambiguous_sibling": dict(config=dict(preprocess=False), pattern=[(4, 4)], note="4, not 3: without edge disambiguation the side branch is left behind"),
    # -- shape-sensitive operators: is the shape an attribute or an input?
    "transpose_perm": dict(config=dict(preprocess=False), pattern=[(3, 2), (3, 2)], note="two patterns, not one of four: `perm` is an attribute and is baked into the body"),
    "concat_axis": dict(config=dict(preprocess=False), pattern=[(3, 2), (3, 2)], note="same as `transpose_perm`, but a wrong merge is not even runnable"),
    "reshape_shape_input": dict(config=dict(preprocess=False), pattern=[(4, 4)], note="one pattern of four: the target shape is an input, passed per call site"),
    "planted_block": dict(config=dict(preprocess=False), pattern=[(5, 2)], note="`method='serial'` alone reports (4, 2), lockstep growth recovers the 5th node"),
    # -- `Loop` / `If`
    "loop_scan_output": dict(config=dict(preprocess=False), pattern=[(3, 4)], note="with `backend='loop'`: 1 carried + 1 scan output + 4 slice nodes"),
    "loop_two_carried": dict(config=dict(preprocess=False), pattern=[(4, 4)], note="with `backend='loop'`: two loop-carried variables"),
    "loop_with_repeats": dict(config=dict(preprocess=False), pattern=[(3, 4)], note="found inside the body; `subgraph=False` finds nothing"),
    "if_with_repeats": dict(config=dict(preprocess=False), pattern=[(3, 4), (3, 4)], note="one pattern per branch"),
    # -- exported from PyTorch
    "flat_mlp": dict(config=dict(preprocess=False), pattern=[(4, 4)], note="Gemm/Relu/Gemm/Relu, the module boundary recovered from a flat export"),
    "flat_mlp_as_function": dict(config=dict(), pattern=[], note="PyTorch already outlined it; the existing function is carried through untouched"),
    "transformer_6layer": dict(config=dict(), pattern=[(41, 6)], note="the main target: 522 nodes -> 6"),
    "transformer_two_stage": dict(config=dict(), pattern=[(41, 6)], note="`max_level=2` turns this into 2 nodes of 124 original nodes each"),
    "transformer_branchy": dict(config=dict(), pattern=[(39, 5)], note="39 not 41: onnxslim's CSE merges a Transpose+Reshape shared by two layers"),
    "transformer_branchy_transitive": dict(config=dict(), pattern=[(39, 5)], note="the other reading of the ambiguous `A->D` edge; same 5 instances"),
    "moe_6expert": dict(config=dict(), pattern=[(4, 5)], note="the block is `Gemm/Relu/Gemm/Add`, and expert 0 has no `Add`, so 5 instances not 6"),
    "loop_body_repeats": dict(config=dict(preprocess=False), pattern=[(3, 4)], note="the main graph is one `Loop`, everything is inside the body"),
    "loop_shared_weight": dict(config=dict(preprocess=False), pattern=[], note="the body applies one block once, so there is nothing to fold"),
}

# ================================================================ Building the files on disk

def model_file(name: str) -> Path:
    """Where the zoo keeps `name`."""
    return model_path / f"{name}.onnx"

def ensure(name: str) -> Path:
    """Return the path of one zoo model, building it if it is not on disk yet."""
    onnx_file = model_file(name)
    if onnx_file.exists():
        return onnx_file
    model_path.mkdir(exist_ok=True)
    for builder, _ in SYNTHETIC:
        if builder.__name__ == name:
            onnx.save(builder(), onnx_file)
            return onnx_file
    for torch_name, builder, _ in torch_model_list():
        if torch_name == name:
            builder(onnx_file)
            return onnx_file
    raise KeyError(f"{name} is not in the model zoo, pick one of {sorted(name_list())}")

def name_list() -> list:
    """Every model name in the zoo."""
    return [builder.__name__ for builder, _ in SYNTHETIC] + [name for name, _, _ in torch_model_list()]
