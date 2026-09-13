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
"""Clean up the noise a framework exporter leaves behind, the PackNet way.

`samples/python/onnx_packnet` in TensorRT-OSS is built around three graph rewrites that
undo what the PyTorch exporter does to a depth-estimation network: a `Pad` whose padding
amounts arrive through a 14-node constant subgraph, a `Resize` whose scales arrive the
same way, and a `GroupNorm` that comes out as `Reshape`+`InstanceNormalization`+`Mul`+`Add`.

Two of those three rewrites are **no longer worth writing**, and this example measures
that instead of asserting it:

+ the Pad and Resize subgraphs are pure constant arithmetic, so `fold_constants()`
  removes them for free -- 33 nodes down to 12 here, with no pattern matching at all.
+ the GroupNorm pattern survives folding, because it depends on a runtime `Shape`.
  That one still needs a real rewrite, and it is the interesting half.

The upstream rewrite replaces GroupNorm with a `GroupNormalizationPlugin` node. **Do not
copy that part**: the plugin has been unsupported on Blackwell and later since TensorRT
10.7, and its own README tells you to use `INormalizationLayer` instead. So this example
rewrites the subgraph into the *native* ONNX `GroupNormalization` operator (opset 21),
which the TensorRT parser turns into a single `LayerType.NORMALIZATION`.

The other lesson worth carrying over is negative: the upstream code reaches into the
subgraph with chains like `node.i(1).i(0).i(0).i(0).i(0).i(0)` and needs three different
versions of that chain for three ranges of `torch.__version__`. Matching on the *pattern*
and reading values out of the matched nodes, as below, does not care which exporter
produced the graph.
"""

import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime
import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(31193)
torch.manual_seed(31193)

N_CHANNEL = 8
N_GROUP = 4
OPSET_EXPORT = 11  # What the legacy exporter is usually asked for, and what upstream targets
OPSET_REWRITTEN = 21  # `GroupNormalization` became a standard operator in opset 21

onnx_file = Path(f"model_{Path(__file__).name.split('.')[0]}")
onnx_file_0 = Path(str(onnx_file) + "_00.onnx")  # Straight out of the exporter
onnx_file_1 = Path(str(onnx_file) + "_01.onnx")  # After constant folding
onnx_file_2 = Path(str(onnx_file) + "_02.onnx")  # After the GroupNorm rewrite

data = {"x": np.random.rand(1, 3, 16, 16).astype(np.float32)}

# ------------------------------------------------------------------------------------------------
# A model small enough to read, shaped to reproduce all three exporter artefacts

class Net(nn.Module):
    """Conv -> GroupNorm -> ReLU -> pad -> Conv -> interpolate -> ReLU."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, N_CHANNEL, 3, padding=1)
        self.group_norm = nn.GroupNorm(N_GROUP, N_CHANNEL)
        self.conv2 = nn.Conv2d(N_CHANNEL, N_CHANNEL, 3)
        # Random affine parameters, otherwise scale=1 / bias=0 would hide a wrong rewrite
        nn.init.uniform_(self.group_norm.weight, 0.5, 1.5)
        nn.init.uniform_(self.group_norm.bias, -0.5, 0.5)

    def forward(self, x):
        x = F.relu(self.group_norm(self.conv1(x)))
        x = F.pad(x, (1, 1, 1, 1), mode="constant", value=0.0)
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return F.relu(x)

def count_op(graph) -> OrderedDict:
    """Node count per operator type, for the before/after tables."""
    counter = OrderedDict()
    for node in graph.nodes:
        counter[node.op] = counter.get(node.op, 0) + 1
    return OrderedDict(sorted(counter.items()))

def run_onnxruntime(onnx_file: Path) -> np.ndarray:
    """Numerical reference, so every rewrite is checked rather than eyeballed."""
    option = onnxruntime.SessionOptions()
    option.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(str(onnx_file), option, providers=["CPUExecutionProvider"])
    return session.run(None, data)[0]

def count_trt_layer(onnx_file: Path) -> int:
    """Parse with TensorRT and report the layer count, or -1 if the parser refuses."""
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_file)):
        for i in range(parser.num_errors):
            print(f"    {parser.get_error(i)}")
        return -1
    return network.num_layers

# ------------------------------------------------------------------------------------------------
# The one rewrite that constant folding cannot do

def fuse_group_norm(graph) -> int:
    """`Reshape -> InstanceNormalization -> Reshape -> Mul -> Add` becomes `GroupNormalization`.

    Everything the new node needs is read out of the matched nodes themselves:

    + `num_groups` from the first Reshape's target shape, which the exporter writes as
      `[0, num_groups, -1]`,
    + `epsilon` from the `InstanceNormalization` attribute,
    + per-channel `scale` / `bias` from the constant inputs of `Mul` / `Add`, which the
      exporter shapes as `[C, 1, 1]` and the operator wants as `[C]`.

    Returns the number of subgraphs replaced.
    """
    n_replaced = 0
    for node in [n for n in graph.nodes if n.op == "InstanceNormalization"]:
        reshape_in = node.i(0)
        if reshape_in.op != "Reshape" or not isinstance(reshape_in.inputs[1], gs.Constant):
            continue
        # The consumers must be exactly Reshape -> Mul -> Add for this to be a GroupNorm
        try:
            reshape_out = node.outputs[0].outputs[0]
            mul = reshape_out.outputs[0].outputs[0]
            add = mul.outputs[0].outputs[0]
        except IndexError:
            continue
        if (reshape_out.op, mul.op, add.op) != ("Reshape", "Mul", "Add"):
            continue

        scale_constant = [t for t in mul.inputs if isinstance(t, gs.Constant)]
        bias_constant = [t for t in add.inputs if isinstance(t, gs.Constant)]
        if not scale_constant or not bias_constant:
            continue

        num_groups = int(reshape_in.inputs[1].values[1])
        epsilon = float(node.attrs.get("epsilon", 1e-5))
        scale = gs.Constant(f"{node.name}_scale", np.ascontiguousarray(scale_constant[0].values.reshape(-1).astype(np.float32)))
        bias = gs.Constant(f"{node.name}_bias", np.ascontiguousarray(bias_constant[0].values.reshape(-1).astype(np.float32)))

        graph.nodes.append(gs.Node(
            "GroupNormalization",
            f"{node.name}_fused",
            inputs=[reshape_in.inputs[0], scale, bias],
            outputs=[add.outputs[0]],
            attrs=OrderedDict([("num_groups", num_groups), ("epsilon", epsilon)]),
        ))
        add.outputs.clear()  # Detach the old tail so `cleanup()` can collect the whole subgraph
        n_replaced += 1

    graph.cleanup().toposort()
    return n_replaced

# ------------------------------------------------------------------------------------------------

def main() -> None:
    # ---- Export straight out of PyTorch
    model = Net().eval()
    with warnings.catch_warnings():  # The legacy exporter is noisy about Slice constant folding
        warnings.simplefilter("ignore")
        torch.onnx.export(model, (torch.from_numpy(data["x"]), ), onnx_file_0, dynamo=False, opset_version=OPSET_EXPORT, input_names=["x"], output_names=["y"])
    graph = gs.import_onnx(onnx.load(onnx_file_0))
    print(f"[00] straight from the exporter : {len(graph.nodes):>3} nodes")
    print(f"     {dict(count_op(graph))}")
    reference = run_onnxruntime(onnx_file_0)

    # ---- Rewrite 1 and 2, for free
    graph.fold_constants().cleanup().toposort()
    onnx.save(gs.export_onnx(graph), onnx_file_1)
    print(f"[01] after fold_constants()     : {len(graph.nodes):>3} nodes")
    print(f"     {dict(count_op(graph))}")
    print("     The Pad and Resize input subgraphs are gone; upstream hand-writes both of these.")
    output_1 = run_onnxruntime(onnx_file_1)
    print(f"     max |diff| vs exporter output: {np.max(np.abs(output_1 - reference)):.3e}")

    # ---- Rewrite 3, which folding cannot do
    n_replaced = fuse_group_norm(graph)
    model_onnx = gs.export_onnx(graph)
    del model_onnx.opset_import[:]
    model_onnx.opset_import.extend([onnx.helper.make_opsetid("", OPSET_REWRITTEN)])
    model_onnx.ir_version = 10
    onnx.checker.check_model(model_onnx)
    onnx.save(model_onnx, onnx_file_2)
    print(f"[02] after the GroupNorm rewrite: {len(graph.nodes):>3} nodes ({n_replaced} subgraph(s) replaced)")
    print(f"     {dict(count_op(graph))}")
    output_2 = run_onnxruntime(onnx_file_2)
    print(f"     max |diff| vs exporter output: {np.max(np.abs(output_2 - reference)):.3e}")

    # ---- What TensorRT makes of each
    print("\n" + "=" * 78)
    print(f"{'Stage':<34}{'ONNX nodes':>12}{'TRT layers':>12}{'max|diff|':>16}")
    print("-" * 78)
    for name, path, output in [("exporter output", onnx_file_0, reference), ("+ fold_constants", onnx_file_1, output_1), ("+ GroupNormalization rewrite", onnx_file_2, output_2)]:
        n_node = len(gs.import_onnx(onnx.load(path)).nodes)
        print(f"{name:<34}{n_node:>12}{count_trt_layer(path):>12}{np.max(np.abs(output - reference)):>16.3e}")
    print("=" * 78)

    assert n_replaced == 1, "The GroupNorm subgraph was not matched"
    assert np.max(np.abs(output_2 - reference)) < 1e-5, "The rewritten graph does not match the original"
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
