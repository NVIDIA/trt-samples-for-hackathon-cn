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
"""Build the toy model that `main.sh` matches against `match_and_replace_plug/plugins/toyPlugin`.

The upstream Polygraphy example ships a `toy_subgraph.onnx`, which this cookbook does not carry
(`.onnx` files are git-ignored here), so it is generated instead. The graph is built to exercise
all three outcomes of the matcher:

```txt
    i0        i1                     i2
    |         |                      |
    A         B                      O          <- `O` is not part of the pattern
     \\       /                       |
       C (x=1)  <- matches, x < 2.0   A         <- an `A` with no `B` next to it
      /   \\                          |
     D     E                          C (x=9)   <- an `C` whose attribute fails the check
     |     |                          |
    o0     o1                        o2
```

So the pattern `A,B -> C(x<2) -> D,E` matches exactly once, and the two decoys are there to make
the "no match because ..." lines in the log meaningful.
"""

from pathlib import Path

import onnx

output_file = Path(__file__).parent / "toy_subgraph.onnx"

def value(name: str):
    """A float32 tensor of unknown shape. The ops are fictional, so only the topology matters."""
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)

node_list = [
    # The subgraph that matches the toyPlugin pattern
    onnx.helper.make_node("A", ["i0"], ["a0"], "n_a0"),
    onnx.helper.make_node("B", ["i1"], ["b0"], "n_b0"),
    onnx.helper.make_node("C", ["a0", "b0"], ["c0"], "n_c0", x=1),
    onnx.helper.make_node("D", ["c0"], ["o0"], "n_d0"),
    onnx.helper.make_node("E", ["c0"], ["o1"], "n_e0"),
    # Decoy 1: an op that is not in the pattern at all
    onnx.helper.make_node("O", ["i2"], ["t0"], "n_o0"),
    # Decoy 2 + 3: right ops, but no `B` feeding `C`, and `x = 9` fails `check_func` (`x < 2.0`)
    onnx.helper.make_node("A", ["t0"], ["a1"], "n_a1"),
    onnx.helper.make_node("C", ["a1", "a1"], ["c1"], "n_c1", x=9),
    onnx.helper.make_node("D", ["c1"], ["o2"], "n_d1"),
]

graph = onnx.helper.make_graph(
    node_list,
    "toy_subgraph",
    [value("i0"), value("i1"), value("i2")],
    [value("o0"), value("o1"), value("o2")],
)
# `A` / `B` / ... are made-up ops, so the model is deliberately not runnable and not checkable.
# `polygraphy plugin` only ever imports it with ONNX-GraphSurgeon and rewrites the topology.
model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
model.ir_version = 10
onnx.save(model, output_file)
print(f"Succeed saving {output_file.name}: {len(node_list)} nodes")
