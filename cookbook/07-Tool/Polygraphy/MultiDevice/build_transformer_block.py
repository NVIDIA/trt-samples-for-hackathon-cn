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
"""Build the toy transformer block that `main.sh` shards.

`polygraphy multi-device` does not shard "a model", it shards **the two patterns it recognises**,
and the two modes look for completely different things
(`polygraphy/tools/multi_device/subtool/shard.py`):

+ CP (`get_attention_pattern`) matches a plain attention body:

```txt
    Q    K
     \\  /
     MatMul -> Softmax -> MatMul(., V) -> output
```

+ TP (`TPManager.tp_match_mlp`) matches a **SwiGLU MLP**, keyed off a `Sigmoid` node:

```txt
    x -> MatMul(Wg) -> Sigmoid -.
      \\        \\               Mul  -> Mul -> MatMul(Wd) -> output
       \\        `--------------'      /
        `----> MatMul(Wu) -------------'
```

  (TP also matches an `AttentionPlugin` node, which is a TensorRT-LLM op, not something an ordinary
  ONNX export contains, so the MLP is the reachable half here.)

No model under `00-Data/model/` holds either pattern, and running the tools on a model without them
is not a mistake that reports itself: the hints file is written happily with
`"attention_layers": []` and sharding then rewrites nothing at all. So the block is generated here,
small but real (it runs in onnxruntime), which is what makes the multi-GPU numerical check of
case 04 possible once a multi-GPU machine is available.

`K` is fed already transposed (`[B, H, S]`), like a real exported attention block does, so that no
`Transpose` sits between the input and the first `MatMul`, where the pattern does not expect one.
"""

from pathlib import Path

import numpy as np
import onnx

output_file = Path(__file__).parent / "model-transformer-block.onnx"

B, S, H, I = 1, 4, 8, 16  # batch, sequence length, hidden size, intermediate size of the MLP

rng = np.random.default_rng(31193)

def value(name: str, shape: list):
    """A float32 tensor of the given shape."""
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)

def weight(name: str, shape: list):
    """A small random initializer, scaled so that the output of the block stays in a sane range."""
    return onnx.numpy_helper.from_array((rng.random(shape).astype(np.float32) - 0.5) / shape[0], name)

node_list = [
    # ---- Attention: what CP shards
    onnx.helper.make_node("MatMul", ["q", "k"], ["qk"], "node_qk"),
    onnx.helper.make_node("Softmax", ["qk"], ["attention_weight"], "node_softmax", axis=-1),
    onnx.helper.make_node("MatMul", ["attention_weight", "v"], ["context"], "node_context"),
    onnx.helper.make_node("MatMul", ["context", "w_o"], ["attention_out"], "node_o_projection"),
    # ---- SwiGLU MLP: what TP shards. `gate` and `up` are column-parallel, `down` is row-parallel.
    onnx.helper.make_node("MatMul", ["attention_out", "w_gate"], ["gate"], "node_gate_projection"),
    onnx.helper.make_node("Sigmoid", ["gate"], ["gate_sigmoid"], "node_sigmoid"),
    onnx.helper.make_node("Mul", ["gate", "gate_sigmoid"], ["silu"], "node_silu"),
    onnx.helper.make_node("MatMul", ["attention_out", "w_up"], ["up"], "node_up_projection"),
    onnx.helper.make_node("Mul", ["silu", "up"], ["hidden"], "node_gating"),
    onnx.helper.make_node("MatMul", ["hidden", "w_down"], ["y"], "node_down_projection"),
]

graph = onnx.helper.make_graph(
    node_list,
    "toy_transformer_block",
    [value("q", [B, S, H]), value("k", [B, H, S]), value("v", [B, S, H])],
    [value("y", [B, S, H])],
    [weight("w_o", [H, H]), weight("w_gate", [H, I]), weight("w_up", [H, I]), weight("w_down", [I, H])],
)
model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
model.ir_version = 10
onnx.checker.check_model(model)
onnx.save(model, output_file)
print(f"Succeed saving {output_file.name}: {len(node_list)} nodes, B={B} S={S} H={H} I={I}")
