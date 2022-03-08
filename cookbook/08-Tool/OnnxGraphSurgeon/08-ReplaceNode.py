#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#

import onnx
import onnx_graphsurgeon as gs
import numpy as np


# 生成 .onnx
@gs.Graph.register()
def min(self, *args):
    return self.layer(op="Min", inputs=args, outputs=["min_out"])[0]

@gs.Graph.register()
def max(self, *args):
    return self.layer(op="Max", inputs=args, outputs=["max_out"])[0]

@gs.Graph.register()
def identity(self, inp):
    return self.layer(op="Identity", inputs=[inp], outputs=["identity_out"])[0]

graph = gs.Graph()
graph.inputs = [gs.Variable("input", shape=(4, 4), dtype=np.float32)]

# Clip values to [0, 6]
MIN_VAL = np.array(0, np.float32)
MAX_VAL = np.array(6, np.float32)

# Add identity nodes to make the graph structure a bit more interesting
inp = graph.identity(graph.inputs[0])
max_out = graph.max(graph.min(inp, MAX_VAL), MIN_VAL)
graph.outputs = [
    graph.identity(max_out),
]

# Graph outputs must include dtype information
graph.outputs[0].to_variable(dtype=np.float32, shape=(4, 4))

onnx.save(gs.export_onnx(graph), "08-ReplaceNode_0.onnx")

# 读取 .onnx 并进行调整
# Here we'll register a function to do all the subgraph-replacement heavy-lifting.
# NOTE: Since registered functions are entirely reusable, it may be a good idea to
# refactor them into a separate module so you can use them across all your models.
@gs.Graph.register()
def replace_with_clip(self, inputs, outputs):
    # Disconnect output nodes of all input tensors
    for inp in inputs:
        inp.outputs.clear()

    # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()

    # Insert the new node.
    return self.layer(op="Clip", inputs=inputs, outputs=outputs)

# Now we'll do the actual replacement
graph = gs.import_onnx(onnx.load("08-ReplaceNode_0.onnx"))

tmap = graph.tensors()
# You can figure out the input and output tensors using Netron. In our case:
# Inputs: [inp, MIN_VAL, MAX_VAL]
# Outputs: [max_out]
inputs = [tmap["identity_out_0"], tmap["onnx_graphsurgeon_constant_5"], tmap["onnx_graphsurgeon_constant_2"]]
outputs = [tmap["max_out_6"]]

graph.replace_with_clip(inputs, outputs)

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "08-ReplaceNode_1.onnx")
