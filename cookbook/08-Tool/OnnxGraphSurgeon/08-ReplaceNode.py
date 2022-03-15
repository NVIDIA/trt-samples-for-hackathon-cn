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

# 剪切范围 [0, 6]
MIN_VAL = np.array(0, np.float32)
MAX_VAL = np.array(6, np.float32)

_0 = graph.identity(graph.inputs[0])
_1 = graph.max(graph.min(_0, MAX_VAL), MIN_VAL)
_2 = graph.identity(_1)
graph.outputs = [_2]
graph.outputs[0].to_variable(dtype=np.float32, shape=(4, 4))

onnx.save(gs.export_onnx(graph), "08-ReplaceNode_0.onnx")

# 读取 .onnx 并进行调整
# 重新创建一个函数（节点）用于剪切
@gs.Graph.register()
def replace_with_clip(self, inputs, outputs):
    # 砍掉末尾节点的输出张量和头节点的输入张量
    for inp in inputs:
        inp.outputs.clear()
    for out in outputs:
        out.inputs.clear()
    # 插入新节点
    return self.layer(op="Clip", inputs=inputs, outputs=outputs)

graph = gs.import_onnx(onnx.load("08-ReplaceNode_0.onnx"))
tmap = graph.tensors()

# 手工找出要砍掉的输入和输出张量，交给 replace_with_clip 函数
inputs = [tmap["identity_out_0"], tmap["onnx_graphsurgeon_constant_5"], tmap["onnx_graphsurgeon_constant_2"]]
outputs = [tmap["max_out_6"]]
graph.replace_with_clip(inputs, outputs)

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "08-ReplaceNode_1.onnx")
