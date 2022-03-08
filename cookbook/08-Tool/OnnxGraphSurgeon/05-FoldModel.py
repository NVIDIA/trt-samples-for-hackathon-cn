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
import onnxruntime
import onnx_graphsurgeon as gs
import numpy as np

# 生成 .onnx
shape = (1, 3)
x = gs.Variable("x", shape=shape, dtype=np.float32)
y = gs.Variable("y", shape=shape, dtype=np.float32)
a = gs.Constant("a", values=np.ones(shape=shape, dtype=np.float32))
b = gs.Constant("b", values=np.ones(shape=shape, dtype=np.float32))
c = gs.Variable("c")
d = gs.Constant("d", values=np.ones(shape=shape, dtype=np.float32))
e = gs.Variable("e")

nodes = [
    gs.Node("Add", inputs=[a, b], outputs=[c]),
    gs.Node("Add", inputs=[c, d], outputs=[e]),
    gs.Node("Add", inputs=[x, e], outputs=[y]),
]

graph = gs.Graph(nodes=nodes, inputs=[x], outputs=[y])
onnx.save(gs.export_onnx(graph), "05-FoldModel_0.onnx")

# 读取 .onnx 并进行调整
graph = gs.import_onnx(onnx.load("05-FoldModel_0.onnx"))

session = onnxruntime.InferenceSession("05-FoldModel_0.onnx", providers=['CUDAExecutionProvider'])

graph.fold_constants().cleanup()
onnx.save(gs.export_onnx(graph), "05-FoldModel_1.onnx")
