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

import numpy as np
import onnx
import onnx_graphsurgeon as gs

tensor0 = gs.Variable("tensor0", np.float32, ['B', 3, 64, 64])  # 三个真正有用的张量
tensor1 = gs.Variable("tensor1", np.float32, ['B', 3, 64, 64])
tensor2 = gs.Variable("tensor2", np.float32, ['B', 3, 64, 64])
tensor3 = gs.Variable("tensor3", np.float32, ['B', 3, 64, 64])  # 一个假输入张量
tensor4 = gs.Variable("tensor4", np.float32, ['B', 1, 64, 64])  # 一个假输出张量
tensor5 = gs.Variable("tensor5", np.float32, ['B', 1, 64, 64])  # 两个无用张量
tensor6 = gs.Variable("tensor6", np.float32, ['B', 1, 64, 64])
tensor7 = gs.Variable("tensor7", np.float32, None)  # 中间结果张量
tensor8 = gs.Variable("tensor8", np.float32, None)

constant0 = gs.Constant(name="w", values=np.ones(shape=[1, 1, 1, 1], dtype=np.float32))

node0 = gs.Node("Add", "myAdd0", inputs=[constant0, constant0], outputs=[tensor7])
node1 = gs.Node("Add", "myAdd1", inputs=[tensor7, constant0], outputs=[tensor8])
node2 = gs.Node("Add", "myAdd2", inputs=[tensor0, tensor8], outputs=[tensor1])  # 有效节点
node3 = gs.Node("Add", "myAdd3", inputs=[tensor1, constant0], outputs=[tensor2])  # 有效节点
node4 = gs.Node("Add", "myAdd4", inputs=[tensor5, constant0], outputs=[tensor6])  # 无效节点

graph = gs.Graph(nodes=[node4, node3, node2, node1, node0], inputs=[tensor0, tensor3], outputs=[tensor2, tensor4])

onnx.save(gs.export_onnx(graph), "model-06-01.onnx")  # 原始计算图，可见 4 个无边张量和 1 个无边的节点，还有 1 个常数计算链
onnx.save(gs.export_onnx(graph.fold_constants()), "model-06-02.onnx")  # 常数折叠后的计算图，常数计算链合并到主链中，多出 2 个无边 Add 节点，注意常数折叠并不做节点融合的工作，主链上两个 Add 没有合并掉
onnx.save(gs.export_onnx(graph.fold_constants().cleanup()), "model-06-03.onnx")  # 打扫后的计算图，可见 3 个无用的 Add 节点被清除

print("Before toposort:")  # 原始节点顺序
for index, node in enumerate(graph.nodes):
    print("No.%d->%s" % (index, node.name))

print("After toposort:")  # 拓扑排序后的节点顺序，节点基本按照计算图的计算顺序进行排列
graph.cleanup().toposort()
for index, node in enumerate(graph.nodes):
    print("No.%d->%s" % (index, node.name))

graph.inputs = [tensor0]
graph.outputs = [tensor2]
onnx.save(gs.export_onnx(graph), "model-06-04.onnx")  # 去掉多与输入输出的计算图，才能正确被 TensorRT 处理
