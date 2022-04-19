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

from collections import OrderedDict
import numpy as np
import onnx
import onnx_graphsurgeon as gs

tensor0 = gs.Variable(name="tensor0", dtype=np.float32, shape=['B', 3, 64, 64])  # 定义张量（变量）
tensor1 = gs.Variable(name="tensor1", dtype=np.float32, shape=['B', 1, 64, 64])
tensor2 = gs.Variable(name="tensor2", dtype=np.float32, shape=None)  # 可以不知道形状或者数据类型
tensor3 = gs.Variable(name="tensor3", dtype=np.float32, shape=None)

constant0 = gs.Constant(name="constant0", values=np.ones(shape=[1, 3, 3, 3], dtype=np.float32))  # 定义张量（常量）
constant1 = gs.Constant(name="constant1", values=np.ones(shape=[1], dtype=np.float32))

node0 = gs.Node(name="myConv", op="Conv", inputs=[tensor0, constant0], outputs=[tensor1])  # 定义节点，使用张量作为输入和输出
node0.attrs = OrderedDict([
    ('dilations', [1, 1]),
    ('kernel_shape', [3, 3]),
    ('pads', [1, 1, 1, 1]),
    ('strides', [1, 1]),
])  # 节点的属性参数

node1 = gs.Node(name="myAdd", op="Add", inputs=[tensor1, constant1], outputs=[tensor2])
node2 = gs.Node(name="myRelu", op="Relu", inputs=[tensor2], outputs=[tensor3])

graph = gs.Graph(nodes=[node0, node1, node2], inputs=[tensor0], outputs=[tensor3])  # 定义计算图，要求给出所有节点和输入输出张量

graph.cleanup().toposort()  # 保存计算图前的收尾工作，详细作用见 06-Fold.py
onnx.save(gs.export_onnx(graph), "model-01.onnx")
