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

# 创建节点
# 使用 onnx_graphsurgeon.Graph.register() 将一个函数注册为计算图
@gs.Graph.register()
def add(self, a, b):
    return self.layer(op="Add", inputs=[a, b], outputs=["add_out_gs"])

@gs.Graph.register()
def mul(self, a, b):
    return self.layer(op="Mul", inputs=[a, b], outputs=["mul_out_gs"])

@gs.Graph.register()
def gemm(self, a, b, trans_a=False, trans_b=False):
    attrs = {"transA": int(trans_a), "transB": int(trans_b)}
    return self.layer(op="Gemm", inputs=[a, b], outputs=["gemm_out_gs"], attrs=attrs)

# 注册函数时注明其版本
@gs.Graph.register(opsets=[11])
def relu(self, a):
    return self.layer(op="Relu", inputs=[a], outputs=["act_out_gs"])

# 注册其他版本的同名函数，在 graph 创建时只会选用指定版本的函数
@gs.Graph.register(opsets=[1])
def relu(self, a):
    raise NotImplementedError("This function has not been implemented!")

# 创建计算图
graph = gs.Graph(opset=11)
x = gs.Variable(name="x", shape=(64, 64), dtype=np.float32)
a = np.ones(shape=(64, 64), dtype=np.float32)
b = np.ones((64, 64), dtype=np.float32) * 0.5
c = gs.Constant(name="c", values=np.ones(shape=(64, 64), dtype=np.float32))
d = np.ones(shape=(64, 64), dtype=np.float32)

_0 = graph.gemm(a, x, trans_b=True)
_1 = graph.relu(*graph.add(*_0, b))
_2 = graph.add(*graph.mul(*_1, c), d)
graph.inputs = [x]
graph.outputs = _2

# 指定计算图的输出精度
for out in graph.outputs:
    out.dtype = np.float32

onnx.save(gs.export_onnx(graph), "07-BuildModelWithAPI.onnx")
