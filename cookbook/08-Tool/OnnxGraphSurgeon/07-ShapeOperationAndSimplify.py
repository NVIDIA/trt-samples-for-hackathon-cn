#!/usr/bin/env python3
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

tensor0 = gs.Variable(name="tensor0", dtype=np.float32, shape=['A', 3, 'B', 5])
tensor1 = gs.Variable(name="tensor1", dtype=np.int64, shape=None)
tensor2 = gs.Variable(name="tensor2", dtype=np.int64, shape=None)
tensor3 = gs.Variable(name="tensor3", dtype=np.float32, shape=None)
tensor4 = gs.Variable(name="tensor4", dtype=np.int64, shape=None)
tensor5 = gs.Variable(name="tensor5", dtype=np.int64, shape=None)
tensor6 = gs.Variable(name="tensor6", dtype=np.int64, shape=None)
tensor7 = gs.Variable(name="tensor7", dtype=np.int64, shape=None)
tensor8 = gs.Variable(name="tensor8", dtype=np.float32, shape=None)

constant0 = gs.Constant(name="constant0", values=np.array([0, 1], dtype=np.int32))  # 定义张量（常量）
constant1 = gs.Constant(name="constant1", values=np.array([2, 3], dtype=np.int32))

node0 = gs.Node(name="myShape", op="Shape", inputs=[tensor0], outputs=[tensor1])  # value=(A,3,B,5), shape=(4,)
node1 = gs.Node(name="myReduceProd0", op="ReduceProd", inputs=[tensor1], attrs={"axes": [0], "keepdims": int(True)}, outputs=[tensor2])  # value=(A*3*B*5), shape=()
node2 = gs.Node(name="myReshape0", op="Reshape", inputs=[tensor0, tensor2], outputs=[tensor3])  # shape=(A*3*B*5,)

node3 = gs.Node(name="myGather0", op="Gather", inputs=[tensor1, constant0], outputs=[tensor4])  # value=(A,3), shape=(2,)
node4 = gs.Node(name="myGather1", op="Gather", inputs=[tensor1, constant1], outputs=[tensor5])  # value=(B,5), shape=(2,)
node5 = gs.Node(name="myReduceProd1", op="ReduceProd", inputs=[tensor5], attrs={"axes": [0], "keepdims": int(True)}, outputs=[tensor6])  # value=(B*5), shape=()
node6 = gs.Node(name="myConcat", op="Concat", inputs=[tensor4, tensor6], attrs={"axis": 0}, outputs=[tensor7])  # value=(A,3,B*5), shape=()
node7 = gs.Node(name="myReshape1", op="Reshape", inputs=[tensor0, tensor7], outputs=[tensor8])  # shape=(A*3*B*5,)

graph = gs.Graph(nodes=[node0, node1, node2, node3, node4, node5, node6, node7], inputs=[tensor0], outputs=[tensor3, tensor8])

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "model-07-01.onnx")

graph.inputs[0].shape = [2, 3, 4, 5]  # 如果是 static shape，则 fold_constants 可以化简计算图
graph.fold_constants().cleanup().toposort()
onnx.save(gs.export_onnx(graph), "model-07-02.onnx")
