#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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

from pathlib import Path

onnx_file = f"model-{Path(__file__).name.split('.')[0]}"

tensor0 = gs.Variable("tensor0", np.float32, ["A", 3, "B", 5])
tensor1 = gs.Variable("tensor1", np.int64, None)
tensor2 = gs.Variable("tensor2", np.int64, None)
tensor3 = gs.Variable("tensor3", np.float32, None)
tensor4 = gs.Variable("tensor4", np.int64, None)
tensor5 = gs.Variable("tensor5", np.int64, None)
tensor6 = gs.Variable("tensor6", np.int64, None)
tensor7 = gs.Variable("tensor7", np.int64, None)
tensor8 = gs.Variable("tensor8", np.float32, None)
constant0 = gs.Constant("constant0", values=np.array([0, 1], dtype=np.int32))
constant1 = gs.Constant("constant1", values=np.array([2, 3], dtype=np.int32))

node0 = gs.Node("Shape", "myShape", inputs=[tensor0], outputs=[tensor1])  # value=(A,3,B,5), shape=(4,)
node1 = gs.Node("ReduceProd", "myReduceProd0", inputs=[tensor1], attrs={"axes": [0], "keepdims": int(True)}, outputs=[tensor2])  # value=(A*3*B*5), shape=()
node2 = gs.Node("Reshape", "myReshape0", inputs=[tensor0, tensor2], outputs=[tensor3])  # shape=(A*3*B*5,)
node3 = gs.Node("Gather", "myGather0", inputs=[tensor1, constant0], outputs=[tensor4])  # value=(A,3), shape=(2,)
node4 = gs.Node("Gather", "myGather1", inputs=[tensor1, constant1], outputs=[tensor5])  # value=(B,5), shape=(2,)
node5 = gs.Node("ReduceProd", "myReduceProd1", inputs=[tensor5], attrs={"axes": [0], "keepdims": int(True)}, outputs=[tensor6])  # value=(B*5), shape=()
node6 = gs.Node("Concat", "myConcat", inputs=[tensor4, tensor6], attrs={"axis": 0}, outputs=[tensor7])  # value=(A,3,B*5), shape=()
node7 = gs.Node("Reshape", "myReshape1", inputs=[tensor0, tensor7], outputs=[tensor8])  # shape=(A*3*B*5,)

graph = gs.Graph(nodes=[node0, node1, node2, node3, node4, node5, node6, node7], inputs=[tensor0], outputs=[tensor3, tensor8])

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), onnx_file + "-01.onnx")  # using Dynamic Shape mode, there are many shape operators in the graph

graph.inputs[0].shape = [2, 3, 4, 5]  # shape operators can be simplified if the shape is static
graph.fold_constants().cleanup().toposort()
onnx.save(gs.export_onnx(graph), onnx_file + "-02.onnx")

print("Finish")
