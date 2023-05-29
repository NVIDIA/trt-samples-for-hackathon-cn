#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

x = gs.Variable("x", np.float32, [1, 3, 5, 5])
_0 = gs.Variable("_0", np.float32, [1, 3, 5, 5])
_1 = gs.Variable("_1", np.float32, [1, 3, 5, 5])
y = gs.Variable("y", np.float32, [1, 3, 5, 5])

node0 = gs.Node(op="Identity", inputs=[x], outputs=[_0])
node1 = gs.Node(op="UnknownNode", inputs=[_0], outputs=[_1])
node2 = gs.Node(op="Identity", inputs=[_1], outputs=[y])

graph = gs.Graph(nodes=[node0, node1, node2], inputs=[x], outputs=[y])
onnx.save(gs.export_onnx(graph), "model-UnknowNode.onnx")
