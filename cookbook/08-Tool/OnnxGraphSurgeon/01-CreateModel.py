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

x = gs.Variable(name="x", dtype=np.float32, shape=[1, 3, 5, 5])
y = gs.Variable(name="y", dtype=np.float32, shape=[1, 3, 1, 1])
node = gs.Node(op="GlobalLpPool", attrs={"p": 2}, inputs=[x], outputs=[y])

graph = gs.Graph(nodes=[node], inputs=[x], outputs=[y])
onnx.save(gs.export_onnx(graph), "01-CreateModel.onnx")
