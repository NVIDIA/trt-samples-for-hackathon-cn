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

from collections import OrderedDict
import numpy as np
import onnx
import onnx_graphsurgeon as gs

tensor0 = gs.Variable("tensor0", np.float32, ["B", 1, 64, 64])
tensor1 = gs.Variable("tensor1", np.float32, ["B", 1, 64, 64])
node0 = gs.Node("AddScalar", "myAddAcalar", inputs=[tensor0], outputs=[tensor1], attrs=OrderedDict([('scalar', np.array([10],dtype=np.float32))]))
graph = gs.Graph(nodes=[node0], inputs=[tensor0], outputs=[tensor1])

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "./model.onnx")

np.random.seed(31193)
dd = {}
dd["inferenceData"] = np.random.rand(4 * 1 * 64 * 64).astype(np.float32).reshape([4, 1, 64, 64])
np.savez("data.npz",**dd)
