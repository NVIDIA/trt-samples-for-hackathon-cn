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
import os
import tensorrt as trt

nLoop = 10
nC = 32
onnxFile0 = "model-0.onnx"
onnxFile1 = "model-1.onnx"

tensor0 = gs.Variable("tensor-0", np.float32, ["B", 1, 16, 16])
constant32x1 = gs.Constant("constant32x1", np.ascontiguousarray(np.random.rand(nC, 1, 3, 3).reshape(nC, 1, 3, 3).astype(np.float32) * 2 - 1))
constant32x32 = gs.Constant("constant32x32", np.ascontiguousarray(np.random.rand(nC, nC, 3, 3).reshape(nC, nC, 3, 3).astype(np.float32) * 2 - 1))
constant32 = gs.Constant("constant32", np.ascontiguousarray(np.random.rand(1, nC, 1, 1).reshape(1, nC, 1, 1).astype(np.float32) * 2 - 1))
constant32t = gs.Constant("constant32t", np.ascontiguousarray(np.random.rand(1, 1, 1, nC).reshape(1, 1, 1, nC).astype(np.float32) * 2 - 1))
constant1x32 = gs.Constant("constant1x32", np.ascontiguousarray(np.random.rand(1, nC, 3, 3).reshape(1, nC, 3, 3).astype(np.float32) * 2 - 1))
constant1 = gs.Constant("constant1", np.ascontiguousarray(np.array([1], dtype=np.int64)))
constant32r = gs.Constant("constant32r", np.ascontiguousarray(np.random.rand(1, nC, 1, 1).reshape(1, nC, 1, 1).astype(np.float32) * 2 - 1))

graphNodeList = []

tensor1 = gs.Variable("tensor-1", np.float32, None)
node1 = gs.Node("Conv", "Conv0", inputs=[tensor0, constant32x1], outputs=[tensor1])
node1.attrs = OrderedDict([("kernel_shape", [3, 3]), ("pads", [1, 1, 1, 1])])
"""
node1.attrs = OrderedDict([
    ("dilations", [1, 1]),
    ("kernel_shape", [3, 3]),
    ("pads", [1, 1, 1, 1]),
    ("strides", [1, 1]),
])
"""
graphNodeList.append(node1)

tensorLoop = tensor1
for i in range(nLoop // 2):
    tensor2 = gs.Variable("tensor-%d-1" % i, np.float32, None)
    node2 = gs.Node("Conv", "Conv-" + str(i), inputs=[tensorLoop, constant32x32], outputs=[tensor2])
    node2.attrs = OrderedDict([("kernel_shape", [3, 3]), ("pads", [1, 1, 1, 1])])
    graphNodeList.append(node2)

    tensor3 = gs.Variable("tensor-%d-2" % i, np.float32, None)
    node3 = gs.Node("Unsqueeze", "Unsqueeze-%d" + str(i), inputs=[tensor2, constant1], outputs=[tensor3])
    graphNodeList.append(node3)

    tensor4 = gs.Variable("tensor-%d-3" % i, dtype=np.float32, shape=None)
    node4 = gs.Node("Add", "Add-" + str(i), inputs=[tensor3, constant32], outputs=[tensor4])
    graphNodeList.append(node4)

    tensor5 = gs.Variable("tensor-%d-4" % i, np.float32, None)
    node5 = gs.Node("Squeeze", "Squeeze-%d" + str(i), inputs=[tensor4, constant1], outputs=[tensor5])
    graphNodeList.append(node5)

    tensor6 = gs.Variable("tensor-%d-5" % i, dtype=np.float32, shape=None)
    node6 = gs.Node("Relu", "ReLU-" + str(i), inputs=[tensor5], outputs=[tensor6])
    graphNodeList.append(node6)

    tensorLoop = tensor6

for i in range(nLoop // 2, nLoop):
    tensor2 = gs.Variable("tensor-%d-1" % i, np.float32, None)
    node2 = gs.Node("Conv", "Conv-" + str(i), inputs=[tensorLoop, constant32x32], outputs=[tensor2])
    node2.attrs = OrderedDict([("kernel_shape", [3, 3]), ("pads", [1, 1, 1, 1])])
    graphNodeList.append(node2)

    tensor3 = gs.Variable("tensor-%d-2" % i, np.float32, None)
    node3 = gs.Node("Transpose", "Transpose-%d" + str(i), inputs=[tensor2], outputs=[tensor3], attrs=OrderedDict([("perm", [0, 2, 3, 1])]))
    graphNodeList.append(node3)

    tensor4 = gs.Variable("tensor-%d-3" % i, dtype=np.float32, shape=None)
    node4 = gs.Node("Add", "Add-" + str(i), inputs=[tensor3, constant32t], outputs=[tensor4])
    graphNodeList.append(node4)

    tensor5 = gs.Variable("tensor-%d-4" % i, np.float32, None)
    node5 = gs.Node("Transpose", "Transpose-%d" + str(i), inputs=[tensor4], outputs=[tensor5], attrs=OrderedDict([("perm", [0, 3, 1, 2])]))
    graphNodeList.append(node5)

    tensor6 = gs.Variable("tensor-%d-5" % i, dtype=np.float32, shape=None)
    node6 = gs.Node("Relu", "ReLU-" + str(i), inputs=[tensor5], outputs=[tensor6])
    graphNodeList.append(node6)

    tensorLoop = tensor6

tensor7 = gs.Variable("tensor-6", dtype=np.float32, shape=None)
node7 = gs.Node("Conv", "Conv1", inputs=[tensorLoop, constant1x32], outputs=[tensor7])
graphNodeList.append(node7)

graph = gs.Graph(nodes=graphNodeList, inputs=[tensor0], outputs=[tensor7], opset=13)

onnx.save(gs.export_onnx(graph.cleanup().toposort()), onnxFile0)
print("Succeeded building %s!" % (onnxFile0))

# Remove pairs of Transpose or Squeeze/Unsqueeze ndoes
graph = gs.import_onnx(onnx.load(onnxFile0))

for node in graph.nodes:
    if node.op in ["Unsqueeze", "Squeeze"]:
        node.o().inputs[0] = node.inputs[0]

    if node.op == "Transpose":
        if node.o().op == "Add":
            node.o().inputs[1] = constant32r
        node.o().inputs[0] = node.inputs[0]

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), onnxFile1)
print("Succeeded building %s!" % (onnxFile1))

def run(onnxFile):
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.max_workspace_size = 22 << 30

    parser = trt.OnnxParser(network, logger)
    with open(onnxFile, "rb") as model:
        parser.parse(model.read())

    inputT0 = network.get_input(0)
    inputT0.shape = [-1, 1, 16, 16]
    profile.set_shape(inputT0.name, [1, 1, 16, 16], [8, 1, 16, 16], [8, 1, 16, 16])
    config.add_optimization_profile(profile)

    engineString = builder.build_serialized_network(network, config)
    planFile = onnxFile.split(".")[0] + ".plan"
    with open(planFile, "wb") as f:
        f.write(engineString)
    print("Succeeded building %s!" % (planFile))

    os.system("trtexec --loadEngine=%s --verbose --useCudaGraph --noDataTransfers --shapes=tensor-0:8x1x16x16" % planFile)

run(onnxFile0)
run(onnxFile1)
