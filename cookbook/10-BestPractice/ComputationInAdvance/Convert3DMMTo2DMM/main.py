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

onnxFile3D = "model-3D.onnx"
onnxFile2D = "model-2D.onnx"
nLoop = 10
nBS = 32
nSL = 256
np.random.seed(31193)

# ONNX network with 3D matrix multipication ------------------------------------
tensor0 = gs.Variable("tensor-0", np.float32, ["B", "T", 1])
constant1x256 = gs.Constant("constant1x256", np.ascontiguousarray(np.random.rand(1, 256).reshape(1, 256).astype(np.float32) * 2 - 1))
constant256 = gs.Constant("constant256", np.ascontiguousarray(np.random.rand(256).astype(np.float32) * 2 - 1))
constant256x2048 = gs.Constant("constant256x2048", np.ascontiguousarray(np.random.rand(256, 2048).reshape(256, 2048).astype(np.float32) * 2 - 1))
constant2048 = gs.Constant("constant2048", np.ascontiguousarray(np.random.rand(2048).astype(np.float32) * 2 - 1))
constant2048x256 = gs.Constant("constant2048x256", np.ascontiguousarray(np.random.rand(2048, 256).reshape(2048, 256).astype(np.float32) * 2 - 1))
constantM1 = gs.Constant("constantM1", np.ascontiguousarray(np.array([-1], dtype=np.int64)))

graphNodeList = []

tensor1 = gs.Variable("tensor-1", np.float32, None)
node1 = gs.Node("MatMul", "MMU0", inputs=[tensor0, constant1x256], outputs=[tensor1])
graphNodeList.append(node1)

tensorLoop = tensor1
for i in range(nLoop):
    tensor2 = gs.Variable("tensor-%d-1" % i, np.float32, None)
    node2 = gs.Node("MatMul", "MMU-" + str(i), inputs=[tensorLoop, constant256x2048], outputs=[tensor2])
    graphNodeList.append(node2)

    tensor3 = gs.Variable("tensor-%d-2" % i, dtype=np.float32, shape=None)
    node3 = gs.Node("Add", "AddU-" + str(i), inputs=[tensor2, constant2048], outputs=[tensor3])
    graphNodeList.append(node3)

    tensor4 = gs.Variable("tensor-%d-3" % i, dtype=np.float32, shape=None)
    node4 = gs.Node("Relu", "ReLUU-" + str(i), inputs=[tensor3], outputs=[tensor4])
    graphNodeList.append(node4)

    tensor5 = gs.Variable("tensor-%d-4" % i, dtype=np.float32, shape=None)
    node5 = gs.Node("MatMul", "MMD-" + str(i), inputs=[tensor4, constant2048x256], outputs=[tensor5])
    graphNodeList.append(node5)

    tensor6 = gs.Variable("tensor-%d-5" % i, dtype=np.float32, shape=None)
    node6 = gs.Node("Add", "AddD-" + str(i), inputs=[tensor5, constant256], outputs=[tensor6])
    graphNodeList.append(node6)

    tensor7 = gs.Variable("tensor-%d-6" % i, dtype=np.float32, shape=None)
    node7 = gs.Node("Relu", "ReLUD-" + str(i), inputs=[tensor6], outputs=[tensor7])
    graphNodeList.append(node7)

    tensorLoop = tensor7

tensor8 = gs.Variable("tensor-8", dtype=np.float32, shape=None)
node8 = gs.Node("ReduceSum", "Reduce", inputs=[tensorLoop, constantM1], outputs=[tensor8], attrs=OrderedDict([("keepdims", 0)]))
graphNodeList.append(node8)

graph = gs.Graph(nodes=graphNodeList, inputs=[tensor0], outputs=[tensor8], opset=13)

onnx.save(gs.export_onnx(graph.cleanup().toposort()), onnxFile3D)
print("Succeeded building %s!" % (onnxFile3D))

# Add two Reshape nodes to convert the 3D matrix multipication into 2D matrix multipication
graph = gs.import_onnx(onnx.load(onnxFile3D))

constant0 = gs.Constant("constant0", np.ascontiguousarray(np.array([0], dtype=np.int64)))
constant1 = gs.Constant("constant1", np.ascontiguousarray(np.array([1], dtype=np.int64)))
constantS0 = gs.Constant("constantS0", np.array(0, dtype=np.int64)).to_variable(np.dtype(np.int64), []).to_constant(np.array(0, dtype=np.dtype(np.int64)))
constantS1 = gs.Constant("constantS1", np.array(1, dtype=np.int64)).to_variable(np.dtype(np.int64), []).to_constant(np.array(1, dtype=np.dtype(np.int64)))

shapeV = gs.Variable("myShapeV", np.dtype(np.int64), [3])
shapeN = gs.Node("Shape", "myShapeN", inputs=[graph.inputs[0]], outputs=[shapeV])
graph.nodes.append(shapeN)

# shape = [], value = ["B"]
bTensorScalar = gs.Variable("bTensorScalar", np.dtype(np.int64), [])
gatherN = gs.Node("Gather", "myGatherN0", inputs=[shapeV, constantS0], outputs=[bTensorScalar], attrs=OrderedDict([("axis", 0)]))
graph.nodes.append(gatherN)

# shape = [1,], value = ["B"]
bTensor = gs.Variable("bTensor", np.dtype(np.int64), [1])
unsqueezeN = gs.Node("Unsqueeze", "myUnsqueezeN0", inputs=[bTensorScalar, constant0], outputs=[bTensor])
graph.nodes.append(unsqueezeN)

# shape = [], value = ["T"]
tTensorScalar = gs.Variable("tTensorScalar", np.dtype(np.int64), [])
gatherN = gs.Node("Gather", "myGatherN1", inputs=[shapeV, constantS1], outputs=[tTensorScalar], attrs=OrderedDict([("axis", 0)]))
graph.nodes.append(gatherN)

# shape = [1,], value = ["T"]
tTensor = gs.Variable("tTensor", np.dtype(np.int64), [1])
unsqueezeN = gs.Node("Unsqueeze", "myUnsqueezeN1", inputs=[tTensorScalar, constant0], outputs=[tTensor])
graph.nodes.append(unsqueezeN)

# shape = [1,], value = ["B"*"T"]
bTTensor = gs.Variable("bTTensor", np.dtype(np.int64), [1])
mulN = gs.Node("Mul", "myMulN", inputs=[bTensor, tTensor], outputs=[bTTensor])
graph.nodes.append(mulN)

# shape = [2,], value = ["B"*"T",1]
bTComma1Tensor = gs.Variable("bTComma1Tensor", np.dtype(np.int64), [2])
concatN = gs.Node("Concat", "myConcatN", inputs=[bTTensor, constant1], outputs=[bTComma1Tensor], attrs=OrderedDict([("axis", 0)]))
graph.nodes.append(concatN)

for node in graph.nodes:
    if node.name == "MMU0":
        reshapeV = gs.Variable("reshapeV-input", np.dtype(np.float32), ['B*T', 1])
        reshapeN = gs.Node("Reshape", "myReshapeN-input", inputs=[node.inputs[0], bTComma1Tensor], outputs=[reshapeV])
        graph.nodes.append(reshapeN)
        node.inputs[0] = reshapeV

    if node.name == "Reduce":
        reshapeV = gs.Variable("reshapeV-output", np.dtype(np.float32), ["B", "T", 1])
        reshapeN = gs.Node("Reshape", "myReshapeN-output", inputs=[node.outputs[0], shapeV], outputs=[reshapeV])
        graph.nodes.append(reshapeN)
        graph.outputs = [reshapeV]

onnx.save(gs.export_onnx(graph.cleanup().toposort()), onnxFile2D)
print("Succeeded building %s!" % (onnxFile2D))

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
    inputT0.shape = [-1, -1, 1]
    profile.set_shape(inputT0.name, [1, 1, 1], [nBS, nSL, 1], [nBS, nSL, 1])
    config.add_optimization_profile(profile)

    engineString = builder.build_serialized_network(network, config)
    planFile = onnxFile.split(".")[0] + ".plan"
    with open(planFile, "wb") as f:
        f.write(engineString)

    print("Succeeded building %s!" % (planFile))

    os.system("trtexec --loadEngine=%s --verbose --useCudaGraph --noDataTransfers --shapes=tensor-0:%dx%dx1" % (planFile, nBS, nSL))

run(onnxFile3D)
run(onnxFile2D)
