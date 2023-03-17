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
onnxFile = "model.onnx"
np.random.seed(31193)

# Create a ONNX graph with Onnx Graphsurgeon -----------------------------------
tensor0 = gs.Variable("tensor0", np.float32, ["B", 1])

constant1x256 = gs.Constant("constant1x256", np.ascontiguousarray(np.random.rand(1, 256).reshape(1, 256).astype(np.float32) * 2 - 1))
constant256x2048 = gs.Constant("constant256x2048", np.ascontiguousarray(np.random.rand(256, 2048).reshape(256, 2048).astype(np.float32) * 2 - 1))
constant2048 = gs.Constant("constant2048", np.ascontiguousarray(np.random.rand(2048).astype(np.float32) * 2 - 1))
constant2048x256 = gs.Constant("constant2048x256", np.ascontiguousarray(np.random.rand(2048, 256).reshape(2048, 256).astype(np.float32) * 2 - 1))
constant256 = gs.Constant("constant256", np.ascontiguousarray(np.random.rand(256).astype(np.float32) * 2 - 1))
constantM1 = gs.Constant("constantM1", np.ascontiguousarray(np.array([-1], dtype=np.int64)))

graphNodeList = []

tensor1 = gs.Variable("tensor1", np.float32, None)
node1 = gs.Node("MatMul", "MMU1", inputs=[tensor0, constant1x256], outputs=[tensor1])
graphNodeList.append(node1)

tensorLoop = tensor1
for i in range(nLoop):
    tensor2 = gs.Variable("tensor%d-1" % i, np.float32, None)
    node2 = gs.Node("MatMul", "MMU-" + str(i), inputs=[tensorLoop, constant256x2048], outputs=[tensor2])
    graphNodeList.append(node2)

    tensor3 = gs.Variable("tensor%d-2" % i, dtype=np.float32, shape=None)
    node3 = gs.Node("Add", "AddU-" + str(i), inputs=[tensor2, constant2048], outputs=[tensor3])
    graphNodeList.append(node3)

    tensor4 = gs.Variable("tensor%d-3" % i, dtype=np.float32, shape=None)
    node4 = gs.Node("Relu", "ReLUU-" + str(i), inputs=[tensor3], outputs=[tensor4])
    graphNodeList.append(node4)

    tensor5 = gs.Variable("tensor%d-4" % i, dtype=np.float32, shape=None)
    node5 = gs.Node("MatMul", "MMD-" + str(i), inputs=[tensor4, constant2048x256], outputs=[tensor5])
    graphNodeList.append(node5)

    tensor6 = gs.Variable("tensor%d-5" % i, dtype=np.float32, shape=None)
    node6 = gs.Node("Add", "AddD-" + str(i), inputs=[tensor5, constant256], outputs=[tensor6])
    graphNodeList.append(node6)

    tensor7 = gs.Variable("tensor%d-6" % i, dtype=np.float32, shape=None)
    node7 = gs.Node("Relu", "ReLUD-" + str(i), inputs=[tensor6], outputs=[tensor7])
    graphNodeList.append(node7)

    tensorLoop = tensor7

tensor8 = gs.Variable("tensor8", dtype=np.float32, shape=None)
node8 = gs.Node("ReduceSum", "Reduce", inputs=[tensorLoop, constantM1], outputs=[tensor8], attrs=OrderedDict([("keepdims", 0)]))
graphNodeList.append(node8)

graph = gs.Graph(nodes=graphNodeList, inputs=[tensor0], outputs=[tensor8], opset=13)

onnx.save(gs.export_onnx(graph.cleanup().toposort()), onnxFile)
print("Succeeded building %s!" % (onnxFile))

def run(nBS):
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.max_workspace_size = 22 << 30

    parser = trt.OnnxParser(network, logger)
    with open(onnxFile, "rb") as model:
        parser.parse(model.read())

    inputT0 = network.get_input(0)
    inputT0.shape = [nBS, 1]

    engineString = builder.build_serialized_network(network, config)
    planFile = onnxFile.split(".")[0] + ".plan"
    with open(planFile, "wb") as f:
        f.write(engineString)

    print("Succeeded building %s!" % planFile)

    os.system("trtexec --loadEngine=%s --verbose --useCudaGraph --noDataTransfers" % planFile)

run(1)
run(2)
run(4)
run(8)
run(16)
run(32)
run(64)
run(128)
run(256)
run(512)
run(1024)
