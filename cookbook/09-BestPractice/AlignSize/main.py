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
np.random.seed(31193)

def run(nM, nK, nN, bUseScriptToBuild):
    tensor0 = gs.Variable("tensor0", np.float32, [nM, 1])

    constant1xK = gs.Constant("constant1xK", np.ascontiguousarray(np.random.rand(1, nK).reshape(1, nK).astype(np.float32) * 2 - 1))
    constantKxN = gs.Constant("constantKxN", np.ascontiguousarray(np.random.rand(nK, nN).reshape(nK, nN).astype(np.float32) * 2 - 1))
    constantN = gs.Constant("constantN", np.ascontiguousarray(np.random.rand(nN).astype(np.float32) * 2 - 1))
    constantNxK = gs.Constant("constantNxK", np.ascontiguousarray(np.random.rand(nN, nK).reshape(nN, nK).astype(np.float32) * 2 - 1))
    constantK = gs.Constant("constantK", np.ascontiguousarray(np.random.rand(nK).astype(np.float32) * 2 - 1))
    constantM1 = gs.Constant("constantM1", np.ascontiguousarray(np.array([-1], dtype=np.int64)))

    graphNodeList = []

    tensor1 = gs.Variable("tensor1", np.float32, None)
    node1 = gs.Node("MatMul", "MMU1", inputs=[tensor0, constant1xK], outputs=[tensor1])
    graphNodeList.append(node1)

    tensorLoop = tensor1
    for i in range(nLoop):
        tensor2 = gs.Variable("tensor%d-1" % i, np.float32, None)
        node2 = gs.Node("MatMul", "MMU-" + str(i), inputs=[tensorLoop, constantKxN], outputs=[tensor2])
        graphNodeList.append(node2)

        tensor3 = gs.Variable("tensor%d-2" % i, dtype=np.float32, shape=None)
        node3 = gs.Node("Add", "AddU-" + str(i), inputs=[tensor2, constantN], outputs=[tensor3])
        graphNodeList.append(node3)

        tensor4 = gs.Variable("tensor%d-3" % i, dtype=np.float32, shape=None)
        node4 = gs.Node("Relu", "ReLUU-" + str(i), inputs=[tensor3], outputs=[tensor4])
        graphNodeList.append(node4)

        tensor5 = gs.Variable("tensor%d-4" % i, dtype=np.float32, shape=None)
        node5 = gs.Node("MatMul", "MMD-" + str(i), inputs=[tensor4, constantNxK], outputs=[tensor5])
        graphNodeList.append(node5)

        tensor6 = gs.Variable("tensor%d-5" % i, dtype=np.float32, shape=None)
        node6 = gs.Node("Add", "AddD-" + str(i), inputs=[tensor5, constantK], outputs=[tensor6])
        graphNodeList.append(node6)

        tensor7 = gs.Variable("tensor%d-6" % i, dtype=np.float32, shape=None)
        node7 = gs.Node("Relu", "ReLUD-" + str(i), inputs=[tensor6], outputs=[tensor7])
        graphNodeList.append(node7)

        tensorLoop = tensor7

    tensor8 = gs.Variable("tensor8", dtype=np.float32, shape=None)
    node8 = gs.Node("ReduceSum", "Reduce", inputs=[tensorLoop, constantM1], outputs=[tensor8], attrs=OrderedDict([("keepdims", 0)]))
    graphNodeList.append(node8)

    graph = gs.Graph(nodes=graphNodeList, inputs=[tensor0], outputs=[tensor8], opset=13)

    onnxFile = "model-%d-%d-%d.onnx" % (nM, nK, nN)
    onnx.save(gs.export_onnx(graph.cleanup().toposort()), onnxFile)
    print("Succeeded building %s!" % (onnxFile))

    if bUseScriptToBuild:
        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        config.max_workspace_size = 22 << 30

        parser = trt.OnnxParser(network, logger)
        with open(onnxFile, "rb") as model:
            parser.parse(model.read())

        engineString = builder.build_serialized_network(network, config)
        planFile = onnxFile.split(".")[0] + ".plan"
        with open(planFile, "wb") as f:
            f.write(engineString)

        print("Succeeded building %s!" % (planFile))

        os.system("trtexec --loadEngine=%s --useCudaGraph --noDataTransfers --fp16" % planFile)
    else:
        os.system("trtexec --onnx=%s --useCudaGraph --noDataTransfers --fp16" % onnxFile)

run(32, 256, 2048, True)
run(31, 256, 2048, True)  # nM -> nM-1
run(32, 255, 2048, True)  # nK -> nK-1
run(32, 256, 2047, True)  # nN -> nN-1

run(32, 256, 2048, False)
run(31, 256, 2048, False)  # nM -> nM-1
run(32, 255, 2048, False)  # nK -> nK-1
run(32, 256, 2047, False)  # nN -> nN-1
