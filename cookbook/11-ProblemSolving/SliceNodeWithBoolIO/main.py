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

onnxFile1 = "model-01.onnx"
onnxFile2 = "model-02.onnx"

# Create a ONNX graph with Onnx Graphsurgeon -----------------------------------
a0 = gs.Constant("a0", np.ascontiguousarray(np.array([0], dtype=np.int64)))
a1 = gs.Constant("a1", np.ascontiguousarray(np.array([1], dtype=np.int64)))
a2 = gs.Constant("a2", np.ascontiguousarray(np.array([2], dtype=np.int64)))
am1 = gs.Constant("am1", np.ascontiguousarray(np.array([-1], dtype=np.int64)))
am2 = gs.Constant("am2", np.ascontiguousarray(np.array([-2], dtype=np.int64)))

nodeList = []

tensor0 = gs.Variable("tensor0", np.int32, ["B", "T"])
tensor1 = gs.Variable("tensor1", np.int32, ["B", "T"])

tensor2 = gs.Variable("tensor2", bool, ["B", "T"])
node0 = gs.Node("GreaterOrEqual", "GreaterOrEqual_27", inputs=[tensor0, tensor1], outputs=[tensor2])
nodeList.append(node0)

tensor3 = gs.Variable("tensor3", bool, ["B", 1, "T"])
node1 = gs.Node("Unsqueeze", "Unsqueeze_29", inputs=[tensor2], outputs=[tensor3], attrs=OrderedDict([("axes", [1])]))
nodeList.append(node1)

tensor4 = gs.Variable("tensor4", bool, ["B", 1, "T"])
node2 = gs.Node("Not", "Not_30", inputs=[tensor3], outputs=[tensor4])
nodeList.append(node2)

tensor5 = gs.Variable("tensor5", bool, ["B", 1, "T2"])
node3 = gs.Node("Slice", "Slice_79", inputs=[tensor4, a0, am2, a2, a2], outputs=[tensor5])
nodeList.append(node3)

tensor6 = gs.Variable("tensor6", bool, ["B", 1, "T3"])
node4 = gs.Node("Slice", "Slice_84", inputs=[tensor5, a0, am2, a2, a2], outputs=[tensor6])
nodeList.append(node4)

graph = gs.Graph(nodes=nodeList, inputs=[tensor0, tensor1], outputs=[tensor6])

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), onnxFile1)

# Edit the network with Onnx Graphsurgeon --------------------------------------
graph = gs.import_onnx(onnx.load(onnxFile1))

for node in graph.nodes:
    if node.op == "Slice" and node.name == "Slice_79":

        castV0 = gs.Variable("CastV-0", np.dtype(np.int32), None)
        castN0 = gs.Node("Cast", "CastN-0", inputs=[node.inputs[0]], outputs=[castV0], attrs=OrderedDict([("to", onnx.TensorProto.INT32)]))
        graph.nodes.append(castN0)

        node.inputs[0] = castV0
        nextSliceNode = node.o()

        castV1 = gs.Variable("CastV-1", bool, None)
        castN1 = gs.Node("Cast", "CastN-1", inputs=[nextSliceNode.outputs[0]], outputs=[castV1], attrs=OrderedDict([("to", onnx.TensorProto.BOOL)]))
        graph.nodes.append(castN1)

        for i in range(len(graph.outputs)):
            if graph.outputs[i] == nextSliceNode.outputs[0]:
                graph.outputs[i] = castV1

        break

graph.cleanup()
onnx.save(gs.export_onnx(graph), onnxFile2)

# parse ONNX into TensorRT by trtexec ------------------------------------------
def parseOnnxToTRT(logger, onnxFile):
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()

    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding ONNX file!")
        return
    print("Succeeded finding ONNX file!")
    with open(onnxFile, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing ONNX file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return
        print("Succeeded parsing ONNX file!")

    inputT0 = network.get_input(0)
    inputT0.shape = [-1, -1]
    profile.set_shape(inputT0.name, [1, 1], [4, 32], [16, 64])
    inputT1 = network.get_input(1)
    inputT1.shape = [-1, -1]
    profile.set_shape(inputT1.name, [1, 1], [4, 32], [16, 64])
    config.add_optimization_profile(profile)

    engineString = builder.build_serialized_network(network, config)
    print("%s building .plan from %s" % ("Failed" if engineString is None else "Succeeded", onnxFile))

logger = trt.Logger(trt.Logger.ERROR)

parseOnnxToTRT(logger, onnxFile1)
parseOnnxToTRT(logger, onnxFile2)
print("All test finished!")
