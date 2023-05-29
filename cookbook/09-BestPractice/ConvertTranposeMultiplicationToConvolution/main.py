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

onnxFile0 = "model-0.onnx-backup"
onnxFile1 = "model-1.onnx"

if True:  # model bindind parameters, not change
    t0 = 19
    t1 = 256
    t2 = 256

nBS = 16
nSL = 64
"""
# extract subgraph from wenet encoder, should not be used in this example, TODO: rewrite this part by ONNX
onnxFileS = "./encoder.onnx"

graph = gs.import_onnx(onnx.load(onnxFileS))

graph.outputs = []
for node in graph.nodes:
    
    if node.op == "Relu" and node.name == "Relu_38":
        node.outputs[0].name = "inputT0"
        node.outputs[0].shape= ["B",t1,"t4",t0]
        graph.inputs = [node.outputs[0]]
    if node.op == "Add" and node.name == "Add_62":
        graph.outputs = [node.outputs[0]]
graph.cleanup()
onnx.save(gs.export_onnx(graph), onnxFile0)
"""

graph = gs.import_onnx(onnx.load(onnxFile0))

for node in graph.nodes:
    if node.op == "MatMul" and node.name == "MatMul_61":
        convKernel = node.inputs[1].values.transpose(1, 0).reshape(256, t1, 1, t0).astype(np.float32)
        convKernelV = gs.Constant("ConvKernelV", np.ascontiguousarray(convKernel))
        continue

    if node.op == "Add" and node.name == "Add_62":
        convBias = node.inputs[0].values
        convBiasV = gs.Constant("ConvBiasV", np.ascontiguousarray(convBias))
        continue

convV = gs.Variable("ConvV", np.dtype(np.float32), ["B", t1, "t4", 1])
convN = gs.Node("Conv", "ConvN", inputs=[graph.inputs[0], convKernelV, convBiasV], outputs=[convV])
convN.attrs = OrderedDict([
    ("dilations", [1, 1]),
    ("kernel_shape", [1, t0]),
    ("pads", [0, 0, 0, 0]),
    ("strides", [1, 1]),
])
graph.nodes.append(convN)

constant3 = gs.Constant("constant3", np.ascontiguousarray(np.array([3], dtype=np.int64)))

squeezeV = gs.Variable("SqueezeV", np.dtype(np.float32), ["B", t2, "t4"])
squeezeN = gs.Node("Squeeze", "SqueezeN", inputs=[convV, constant3], outputs=[squeezeV])
graph.nodes.append(squeezeN)

transposeV = gs.Variable("TransposeV", np.dtype(np.float32), ["B", "t4", t2])
transposeN = gs.Node("Transpose", "TransposeN", inputs=[squeezeV], outputs=[transposeV], attrs=OrderedDict([("perm", [0, 2, 1])]))
graph.nodes.append(transposeN)

graph.outputs = [transposeV]

graph.cleanup()
onnx.save(gs.export_onnx(graph), onnxFile1)

def run(onnxFile):
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()

    parser = trt.OnnxParser(network, logger)
    with open(onnxFile, "rb") as model:
        parser.parse(model.read())

    inputT0 = network.get_input(0)
    inputT0.shape = [-1, t1, -1, t0]
    profile.set_shape(inputT0.name, [1, t1, 1, t0], [nBS, t1, nSL, t0], [nBS, t1, nSL, t0])
    config.add_optimization_profile(profile)

    engineString = builder.build_serialized_network(network, config)
    planFile = onnxFile.split(".")[0] + ".plan"
    with open(planFile, "wb") as f:
        f.write(engineString)

    print("Succeeded building %s!" % (planFile))

    os.system("trtexec --loadEngine=%s --verbose --useCudaGraph --noDataTransfers --shapes=inputTensor:%dx%dx%dx%d" % (planFile, nBS, t1, nSL, t0))

run(onnxFile0)
run(onnxFile1)
