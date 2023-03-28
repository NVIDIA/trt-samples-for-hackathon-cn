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
from cuda import cudart
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import os
import tensorrt as trt
from time import time_ns

nLoop = 10
nWarm = 10
nTest = 100
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

def test(engine, context, nBatchSize):
    nProfile = engine.num_optimization_profiles
    if nProfile == 1:
        bindingBias = 0
    else:
        if nBatchSize <= 4:
            bindingBias = 0
            context.set_optimization_profile_async(0, 0)
            cudart.cudaStreamSynchronize(0)
        else:
            bindingBias = 2
            context.set_optimization_profile_async(1, 0)
            cudart.cudaStreamSynchronize(0)

    context.set_binding_shape(bindingBias, [nBatchSize, 1])
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    for i in range(nInput):
        print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
    for i in range(nInput, nInput + nOutput):
        print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

    nInput = nInput // nProfile
    nOutput = nOutput // nProfile

    data = np.random.rand(nBatchSize).reshape(nBatchSize, 1).astype(np.float32)
    bufferH = []
    bufferH.append(np.ascontiguousarray(data.reshape(-1)))
    for i in range(nInput, nInput + nOutput):
        bufferH.append(np.empty(context.get_binding_shape(bindingBias + i), dtype=trt.nptype(engine.get_binding_dtype(bindingBias + i))))
    bufferD = []
    for i in range(nInput + nOutput):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    if nProfile == 1 or nBatchSize <= 4:
        bufferD = bufferD + [int(0), int(0)]
    else:
        bufferD = [int(0), int(0)] + bufferD

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_v2(bufferD)
    for i in range(nInput, nInput + nOutput):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for i in range(nWarm):
        context.execute_v2(bufferD)

    t0 = time_ns()
    for i in range(nTest):
        context.execute_v2(bufferD)
    t1 = time_ns()
    print("+---- BatchSize=%2d: %.4fms\n" % (nBatchSize, (t1 - t0) / 1e6 / nTest))

    if nProfile == 1 or nBatchSize <= 4:
        bufferD = bufferD[:2]
    else:
        bufferD = bufferD[-2:]

    for b in bufferD:
        cudart.cudaFree(b)

def run(nProfile):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()

    parser = trt.OnnxParser(network, logger)
    with open(onnxFile, "rb") as model:
        parser.parse(model.read())

    if nProfile == 1:
        profile = builder.create_optimization_profile()
        inputT0 = network.get_input(0)
        inputT0.shape = [-1, 1]
        profile.set_shape(inputT0.name, [1, 1], [510, 1], [512, 1])
        config.add_optimization_profile(profile)
    else:
        profile0 = builder.create_optimization_profile()
        inputT0 = network.get_input(0)
        inputT0.shape = [-1, 1]
        profile0.set_shape(inputT0.name, (1, 1), (4, 1), (4, 1))
        config.add_optimization_profile(profile0)

        profile1 = builder.create_optimization_profile()
        inputT0 = network.get_input(0)
        inputT0.shape = [-1, 1]
        profile1.set_shape(inputT0.name, (510, 1), (510, 1), (512, 1))
        config.add_optimization_profile(profile1)

    engineString = builder.build_serialized_network(network, config)
    planFile = onnxFile.split(".")[0] + "-%d.plan" % nProfile
    with open(planFile, "wb") as f:
        f.write(engineString)

    print("Succeeded building %s!" % (planFile))

    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    context = engine.create_execution_context()

    # MultiOptimizationProfile is not supported by TensorRT-8.5, we use script to test
    test(engine, context, 1)
    test(engine, context, 4)
    test(engine, context, 510)
    test(engine, context, 512)

run(1)
run(2)
