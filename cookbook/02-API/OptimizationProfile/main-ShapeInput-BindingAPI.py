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

# For TensorRT <8.5 with deprecated Binding API

import numpy as np
import tensorrt as trt
from cuda import cudart

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()

inputT0 = network.add_input("inputT0", trt.float32, [3, 4, 5])
inputT1 = network.add_input("inputT1", trt.int32, [3])
profile.set_shape_input(inputT1.name, [1, 1, 1], [3, 4, 5], [5, 5, 5])
config.add_optimization_profile(profile)

shuffleLayer = network.add_shuffle(inputT0)
shuffleLayer.set_input(1, inputT1)

network.mark_output(shuffleLayer.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_bindings
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])

context = engine.create_execution_context()

def run(shape):
    context.set_shape_input(1, shape)  # set shape input tensor using CPU buffer
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), ("ShapeTensor    " if engine.is_shape_binding(i) else "ExecutionTensor"), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

    bufferH = []
    bufferH.append(np.ascontiguousarray(np.arange(np.prod(shape), dtype=np.float32).reshape(shape)))
    bufferH.append([])  # placeholder for input shape tensor, we need not to pass input shape tensor to GPU
    # we can also use a dummy input shape tenor "bufferH.append(np.ascontiguousarray([0],dtype=np.int32))" here to avoid 3 if-condition statments "if engine.is_shape_binding(i)" below
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))
    bufferD = []
    for i in range(nIO):
        if engine.is_shape_binding(i):
            bufferD.append(int(0))
        else:
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        if engine.is_shape_binding(i):
            continue
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_async_v2(bufferD, 0)

    for i in range(nInput, nIO):
        if engine.is_shape_binding(i):
            continue
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for i in range(nIO):
        print(i)
        print(bufferH[i])

    for b in bufferD:
        cudart.cudaFree(b)

    return

# do inference with a shape
run([3, 4, 5])

# do inference with another shape
run([5, 4, 3])
