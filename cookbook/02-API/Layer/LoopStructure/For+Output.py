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

import numpy as np
from cuda import cudart
import tensorrt as trt

nB, nC, nH, nW = 1, 3, 4, 5
t = np.array([6], dtype=np.int32)  # number of iterations
data = np.ones([nB, nC, nH, nW], dtype=np.float32)

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.float32, (nB, nC, nH, nW))
#------------------------------------------------------------------------------- Network
loop = network.add_loop()  # Add Loop structure

limit = network.add_constant((), np.array([t], dtype=np.int32))  # count of iteration, here as buildtime constant
loop.add_trip_limit(limit.get_output(0), trt.TripLimit.COUNT)  # use Count-Loop (similar to for-loop）

rLayer = loop.add_recurrence(inputT0)  # entrance of Loop
_H0 = network.add_elementwise(rLayer.get_output(0), rLayer.get_output(0), trt.ElementWiseOperation.SUM)  # body of Loop
#rLayer.set_input(0,inputT0)  # the zeroth input tensor of rLayer is inputT0 itself
rLayer.set_input(1, _H0.get_output(0))  # the first input tensor of rLayer is the output tensor of the body of the Loop

loopOutput0 = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # the first kind of output, only Keep the final output of the Loop, parameter index is ignored
loopOutput1 = loop.add_loop_output(_H0.get_output(0), trt.LoopOutput.CONCATENATE, 0)  # the second kind of output, keep all output of the Loop
# keep result of iteration 1 to t if passing _H0 to loop.add_loop_output, while result of ieration 0 to t-1 if passing rLayer to loop.add_loop_output
loopOutput1.set_input(1, limit.get_output(0))  # set the count of iteration kept in the output, if the value of given here v <= t, the first v ierations are kept, or 0 padding is used for the part of v > t
#------------------------------------------------------------------------------- Network
network.mark_output(loopOutput0.get_output(0))
network.mark_output(loopOutput1.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()
context.set_input_shape(lTensorName[0], [nB, nC, nH, nW])
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

bufferH = []
for i in range(nIO):
    bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
bufferD = []
for i in range(nIO):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

bufferH[0] = data

for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

for i in range(nIO):
    context.set_tensor_address(lTensorName[i], int(bufferD[i]))

context.execute_async_v3(0)

for i in range(nInput, nIO):
    cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

for i in range(nIO):
    print(lTensorName[i])
    print(bufferH[i])

for b in bufferD:
    cudart.cudaFree(b)