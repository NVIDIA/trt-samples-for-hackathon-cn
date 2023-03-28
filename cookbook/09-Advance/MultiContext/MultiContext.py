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

from cuda import cudart
import numpy as np
import tensorrt as trt

nB, nC, nH, nW = 2, 3, 4, 5
data = np.arange(nB * nC * nH * nW).astype(np.float32).reshape(nB, nC, nH, nW)
np.random.seed(31193)
np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile0 = builder.create_optimization_profile()
profile1 = builder.create_optimization_profile()
config = builder.create_builder_config()

inputT0 = network.add_input("inputT0", trt.float32, [-1, -1, -1, -1])
layer = network.add_unary(inputT0, trt.UnaryOperation.NEG)
network.mark_output(layer.get_output(0))

profile0.set_shape(inputT0.name, (1, nC, nH, nW), (nB, nC, nH, nW), (nB * 2, nC, nH, nW))
profile1.set_shape(inputT0.name, (1, nC, nH, nW), (nB, nC, nH, nW), (nB * 2, nC, nH, nW))
config.add_optimization_profile(profile0)
config.add_optimization_profile(profile1)

engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
_, stream0 = cudart.cudaStreamCreate()
_, stream1 = cudart.cudaStreamCreate()
context0 = engine.create_execution_context()
context1 = engine.create_execution_context()
context0.set_optimization_profile_async(0, stream0)
context1.set_optimization_profile_async(1, stream1)
context0.set_binding_shape(0, [nB, nC, nH, nW])
context1.set_binding_shape(2, [nB, nC, nH, nW])
print("Context0 binding all? %s" % (["No", "Yes"][int(context0.all_binding_shapes_specified)]))
print("Context1 binding all? %s" % (["No", "Yes"][int(context1.all_binding_shapes_specified)]))
for i in range(engine.num_bindings):
    print(i, "Input " if engine.binding_is_input(i) else "Output", engine.get_binding_shape(i), context0.get_binding_shape(i), context1.get_binding_shape(i))

bufferH = []
bufferH.append(np.ascontiguousarray(data))
bufferH.append(np.empty(context0.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1))))
bufferH.append(np.ascontiguousarray(data))
bufferH.append(np.empty(context1.get_binding_shape(3), dtype=trt.nptype(engine.get_binding_dtype(3))))

bufferD = []
for i in range(engine.num_bindings):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

cudart.cudaMemcpyAsync(bufferD[0], bufferH[0].ctypes.data, bufferH[0].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream0)
cudart.cudaMemcpyAsync(bufferD[2], bufferH[2].ctypes.data, bufferH[2].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream1)

context0.execute_async_v2([int(bufferD[0]), int(bufferD[1]), int(0), int(0)], stream0)
context1.execute_async_v2([int(0), int(0), int(bufferD[2]), int(bufferD[3])], stream1)

cudart.cudaMemcpyAsync(bufferH[1].ctypes.data, bufferD[1], bufferH[1].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream0)
cudart.cudaMemcpyAsync(bufferH[3].ctypes.data, bufferD[3], bufferH[3].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream1)

cudart.cudaStreamSynchronize(stream0)
cudart.cudaStreamSynchronize(stream1)

print("check result:", np.all(bufferH[1] == -bufferH[0]))
print("check result:", np.all(bufferH[3] == -bufferH[2]))

cudart.cudaStreamDestroy(stream0)
cudart.cudaStreamDestroy(stream1)

for b in bufferD:
    cudart.cudaFree(b)
