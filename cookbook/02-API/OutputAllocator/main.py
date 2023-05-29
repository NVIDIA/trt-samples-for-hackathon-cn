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
import tensorrt as trt
from cuda import cudart

shape = [4, 5, 6]
data = np.zeros(shape).astype(np.float32)
data[0, 0, 1] = 1
data[0, 2, 3] = 2
data[0, 3, 4] = 3
data[1, 1, 0] = 4
data[1, 1, 1] = 5
data[1, 1, 2] = 6
data[1, 1, 3] = 7
data[1, 1, 4] = 8
data[1, 1, 5] = 9
data[2, 0, 1] = 10
data[2, 1, 1] = 11
data[2, 2, 1] = 12
data[2, 3, 1] = 13
data[2, 4, 1] = 14
np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

class MyOutputAllocator(trt.IOutputAllocator):

    def __init__(self):
        print("[MyOutputAllocator::__init__]")
        super(MyOutputAllocator, self).__init__()
        self.shape = None
        self.size = 0
        self.address = 0

    def reallocate_output(self, tensor_name, memory, size, alignment):
        print("[MyOutputAllocator::reallocate_output] TensorName=%s, Memory=%s, Size=%d, Alignment=%d" % (tensor_name, memory, size, alignment))
        if size <= self.size:  # the buffer is enough to use
            return memory
        
        if memory != 0:
            status = cudart.cudaFree(memory)
            if status != cudart.cudaError_t.cudaSuccess:
                print("Failed freeing old memory")
                return 0
        
        status, adress = cudart.cudaMalloc(size)
        if status != cudart.cudaError_t.cudaSuccess:
            print("Failed allocating size %d")
            return 0
        
        self.size = size
        self.address = adress
        return adress
        
    def notify_shape(self, tensor_name, shape):
        print("[MyOutputAllocator::notify_shape] TensorName=%s, Shape=%s" % (tensor_name, shape))
        self.shape = shape
        return

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.float32, shape)
profile.set_shape(inputT0.name, shape, shape, shape)
config.add_optimization_profile(profile)

nonZeroLayer = network.add_non_zero(inputT0)  # use a data-dependent network as example, normal network is also OK

network.mark_output(nonZeroLayer.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()

myOutputAllocator = MyOutputAllocator()
for i in range(nInput, nIO):
    context.set_output_allocator(lTensorName[i], myOutputAllocator)  # assign Output Allocator to Context, one Output Allocator for each output tensor

for i in range(nIO):
    # context.get_tensor_shape(lTensorName[1]) here returns (3,-1)
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

bufferH = []
bufferH.append(data)  # only prepare input buffer
bufferD = []
for i in range(nInput):  # prepare the input buffer
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])
for i in range(nInput, nIO):  # use nullptr for output buffer
    bufferD.append(int(0))
for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

for i in range(nIO):
    context.set_tensor_address(lTensorName[i], int(bufferD[i]))

context.execute_async_v3(0)

print("After do inference")
for i in range(nIO):
    # context.get_tensor_shape(lTensorName[1]) here returns real shape of output tensor
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

for i in range(nInput, nIO):  # get buffer from Output Allocator
    myOutputAllocator = context.get_output_allocator(lTensorName[i])
    bufferH.append(np.empty(myOutputAllocator.shape, dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    bufferD[i] = myOutputAllocator.address

for i in range(nInput, nIO):
    cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

for i in range(nIO):
    print(lTensorName[i])
    print(bufferH[i])

for b in bufferD:
    cudart.cudaFree(b)