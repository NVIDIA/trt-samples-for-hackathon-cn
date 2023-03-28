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

trtFile = "./model.plan"
timeCacheFile = "./model.cache"
nB, nC, nH, nW = 1, 1, 28, 28
np.random.seed(31193)
data = np.random.rand(nB, nC, nH, nW).astype(np.float32) * 2 - 1

class MyGpuAllocator(trt.IGpuAllocator):

    def __init__(self):
        print("[MyGpuAllocator::__init__]()")
        super(MyGpuAllocator, self).__init__()
        self.sizeList = []
        self.addressList = []
        self.flagList = []

    def allocate(self, size, alignment, flag):
        print("[MyGpuAllocator::allocate](%d,%d,%d)" % (size, alignment, flag))
        status, adress = cudart.cudaMalloc(size)
        if status != cudart.cudaError_t.cudaSuccess:
            print("Failed allocating size %d")
            return 0

        self.sizeList.append(size)
        self.addressList.append(adress)
        self.flagList.append(bool(flag))  # flag == True means the size is flexible (reallocate will be called), this is inconsistent with int(trt.AllocatorFlag.RESIZABLE) == 0

    def deallocate(self, adress):
        #def free(adress): # another name of this API，deprecated since TensorRT 8.0
        print("[MyGpuAllocator::deallocate](%d)" % adress)
        try:
            index = self.addressList.index(adress)
        except:
            print("Failed finding adress %d in addressList" % adress)
            return False

        status = cudart.cudaFree(adress)
        if status[0] != cudart.cudaError_t.cudaSuccess:
            print("Failed deallocating adress %d" % adress)
            return False

        del self.sizeList[index]
        del self.addressList[index]
        del self.flagList[index]
        return True

    def reallocate(self, oldAddress, alignment, newSize):
        print("[MyGpuAllocator::reallocate](%d,%d,%d)" % (oldAddress, alignment, newSize))
        try:
            index = self.addressList.index(oldAddress)
        except:
            print("Failed finding adress %d in addressList" % oldAddress)
            return 0

        if self.flagList[index] == False:
            print("Old buffer is not resizeable")
            return 0

        if newSize <= self.sizeList[index]:  # smaller than the older size
            print("New size is not larger than the old one")
            return oldAddress

        newAddress = self.allocate(newSize, alignment, self.flagList[index])
        if newAddress == 0:
            print("Failed reallocating new buffer")
            return 0

        status = cudart.cudaMemcpy(newAddress, oldAddress, self.sizeList[index], cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        if status[0] != cudart.cudaError_t.cudaSuccess:
            print("Failed copy memory from buffer %d to %d" % (oldAddress, newAddress))
            return oldAddress

        status = self.deallocate(oldAddress)
        if status == False:
            print("Failed deallocating old buffer %d" % oldAddress)
            return newAddress

        return newAddress

np.set_printoptions(precision=3, linewidth=100, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
builder.gpu_allocator = MyGpuAllocator()  # GPU Allocator for build time
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)

inputTensor = network.add_input("inputT0", trt.float32, [-1, nC, nH, nW])
profile.set_shape(inputTensor.name, [1, nC, nH, nW], [nB, nC, nH, nW], [nB * 2, nC, nH, nW])
config.add_optimization_profile(profile)

w = np.ascontiguousarray(np.random.rand(32, 1, 5, 5).astype(np.float32))
b = np.ascontiguousarray(np.random.rand(32, 1, 1).astype(np.float32))
_0 = network.add_convolution_nd(inputTensor, 32, [5, 5], trt.Weights(w), trt.Weights(b))
_0.padding_nd = [2, 2]
_1 = network.add_activation(_0.get_output(0), trt.ActivationType.RELU)
_2 = network.add_pooling_nd(_1.get_output(0), trt.PoolingType.MAX, [2, 2])
_2.stride_nd = [2, 2]

w = np.ascontiguousarray(np.random.rand(64, 32, 5, 5).astype(np.float32))
b = np.ascontiguousarray(np.random.rand(64, 1, 1).astype(np.float32))
_3 = network.add_convolution_nd(_2.get_output(0), 64, [5, 5], trt.Weights(w), trt.Weights(b))
_3.padding_nd = [2, 2]
_4 = network.add_activation(_3.get_output(0), trt.ActivationType.RELU)
_5 = network.add_pooling_nd(_4.get_output(0), trt.PoolingType.MAX, [2, 2])
_5.stride_nd = [2, 2]

_6 = network.add_shuffle(_5.get_output(0))
_6.reshape_dims = (-1, 64 * 7 * 7)

w = np.ascontiguousarray(np.random.rand(64 * 7 * 7, 1024).astype(np.float32))
b = np.ascontiguousarray(np.random.rand(1, 1024).astype(np.float32))
_7 = network.add_constant(w.shape, trt.Weights(w))
_8 = network.add_matrix_multiply(_6.get_output(0), trt.MatrixOperation.NONE, _7.get_output(0), trt.MatrixOperation.NONE)
_9 = network.add_constant(b.shape, trt.Weights(b))
_10 = network.add_elementwise(_8.get_output(0), _9.get_output(0), trt.ElementWiseOperation.SUM)
_11 = network.add_activation(_10.get_output(0), trt.ActivationType.RELU)

w = np.ascontiguousarray(np.random.rand(1024, 10).astype(np.float32))
b = np.ascontiguousarray(np.random.rand(1, 10).astype(np.float32))
_12 = network.add_constant(w.shape, trt.Weights(w))
_13 = network.add_matrix_multiply(_11.get_output(0), trt.MatrixOperation.NONE, _12.get_output(0), trt.MatrixOperation.NONE)
_14 = network.add_constant(b.shape, trt.Weights(b))
_15 = network.add_elementwise(_13.get_output(0), _14.get_output(0), trt.ElementWiseOperation.SUM)

_16 = network.add_softmax(_15.get_output(0))
_16.axes = 1 << 1

_17 = network.add_topk(_16.get_output(0), trt.TopKOperation.MAX, 1, 1 << 1)

network.mark_output(_17.get_output(1))
engineString = builder.build_serialized_network(network, config)

runtime = trt.Runtime(logger)
runtime.gpu_allocator = MyGpuAllocator()  # GPU Allocator for runtime, it can be assigned to Runtime or ExecutionContext
engine = runtime.deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()
#context.temporary_allocator = MyGpuAllocator()  # GPU Allocator for runtime

context.set_input_shape(lTensorName[0], [nB, nC, nH, nW])
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

bufferH = []
bufferH.append(np.ascontiguousarray(data.reshape(-1)))
for i in range(nInput, nIO):
    bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
bufferD = []
for i in range(nIO):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

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