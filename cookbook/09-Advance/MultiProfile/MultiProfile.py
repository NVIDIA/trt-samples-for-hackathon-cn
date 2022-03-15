#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

np.random.seed(97)
nIn, cIn, hIn, wIn = 8, 3, 224, 224
data = np.random.rand(nIn * cIn * hIn * wIn).astype(np.float32).reshape(nIn, cIn, hIn, wIn)

np.set_printoptions(precision=8, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile0 = builder.create_optimization_profile()
profile1 = builder.create_optimization_profile()
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30

inputT0 = network.add_input('inputT0', trt.DataType.FLOAT, [-1, cIn, hIn, wIn])
layer = network.add_unary(inputT0, trt.UnaryOperation.NEG)
network.mark_output(layer.get_output(0))

profile0.set_shape(inputT0.name, (1, cIn, hIn, wIn), (nIn, cIn, hIn, wIn), (nIn * 2, cIn, hIn, wIn))
profile1.set_shape(inputT0.name, (1, cIn, hIn, wIn), (nIn, cIn, hIn, wIn), (nIn * 2, cIn, hIn, wIn))
config.add_optimization_profile(profile0)
config.add_optimization_profile(profile1)

engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
_, stream = cudart.cudaStreamCreate()
context = engine.create_execution_context()

# 使用 Profile 0
print("Use Profile 0")
context.set_optimization_profile_async(0, stream)
cudart.cudaStreamSynchronize(stream)
#context.active_optimization_profile = 0  # 与上面两行等价的选择 profile 的方法，不需要用 stream，但是将被废弃
context.set_binding_shape(0, [nIn, cIn, hIn, wIn])
print("Context binding all? %s" % (["No", "Yes"][int(context.all_binding_shapes_specified)]))
for i in range(engine.num_bindings):
    print(i, "Input " if engine.binding_is_input(i) else "Output", engine.get_binding_shape(i), context.get_binding_shape(i))

inputH0 = np.ascontiguousarray(data.reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
_, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
_, outputD0 = cudart.cudaMalloc(outputH0.nbytes)

cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
context.execute_v2([int(inputD0), int(outputD0), int(0), int(0)])
cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
print("check result:", np.all(outputH0 == -inputH0.reshape(nIn, cIn, hIn, wIn)))

# 使用 Profile 1
print("Use Profile 1")
context.set_optimization_profile_async(1, stream)
cudart.cudaStreamSynchronize(stream)
#context.active_optimization_profile = 1  # 与上面两行等价的选择 profile 的方法，不需要用 stream，但是将被废弃
context.set_binding_shape(2, [nIn, cIn, hIn, wIn])
print("Context binding all? %s" % (["No", "Yes"][int(context.all_binding_shapes_specified)]))
for i in range(engine.num_bindings):
    print(i, "Input " if engine.binding_is_input(i) else "Output", engine.get_binding_shape(i), context.get_binding_shape(i))

inputH1 = np.ascontiguousarray(data.reshape(-1))
outputH1 = np.empty(context.get_binding_shape(2), dtype=trt.nptype(engine.get_binding_dtype(2)))
_, inputD1 = cudart.cudaMalloc(inputH1.nbytes)
_, outputD1 = cudart.cudaMalloc(outputH1.nbytes)

cudart.cudaMemcpy(inputD1, inputH1.ctypes.data, inputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
context.execute_v2([int(0), int(0), int(inputD1), int(outputD1)])
cudart.cudaMemcpy(outputH1.ctypes.data, outputD1, outputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
print("check result:", np.all(outputH0 == -inputH0.reshape(nIn, cIn, hIn, wIn)))

cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)
cudart.cudaFree(inputD1)
cudart.cudaFree(outputD1)