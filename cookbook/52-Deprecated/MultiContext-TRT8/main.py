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

np.random.seed(31193)
nBatchSize, nHiddenSize = 4, 2
data = np.random.rand(nBatchSize * nHiddenSize).astype(np.float32).reshape(nBatchSize, nHiddenSize)

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30

inputT0 = network.add_input("inputT0", trt.float32, [-1, nHiddenSize])
layer = network.add_unary(inputT0, trt.UnaryOperation.NEG)
network.mark_output(layer.get_output(0))

profile.set_shape(inputT0.name, [1, nHiddenSize], [nBatchSize, nHiddenSize], [nBatchSize * 2, nHiddenSize])
config.add_optimization_profile(profile)

engine = builder.build_engine(network, config)

contextList = []
for i in range(4):
    context = engine.create_execution_context()
    context.active_optimization_profile = 0
    context.set_binding_shape(0, [i + 1, nHiddenSize])
    contextList.append(context)
    print("Context binding all? %s" % (["No", "Yes"][int(context.all_binding_shapes_specified)]))
    for i in range(engine.num_bindings):
        print(i, "Input " if engine.binding_is_input(i) else "Output", engine.get_binding_shape(i), context.get_binding_shape(i))

inputH0 = np.ascontiguousarray(data.reshape(-1))
outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))

_, inputD0 = cudart.cudaMalloc(inputH0.nbytes)
_, outputD0 = cudart.cudaMalloc(outputH0.nbytes)

for i in range(4):
    cudart.cudaMemcpy(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    contextList[i].execute_v2([int(inputD0), int(outputD0)])
    cudart.cudaMemcpy(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    print("check result:", np.all(outputH0[:i, ...] == -inputH0.reshape(outputH0.shape)[:i, ...]))

cudart.cudaFree(inputD0)
cudart.cudaFree(outputD0)
