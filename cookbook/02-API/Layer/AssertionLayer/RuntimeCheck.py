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

shape = [1, 3, 4, 5]
data0 = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
data1 = np.arange(np.prod(shape[:2]), dtype=np.float32).reshape(shape[:2])
data2 = np.arange(shape[0] * (shape[1] + 1), dtype=np.float32).reshape(shape[0], shape[1] + 1)

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.float32, [-1, -1] + shape[2:])
profile.set_shape(inputT0.name, [1, 1] + shape[2:], shape, [2, 6] + shape[2:])
inputT1 = network.add_input("inputT1", trt.float32, (-1, -1))
profile.set_shape(inputT1.name, [1, 1], shape[:2], [2, 6])
config.add_optimization_profile(profile)
#------------------------------------------------------------------------------- Network
_H1 = network.add_shape(inputT0)
_H2 = network.add_slice(_H1.get_output(0), [1], [1], [1])
_H3 = network.add_shape(inputT1)
_H4 = network.add_slice(_H3.get_output(0), [1], [1], [1])
_H5 = network.add_elementwise(_H2.get_output(0), _H4.get_output(0), trt.ElementWiseOperation.EQUAL)  # check condition inputT0.shape[1] == inputT1.shape[1]

_H6 = network.add_identity(_H5.get_output(0))
_H6.get_output(0).dtype = trt.bool
_HA = network.add_assertion(_H6.get_output(0), "inputT0.shape[1] != inputT1.shape[1]")

_H7 = network.add_identity(_H5.get_output(0))
_H7.get_output(0).dtype = trt.int32
#------------------------------------------------------------------------------- Network
network.mark_output(_H7.get_output(0))
engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()
context.set_input_shape(lTensorName[0], data0.shape)

context.set_input_shape(lTensorName[1], data1.shape)  # inputT0[1,3,4,5] <-> inputT1[1,3], no error with this shape
#context.set_input_shape(lTensorName[1], data2.shape)  # inputT0[1,3,4,5] <-> inputT1[1,4], error with this shape

for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])
