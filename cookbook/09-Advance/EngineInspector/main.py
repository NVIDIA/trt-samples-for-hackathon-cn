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

import os

import numpy as np
import tensorrt as trt

trtFile = "./model.plan"
shape = [1, 1, 28, 28]

os.system("rm -rf ./*.plan")

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED  # use profiling_verbosity to get more information

inputTensor = network.add_input("inputT0", trt.float32, [-1] + shape[1:])
profile.set_shape(inputTensor.name, [1] + shape[1:], [2] + shape[1:], [4] + shape[1:])
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
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

inspector = engine.create_engine_inspector()
print("inspector.execution_context=", inspector.execution_context)
print("inspector.error_recorder=", inspector.error_recorder)  # ErrorRecorder can be set into EngineInspector, usage of ErrorRecorder refer to 09-Advance/ErrorRecorder

print("Engine information:")  # engine information is equivalent to put all layer information together
print(inspector.get_engine_information(trt.LayerInformationFormat.ONELINE))  # .txt format
#print(inspector.get_engine_information(trt.LayerInformationFormat.JSON))  # .json format

print("Layer information:")
for i in range(engine.num_layers):
    print(inspector.get_layer_information(i, trt.LayerInformationFormat.ONELINE))
