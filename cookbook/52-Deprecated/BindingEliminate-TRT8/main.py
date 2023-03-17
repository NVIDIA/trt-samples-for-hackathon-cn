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

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30

inputT0 = network.add_input("inputT0", trt.float32, [-1, -1, -1])
profile.set_shape(inputT0.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
inputT1 = network.add_input("inputT1", trt.float32, [-1, -1, -1])
profile.set_shape(inputT1.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
config.add_optimization_profile(profile)

identityLayer = network.add_identity(inputT0)  # 只有 inputT0 有用
network.mark_output(identityLayer.get_output(0))

#engineString = builder.build_serialized_network(network, config)
#engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
engine = builder.build_engine(network, config)

context = engine.create_execution_context()
context.set_binding_shape(0, [3, 4, 5])  # 只设定 inding 0 的形状
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput
print("nBinding=%d, nInput=%d,nOutput=%d" % (engine.num_bindings, nInput, nOutput))
for i in range(nInput):
    print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
for i in range(nInput, nInput + nOutput):
    print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
