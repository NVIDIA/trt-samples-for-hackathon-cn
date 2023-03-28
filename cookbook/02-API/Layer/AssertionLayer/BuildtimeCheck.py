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

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.float32, shape)
#------------------------------------------------------------------------------- Network
_H1 = network.add_shape(inputT0)
_H2 = network.add_slice(_H1.get_output(0), [3], [1], [1])

_C1 = network.add_constant([1], np.array([5], dtype=np.int32))  # check condition inputT0.shape[3] == 5, no error with this
#_C1 = network.add_constant([1], np.array([4], dtype=np.int32))  # check condition inputT0.shape[3] == 4, it certainly fails at build time

_H3 = network.add_elementwise(_H2.get_output(0), _C1.get_output(0), trt.ElementWiseOperation.EQUAL)  # do this check by elementwise layer

_H4 = network.add_identity(_H3.get_output(0))
_H4.get_output(0).dtype = trt.bool
_HA = network.add_assertion(_H4.get_output(0), "inputT0.shape[3] != 5")  # assertion layer has a Bool input tensor and no output tensor
_H5 = network.add_identity(_H4.get_output(0))
_H5.get_output(0).dtype = trt.int32
#------------------------------------------------------------------------------- Network
network.mark_output(_H5.get_output(0))

engineString = builder.build_serialized_network(network, config)
print("%s building serialized network!" % ("Failed" if engineString == None else "Succeeded"))