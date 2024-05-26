#
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

import sys

import numpy as np
import tensorrt as trt

sys.path.append("/trtcookbook/include")
from example_network import get_mnist_network
from utils import MyAlgorithmSelector, TRTWrapperV1

shape = [1, 1, 28, 28]
data = np.random.rand(np.prod(shape)).astype(np.float32).reshape(shape) * 2 - 1

tw = TRTWrapperV1()
tw.config.algorithm_selector = MyAlgorithmSelector(1, True)  # assign Algorithm Selector to BuilderConfig
tw.config.set_flag(trt.BuilderFlag.FP16)  # add FP16 to get more alternative algorithms

output_tensor_list = get_mnist_network(tw.config, tw.network, tw.profile)

tw.build(output_tensor_list)
