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
from utils import TRTWrapperV1, TRTWrapperV2, case_mark

shape = [1, 3, 4, 5]
data = np.random.permutation(np.arange(np.prod(shape), dtype=np.float32)).reshape(shape)
data = {"inputT0": data}

@case_mark
def case_simple():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_topk(tensor, trt.TopKOperation.MAX, 2, 1 << 1)

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_op_k_axes():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_topk(tensor, trt.TopKOperation.MAX, 1, 1 << 0)
    layer.op = trt.TopKOperation.MAX
    layer.k = 2
    layer.axes = 1 << 1

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_set_input():
    tw = TRTWrapperV2()  # USe Data-Dependent-Shape and Shape-Input at the same time

    data_v2 = data["inputT0"]  # Reconstruct input data since we have one more input tensor
    data_v2 = {"inputT0": data_v2, "inputT1": np.array([2], dtype=np.int32)}

    tensor0 = tw.network.add_input("inputT0", trt.float32, data_v2["inputT0"].shape)
    tensor1 = tw.network.add_input("inputT1", trt.int32, [])
    tw.profile.set_shape_input(tensor1.name, [1], [2], [3])
    tw.config.add_optimization_profile(tw.profile)

    layer = tw.network.add_topk(tensor0, trt.TopKOperation.MAX, 1, 1 << 1)
    layer.set_input(1, tensor1)

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data_v2)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using Top 2 layer.
    case_simple()
    # Reset Top K parameters after the constructor
    case_op_k_axes()
    # Use K from input
    case_set_input()

    print("Finish")
