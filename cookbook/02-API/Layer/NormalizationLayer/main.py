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
from utils import TRTWrapperV1, case_mark

shape = [1, 4, 3, 5]
data0 = np.arange(np.prod(shape[2:]), dtype=np.float32).reshape(1, 1, *shape[2:])
data1 = 100 - data0
data2 = np.ones(shape[2:], dtype=np.float32).reshape(1, 1, *shape[2:])
data3 = -data2
data = {"inputT0": np.concatenate([data0, data1, data2, data3], axis=1)}

# scale and bias data varies among examples, so we do not prepare them here

@case_mark
def case_layer_normalization():
    shapeSB = [1, 1] + shape[2:]  # shape=[1,1,3,5]
    data["inputT1"] = np.ones(shapeSB, dtype=np.float32)
    data["inputT2"] = np.zeros(shapeSB, dtype=np.float32)

    tw = TRTWrapperV1()

    tensor0 = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    tensor1 = tw.network.add_input("inputT1", trt.float32, data["inputT1"].shape)
    tensor2 = tw.network.add_input("inputT2", trt.float32, data["inputT2"].shape)
    layer = tw.network.add_normalization(tensor0, tensor1, tensor2, 1 << 2 | 1 << 3)
    #layer.epsilon = 1  # Set epsilon, default value: 1e-5
    #layer.axes = 1 << 3  # Modify the axes to normalize after adding the layer
    #layer.compute_precision = trt.float16  # Set the precision of accumulator, default value: trt.float32

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_group_normalization():
    nGroup = 2
    shapeSB = [1, nGroup, 1, 1]  # shape=[1,2,1,1]
    data["inputT1"] = np.ones(shapeSB, dtype=np.float32)
    data["inputT2"] = np.zeros(shapeSB, dtype=np.float32)

    tw = TRTWrapperV1()

    tensor0 = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    tensor1 = tw.network.add_input("inputT1", trt.float32, data["inputT1"].shape)
    tensor2 = tw.network.add_input("inputT2", trt.float32, data["inputT2"].shape)
    layer = tw.network.add_normalization(tensor0, tensor1, tensor2, 1 << 2 | 1 << 3)
    layer.num_groups = nGroup

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_instance_normalization():
    shapeSB = [1] + shape[1:2] + [1, 1]  # shape[1,4,3,1]
    data["inputT1"] = np.ones(shapeSB, dtype=np.float32)
    data["inputT2"] = np.zeros(shapeSB, dtype=np.float32)

    tw = TRTWrapperV1()

    tensor0 = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    tensor1 = tw.network.add_input("inputT1", trt.float32, data["inputT1"].shape)
    tensor2 = tw.network.add_input("inputT2", trt.float32, data["inputT2"].shape)
    layer = tw.network.add_normalization(tensor0, tensor1, tensor2, 1 << 2 | 1 << 3)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    case_layer_normalization()
    case_group_normalization()
    case_instance_normalization()

    print("Finish")
