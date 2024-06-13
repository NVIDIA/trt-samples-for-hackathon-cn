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

nB, nC, nH, nW = 2, 3, 4, 5
data0 = np.arange(nB).reshape(nB, 1, 1, 1) * 1000
data0 = data0 + np.arange(nC).reshape(1, nC, 1, 1) * 100  # "+=" operator does not work because output shape is not the same as input shape
data0 = data0 + np.arange(nH).reshape(1, 1, nH, 1) * 10
data0 = data0 + np.arange(nW).reshape(1, 1, 1, nW)
data0 = data0.astype(np.float32)
data = {"inputT0": data0}
# input index data varies among examples, so we do not prepare them here

@case_mark
def case_default_mode():
    data["inputT1"] = np.array([[1, 0, 2], [0, 0, -1]], dtype=np.int32)  # index can be negetive

    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    inputT1 = tw.network.add_input("inputT1", trt.int32, data["inputT1"].shape)
    layer = tw.network.add_gather_v2(inputT0, inputT1, trt.GatherMode.DEFAULT)
    layer.axis = 2  # Modify the axis (default value: 0) to gather
    # layer.mode = trt.GatherMode.ELEMENT # Modify the mode (default value: DEFAULT) after adding the layer
    # Equivalent implementation using old API `add_gather()`, but only DEFAULT mode is supported.
    # layer = tw.network.add_gather(inputT0, inputT1, 1)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_default_mode_num_elementwise_axis_1():
    data["inputT1"] = np.array([[1, 0, 2], [0, 0, -1]], dtype=np.int32)

    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    inputT1 = tw.network.add_input("inputT1", trt.int32, data["inputT1"].shape)
    layer = tw.network.add_gather_v2(inputT0, inputT1, trt.GatherMode.DEFAULT)
    layer.axis = 1
    layer.num_elementwise_dims = 1

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_element_mode():
    data1 = np.zeros(data0.shape, dtype=np.int32)
    # use random permutation
    for i in range(data0.shape[0]):
        for j in range(data0.shape[1]):
            for k in range(data0.shape[3]):
                data1[i, j, :, k] = np.random.permutation(range(data0.shape[2]))
    data["inputT1"] = data1

    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    inputT1 = tw.network.add_input("inputT1", trt.int32, data["inputT1"].shape)
    layer = tw.network.add_gather_v2(inputT0, inputT1, trt.GatherMode.ELEMENT)
    layer.axis = 2

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_ND_mode():
    data["inputT1"] = np.array([[1, 0, 2], [0, 2, -1]], dtype=np.int32)

    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    inputT1 = tw.network.add_input("inputT1", trt.int32, data["inputT1"].shape)
    layer = tw.network.add_gather_v2(inputT0, inputT1, trt.GatherMode.ND)
    #layer.num_elementwise_dims = 0

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

def case_ND_mode_num_elementwise_axis_1():
    data["inputT1"] = np.array([[1, 0, 2], [0, 2, -1]], dtype=np.int32)

    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    inputT1 = tw.network.add_input("inputT1", trt.int32, data["inputT1"].shape)
    layer = tw.network.add_gather_v2(inputT0, inputT1, trt.GatherMode.ND)
    layer.num_elementwise_dims = 1

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    case_default_mode()
    case_default_mode_num_elementwise_axis_1()
    case_element_mode()
    case_ND_mode()
    case_ND_mode_num_elementwise_axis_1()

    print("Finish")
