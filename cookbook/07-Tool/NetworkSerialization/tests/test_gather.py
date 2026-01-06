# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import (TRTWrapperDDS, TRTWrapperV1, case_mark, datatype_np_to_trt)

shape = [2, 3, 4, 5]
data0 = np.arange(shape[0]).reshape(shape[0], 1, 1, 1) * 1000 + \
    np.arange(shape[1]).reshape(1, shape[1], 1, 1) * 100 + \
    np.arange(shape[2]).reshape(1, 1, shape[2], 1) * 10 + \
    np.arange(shape[3]).reshape(1, 1, 1, shape[3])
data = {"tensor": data0.astype(np.float32)}
# Input index data varies among examples, so we do not prepare it here

@case_mark
def case_default_mode():
    data["tensor1"] = np.array([[1, 0, 2], [0, 0, -1]], dtype=np.int32)  # Index can be negetive

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    layer = tw.network.add_gather_v2(tensor, tensor1, trt.GatherMode.DEFAULT)
    layer.axis = 2  # [Optional] Modify the axis to gather
    layer.mode = trt.GatherMode.DEFAULT  # [Optional] Reset gahter mode later
    # Equivalent implementation using old API `add_gather()`, but only DEFAULT mode is supported.
    # layer = tw.network.add_gather(tensor, tensor1, 1)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_default_mode_num_elementwise_axis_1():
    data["tensor1"] = np.array([[1, 0, 2], [0, 0, -1]], dtype=np.int32)

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    layer = tw.network.add_gather_v2(tensor, tensor1, trt.GatherMode.DEFAULT)
    layer.axis = 2
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
    data["tensor1"] = data1

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    layer = tw.network.add_gather_v2(tensor, tensor1, trt.GatherMode.ELEMENT)
    layer.axis = 2

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_ND_mode():
    data["tensor1"] = np.array([[1, 0, 2], [0, 2, -1]], dtype=np.int32)

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    layer = tw.network.add_gather_v2(tensor, tensor1, trt.GatherMode.ND)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_ND_mode_num_elementwise_axis_1():
    data["tensor1"] = np.array([[1, 0, 2], [0, 2, -1]], dtype=np.int32)

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    layer = tw.network.add_gather_v2(tensor, tensor1, trt.GatherMode.ND)
    layer.num_elementwise_dims = 1

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_gather_nonzeros():
    data = np.zeros([3, 4, 5]).astype(np.float32)
    data[0, 0, 1] = 1
    data[0, 2, 3] = 2
    data[0, 3, 4] = 3
    data[1, 1, 0] = 4
    data[1, 1, 1] = 5
    data[1, 1, 2] = 6
    data[1, 1, 3] = 7
    data[1, 1, 4] = 8
    data[2, 0, 1] = 9
    data[2, 1, 1] = 10
    data[2, 2, 1] = 11
    data[2, 3, 1] = 12
    data = {"tensor": data}

    tw = TRTWrapperDDS()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_non_zero(tensor)
    layer = tw.network.add_shuffle(layer.get_output(0))
    layer.first_transpose = [1, 0]
    layer = tw.network.add_gather_v2(tensor, layer.get_output(0), trt.GatherMode.ND)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # Gather layer in default mode
    case_default_mode()
    # Gather layer in default mode with parameter num_elementwise_axis=1
    case_default_mode_num_elementwise_axis_1()
    # Gather layer in element mode
    case_element_mode()
    # Gather layer in ND mode
    case_ND_mode()
    # Gather layer in ND mode with num_elementwise_axis=1
    case_ND_mode_num_elementwise_axis_1()
    # Gather all non-zero elements together
    case_gather_nonzeros()

    print("Finish")
