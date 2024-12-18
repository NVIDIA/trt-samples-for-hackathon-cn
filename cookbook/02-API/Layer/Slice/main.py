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

from tensorrt_cookbook import (TRTWrapperDDS, TRTWrapperShapeInput, TRTWrapperV1, case_mark, datatype_np_to_trt)

shape = [1, 3, 4, 5]
data = np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1) * 1000 + \
    np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1, 1) * 100 + \
    np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2], 1) * 10 + \
    np.arange(shape[3], dtype=np.float32).reshape(1, 1, 1, shape[3])
data = {"tensor": data}

@case_mark
def case_simple():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_slice(tensor, [0, 0, 0, 0], [1, 2, 3, 4], [1, 1, 1, 1])
    layer.start = [0, 0, 0, 0]  # [Optional] Reset start index later
    layer.shape = [1, 2, 3, 4]  # [Optional] Reset output shape later
    layer.stride = [1, 1, 1, 1]  # [Optional] Reset stride index later
    layer.mode = trt.SampleMode.WRAP  # [Optional] Modify slice mode

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_pad():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer1 = tw.network.add_constant([1], np.array([-1], dtype=np.float32))  # Value of out-of-bound
    layer = tw.network.add_slice(tensor, [0, 0, 0, 0], [1, 2, 3, 4], [1, 2, 2, 2])
    layer.mode = trt.SampleMode.FILL
    layer.set_input(4, layer1.get_output(0))

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_set_input():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer1 = tw.network.add_constant([4], np.array([0, 0, 0, 0], dtype=np.int32))
    layer2 = tw.network.add_constant([4], np.array([1, 2, 3, 4], dtype=np.int32))
    layer3 = tw.network.add_constant([4], np.array([1, 1, 1, 1], dtype=np.int32))
    layer = tw.network.add_slice(tensor, [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0])
    layer.set_input(1, layer1.get_output(0))
    layer.set_input(2, layer2.get_output(0))
    layer.set_input(3, layer3.get_output(0))

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_shape_input():
    data1 = {"tensor": data["tensor"]}
    data1["tensor1"] = np.array([0, 0, 0, 0], dtype=np.int32)
    data1["tensor2"] = np.array([1, 2, 3, 4], dtype=np.int32)
    data1["tensor3"] = np.array([1, 1, 1, 1], dtype=np.int32)

    tw = TRTWrapperShapeInput()

    tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), data1["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data1["tensor1"].dtype), data1["tensor1"].shape)
    tensor2 = tw.network.add_input("tensor2", datatype_np_to_trt(data1["tensor2"].dtype), data1["tensor2"].shape)
    tensor3 = tw.network.add_input("tensor3", datatype_np_to_trt(data1["tensor3"].dtype), data1["tensor3"].shape)
    tw.profile.set_shape_input(tensor1.name, [0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2])
    tw.profile.set_shape_input(tensor2.name, [1, 1, 1, 1], [1, 2, 3, 4], [1, 3, 4, 5])
    tw.profile.set_shape_input(tensor3.name, [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1])
    tw.config.add_optimization_profile(tw.profile)
    layer = tw.network.add_slice(tensor0, [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0])
    layer.set_input(1, tensor1)
    layer.set_input(2, tensor2)
    layer.set_input(3, tensor3)

    tw.build([layer.get_output(0)])
    tw.setup(data1)
    tw.infer()

@case_mark
def case_dds():
    data1 = {"tensor": data["tensor"]}
    data1["tensor1"] = np.array([1, 2, 3, 4], dtype=np.int32)

    tw = TRTWrapperDDS()  # Use Data-Dependent-Shape mode

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), data1["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data1["tensor1"].dtype), [-1 for _ in data1["tensor1"].shape])  # tensor1 is a execution input tensor
    tw.profile.set_shape(tensor1.name, data1["tensor1"].shape, data1["tensor1"].shape, data1["tensor1"].shape)
    tw.config.add_optimization_profile(tw.profile)

    layer1 = tw.network.add_elementwise(tensor1, tensor1, trt.ElementWiseOperation.SUM)  # Compute shape tensor from earlier layer
    layer = tw.network.add_slice(tensor, [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1])
    layer.set_input(2, layer1.get_output(0))

    tw.build([layer.get_output(0)])
    tw.setup(data1)
    tw.infer()

if __name__ == "__main__":
    # Slice input tensor
    case_simple()
    # Use slice layer to do padding
    case_pad()
    # Use start, shape and stride from earlier layers without Data-Dependent-Shape mode
    case_set_input()
    # Use start, shape and stride from shape input tensor
    case_shape_input()
    # Use start, shape and stride from earlier layers with Data-Dependent-Shape mode
    #case_dds()  # Unfini

    print("Finish")
