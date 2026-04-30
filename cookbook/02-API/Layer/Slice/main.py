# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
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

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import (TRTWrapperDDS, TRTWrapperShapeInput, TRTWrapperV1, case_mark, datatype_cast, print_enumerated_members, check_api_coverage)

@case_mark
def case_simple():
    shape = [1, 3, 4, 5]
    data = {
        "tensor": np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1) * 1000 + \
        np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1, 1) * 100 + \
        np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2], 1) * 10 + \
        np.arange(shape[3], dtype=np.float32).reshape(1, 1, 1, shape[3]),
    }

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_slice(tensor, [0, 0, 0, 0], [1, 2, 3, 4], [1, 1, 1, 1])
    # Input: input0 - tensor of type T with shape [d0,...,dn-1]; input1 (optional) - Int32/Int64 tensor with start [n]; input2 (optional) - Int32/Int64 tensor with size [n]; input3 (optional) - Int32/Int64 tensor with stride [n]; input4 (optional) - tensor of type T containing fill value for FILL mode; input5 (optional) - Int32/Int64 tensor with axes [m]
    # Outputs: output - tensor of type T with shape [size0,...,sizen-1]
    # Data type: T in {bool, int4, int8, int32, int64, float8, float16, float32, bfloat16}
    # Shape: input0 shape [d0,...,dn-1]; output shape [size0,...,sizen-1] determined by start, size, stride
    layer.start = [0, 0, 0, 0]  # Reset later
    layer.shape = [1, 2, 3, 4]  # Reset later
    layer.stride = [1, 1, 1, 1]  # Reset later
    layer.mode = trt.SampleMode.WRAP  # [Optional] Default: trt.SampleMode.STRICT_BOUNDS (error if out of bound)

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_pad():
    shape = [1, 3, 4, 5]
    data = {
        "tensor": np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1) * 1000 + \
        np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1, 1) * 100 + \
        np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2], 1) * 10 + \
        np.arange(shape[3], dtype=np.float32).reshape(1, 1, 1, shape[3]),
    }

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer1 = tw.network.add_constant([1], np.array([-1], dtype=np.float32))  # Value of out-of-bound
    layer = tw.network.add_slice(tensor, [0, 0, 0, 0], [1, 2, 3, 4], [1, 2, 2, 2])
    layer.mode = trt.SampleMode.FILL  # [Optional] Default: trt.SampleMode.STRICT_BOUNDS (error if out of bound)
    layer.set_input(4, layer1.get_output(0))

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_set_input():
    shape = [1, 3, 4, 5]
    data = {
        "tensor": np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1) * 1000 + \
        np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1, 1) * 100 + \
        np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2], 1) * 10 + \
        np.arange(shape[3], dtype=np.float32).reshape(1, 1, 1, shape[3]),
    }

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer1 = tw.network.add_constant([4], np.array([0, 0, 0, 0], dtype=np.int32))
    layer2 = tw.network.add_constant([4], np.array([1, 2, 3, 4], dtype=np.int32))
    layer3 = tw.network.add_constant([4], np.array([1, 1, 1, 1], dtype=np.int32))
    layer = tw.network.add_slice(tensor, [], [], [])
    layer.set_input(1, layer1.get_output(0))
    layer.set_input(2, layer2.get_output(0))
    layer.set_input(3, layer3.get_output(0))

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_shape_input():
    shape = [1, 3, 4, 5]
    data = {
        "tensor": np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1) * 1000 + \
        np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1, 1) * 100 + \
        np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2], 1) * 10 + \
        np.arange(shape[3], dtype=np.float32).reshape(1, 1, 1, shape[3]),
        "tensor1": np.array([0, 0, 0, 0], dtype=np.int32),
        "tensor2": np.array([1, 2, 3, 4], dtype=np.int32),
        "tensor3": np.array([1, 1, 1, 1], dtype=np.int32),
    }

    tw = TRTWrapperShapeInput()
    tensor0 = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_cast(data["tensor1"].dtype, "trt"), data["tensor1"].shape)
    tensor2 = tw.network.add_input("tensor2", datatype_cast(data["tensor2"].dtype, "trt"), data["tensor2"].shape)
    tensor3 = tw.network.add_input("tensor3", datatype_cast(data["tensor3"].dtype, "trt"), data["tensor3"].shape)
    tw.profile.set_shape_input(tensor1.name, [0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2])
    tw.profile.set_shape_input(tensor2.name, [1, 1, 1, 1], [1, 2, 3, 4], [1, 3, 4, 5])
    tw.profile.set_shape_input(tensor3.name, [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1])
    layer = tw.network.add_slice(tensor0, [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0])
    layer.set_input(1, tensor1)
    layer.set_input(2, tensor2)
    layer.set_input(3, tensor3)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_dds():
    shape = [1, 3, 4, 5]
    data = {
        "tensor": np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1) * 1000 + \
        np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1, 1) * 100 + \
        np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2], 1) * 10 + \
        np.arange(shape[3], dtype=np.float32).reshape(1, 1, 1, shape[3]),
        "tensor1": np.array([1, 2, 3, 4], dtype=np.int32),
    }

    tw = TRTWrapperDDS()  # Use Data-Dependent-Shape mode
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_cast(data["tensor1"].dtype, "trt"), [-1 for _ in data["tensor1"].shape])  # tensor1 is a execution input tensor
    tw.profile.set_shape(tensor1.name, data["tensor1"].shape, data["tensor1"].shape, data["tensor1"].shape)

    layer1 = tw.network.add_elementwise(tensor1, tensor1, trt.ElementWiseOperation.SUM)  # Compute shape tensor from earlier layer
    layer = tw.network.add_slice(tensor, [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1])
    layer.set_input(2, layer1.get_output(0))

    tw.build([layer.get_output(0)])
    tw.setup(data)
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
    # case_dds()  # Disable this case since TRT  does not support such usage yet

    print_enumerated_members(trt.SampleMode)
    print_enumerated_members(trt.ElementWiseOperation)

    print("Finish")
