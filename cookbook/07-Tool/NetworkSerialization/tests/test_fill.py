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

# Input data varies among examples, so we do not prepare it here

@case_mark
def case_linspace_1():
    output_shape = [4]

    tw = TRTWrapperV1()
    layer = tw.network.add_fill(output_shape, trt.FillOperation.LINSPACE, trt.DataType.FLOAT)
    layer.alpha = [1000]  # Start value
    layer.beta = [1]  # Stride value
    layer.to_type = trt.DataType.FLOAT  # [Optional] Modify output data type
    print(f"{layer.is_alpha_beta_int64() = }")  # Read-only attribution, whether arguments alpha and beta are INT64

    tw.build([layer.get_output(0)])
    tw.setup()
    tw.infer()

@case_mark
def case_linspace_2():
    output_shape = [3, 4, 5]
    data0 = np.array(1000, dtype=np.float32)  # Start value
    data1 = np.array([100, 10, 1], dtype=np.float32)  # Stride value, which length must be equal to rank of output tensor
    data = {"tensor": data0, "tensor1": data1}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    layer = tw.network.add_fill(output_shape, trt.FillOperation.LINSPACE, trt.DataType.FLOAT)
    layer.set_input(1, tensor)  # Other input tensors can be input tensor or from constant layer
    layer.set_input(2, tensor1)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_random_normal():
    output_shape = [3, 4, 5]
    data0 = np.array(0, dtype=np.float32)  # Mean value
    data1 = np.array(0.92, dtype=np.float32)  # Standard deviation is 1.0 when scale is 0.92
    data = {"tensor": data0, "tensor1": data1}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    layer = tw.network.add_fill(output_shape, trt.FillOperation.RANDOM_NORMAL)
    layer.set_input(1, tensor)
    layer.set_input(2, tensor1)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_random_uniform():
    output_shape = [3, 4, 5]
    data0 = np.array(5, dtype=np.float32)  # Minimum value
    data1 = np.array(10, dtype=np.float32)  # Maximum value
    data = {"tensor": data0, "tensor1": data1}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    layer = tw.network.add_fill(output_shape, trt.FillOperation.RANDOM_UNIFORM)
    layer.set_input(1, tensor)
    layer.set_input(2, tensor1)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_shape_input():
    output_shape = [3, 4, 5]
    data0 = np.array(output_shape, dtype=np.int32)
    data1 = np.float32(1000)
    data2 = np.array([100, 10, 1], dtype=np.float32)
    data = {"tensor": data0, "tensor1": data1, "tensor2": data2}

    tw = TRTWrapperShapeInput()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    tensor2 = tw.network.add_input("tensor2", datatype_np_to_trt(data["tensor2"].dtype), data["tensor2"].shape)
    tw.profile.set_shape_input(tensor.name, [1, 1, 1], output_shape, output_shape)  # Range of value rather than shape
    tw.config.add_optimization_profile(tw.profile)

    layer = tw.network.add_fill([], trt.FillOperation.LINSPACE)
    layer.set_input(0, tensor)  # Use index 0 to set output shape
    layer.set_input(1, tensor1)
    layer.set_input(2, tensor2)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_dds():
    data0 = np.zeros([3, 4, 5]).astype(np.float32)
    data0[0, 0, 1] = 1
    data0[0, 2, 3] = 2
    data0[0, 3, 4] = 3
    data0[1, 1, 0] = 4
    data0[1, 1, 1] = 5
    data0[1, 1, 2] = 6
    data0[1, 1, 3] = 7
    data0[1, 1, 4] = 8
    data0[2, 0, 1] = 9
    data0[2, 1, 1] = 10
    data0[2, 2, 1] = 11
    data0[2, 3, 1] = 12
    data1 = np.float32(1000)
    data2 = np.array([10, 1], dtype=np.float32)
    data = {"tensor": data0, "tensor1": data1, "tensor2": data2}

    tw = TRTWrapperDDS()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), [-1 for _ in data["tensor"].shape])
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    tensor2 = tw.network.add_input("tensor2", datatype_np_to_trt(data["tensor2"].dtype), data["tensor2"].shape)
    tw.profile.set_shape(tensor.name, [1, 1, 1], [3, 4, 5], [3, 4, 5])
    tw.config.add_optimization_profile(tw.profile)
    layer1 = tw.network.add_non_zero(tensor)
    layer2 = tw.network.add_shape(layer1.get_output(0))
    layer = tw.network.add_fill([], trt.FillOperation.LINSPACE)
    layer.set_input(0, layer2.get_output(0))  # Use index 0 to set output shape
    layer.set_input(1, tensor1)
    layer.set_input(2, tensor2)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # Produce a layer in linear space with initinalization and stride value (only for rank 1 tensor)
    case_linspace_1()
    # Produce a layer in linear space with initinalization and stride value tensors
    case_linspace_2()
    # Produce a layer in normal distribution with mean and standard-deviation value
    case_random_normal()
    # Produce a layer in uniform distribution with minimum and maximum value
    case_random_uniform()
    # Decide output shape of the layer at runtime by using shape input tensor
    case_shape_input()
    # Use fill layer in data-dependent-shape (DDS) mode
    case_dds()

    print("Finish")
