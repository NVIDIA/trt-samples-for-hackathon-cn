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
from utils import TRTWrapperShapeInput, TRTWrapperV1, case_mark

# input data varies among examples, so we do not prepare them here

@case_mark
def case_linspace():
    output_shape = [2, 3, 4, 5]
    data0 = np.array(1000, dtype=np.float32)  # Initialization value
    data1 = np.array([0, 100, 10, 1], dtype=np.float32)  # Stride value, Rank of this data muat be equal to rank of output tensor
    data = {"inputT0": data0, "inputT1": data1}

    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    inputT1 = tw.network.add_input("inputT1", trt.float32, data["inputT1"].shape)
    layer = tw.network.add_fill(output_shape, trt.FillOperation.LINSPACE)
    layer.set_input(1, inputT0)
    layer.set_input(2, inputT1)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_random_normal():
    output_shape = [2, 3, 4, 5]
    data0 = np.array(0, dtype=np.float32)  # mean value
    data1 = np.array(0.92, dtype=np.float32)  # when scale is 0.92, output standard deviation is 1.0
    data = {"inputT0": data0, "inputT1": data1}

    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    inputT1 = tw.network.add_input("inputT1", trt.float32, data["inputT1"].shape)
    layer = tw.network.add_fill(output_shape, trt.FillOperation.RANDOM_NORMAL)
    layer.set_input(1, inputT0)
    layer.set_input(2, inputT1)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_random_uniform():
    output_shape = [2, 3, 4, 5]
    data0 = np.array(5, dtype=np.float32)  # minimum value
    data1 = np.array(10, dtype=np.float32)  # maximum value
    data = {"inputT0": data0, "inputT1": data1}

    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    inputT1 = tw.network.add_input("inputT1", trt.float32, data["inputT1"].shape)
    layer = tw.network.add_fill(output_shape, trt.FillOperation.RANDOM_UNIFORM)
    layer.set_input(1, inputT0)
    layer.set_input(2, inputT1)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_to_type():
    output_shape = [2, 3, 4, 5]
    data0 = np.array(1000, dtype=np.int64)  # Initialization value
    data1 = np.array([0, 100, 10, 1], dtype=np.int64)  # Stride value, Rank of this data muat be equal to rank of output tensor
    data = {"inputT0": data0, "inputT1": data1}

    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.int64, data["inputT0"].shape)
    inputT1 = tw.network.add_input("inputT1", trt.int64, data["inputT1"].shape)
    layer = tw.network.add_fill(output_shape, trt.FillOperation.LINSPACE)
    layer.set_input(1, inputT0)
    layer.set_input(2, inputT1)
    layer.to_type = trt.int32  # set output data type
    print(f"{layer.is_alpha_beta_int64() = }")  # always False though we set inputT* as INT64?

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_shape_input():
    output_shape = [2, 3, 4, 5]
    data0 = np.array(output_shape, dtype=np.int32)
    data1 = np.float32(1000)
    data2 = np.array([0, 100, 10, 1], dtype=np.float32)
    data = {"inputT0": data0, "inputT1": data1, "inputT2": data2}

    tw = TRTWrapperShapeInput()

    inputT0 = tw.network.add_input("inputT0", trt.int32, [4])
    inputT1 = tw.network.add_input("inputT1", trt.float32, [])
    inputT2 = tw.network.add_input("inputT2", trt.float32, [4])
    tw.profile.set_shape_input(inputT0.name, [1, 1, 1, 1], output_shape, output_shape)  # range of value rather than shape
    tw.config.add_optimization_profile(tw.profile)

    layer = tw.network.add_fill([1], trt.FillOperation.LINSPACE)
    layer.set_input(0, inputT0)
    layer.set_input(1, inputT1)
    layer.set_input(2, inputT2)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_shape_runtime():
    output_shape = [2, 3, 4, 5]
    data0 = np.ones(output_shape, dtype=np.int32)
    data1 = np.float32(1000)
    data2 = np.array([0, 100, 10, 1], dtype=np.float32)
    data = {"inputT0": data0, "inputT1": data1, "inputT2": data2}

    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, [-1 for _ in output_shape])
    inputT1 = tw.network.add_input("inputT1", trt.float32, [])
    inputT2 = tw.network.add_input("inputT2", trt.float32, [4])
    tw.profile.set_shape(inputT0.name, [1, 1, 1, 1], output_shape, output_shape)
    tw.config.add_optimization_profile(tw.profile)

    layer = tw.network.add_shape(inputT0)
    tensor = layer.get_output(0)

    layer = tw.network.add_fill([1], trt.FillOperation.LINSPACE)
    layer.set_input(0, tensor)
    layer.set_input(1, inputT1)
    layer.set_input(2, inputT2)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # Produce a layer with linear space with initinalization and stride value.
    case_linspace()
    # Produce a layer with normal distribution with mean and standard-deviation value.
    case_random_normal()
    # Produce a layer with uniform distribution with minimum and maximum value.
    case_random_uniform()
    # Modify output data type of the layer.
    case_to_type()
    # Decide output shape of the layer at runtime by using shape input tensor.
    case_shape_input()
    # Decide output shape of the layer at runtime by output tensor from a prior layer.
    case_shape_runtime()

    print("Finish")
