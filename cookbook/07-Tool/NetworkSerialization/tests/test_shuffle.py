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
#

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import (TRTWrapperShapeInput, TRTWrapperV1, case_mark, datatype_np_to_trt)

@case_mark
def case_simple():
    shape = 1, 3, 4, 5
    data = {
        "tensor": np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1) * 1000 + \
            np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1, 1) * 100 + \
            np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2], 1) * 10 + \
            np.arange(shape[3], dtype=np.float32).reshape(1, 1, 1, shape[3]) * 1,
        }

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_shuffle(tensor)
    layer.first_transpose = (0, 2, 1, 3)
    layer.reshape_dims = (1, 4, 5, 3)  # at most one -1 can be used for auto-compute
    layer.second_transpose = (0, 2, 1, 3)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_dynamic():
    shape = 1, 3, 4, 5
    data = {
        "tensor": np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1) * 1000 + \
            np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1, 1) * 100 + \
            np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2], 1) * 10 + \
            np.arange(shape[3], dtype=np.float32).reshape(1, 1, 1, shape[3]) * 1,
        }

    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    one_layer = tw.network.add_constant([1], np.array([1], dtype=np.int64))  # Shape constant tensor need to be INT64

    shape_layer_0 = tw.network.add_shape(tensor)
    shape_layer_1 = tw.network.add_concatenation([shape_layer_0.get_output(0), one_layer.get_output(0)])
    shape_layer_1.axis = 0

    shuffle_layer = tw.network.add_shuffle(tensor)  # add one tail dimension 1 to input tensor
    shuffle_layer.set_input(1, shape_layer_1.get_output(0))
    #shuffle_layer = tw.network.add_shuffle(tensor)  # wrong because shape may contain -1 and cannot be used new shape
    #shuffle_layer.reshape_dims = tuple(tensor.shape) + (1,)

    shape_layer_2 = tw.network.add_shape(shuffle_layer.get_output(0))
    shape_layer_3 = tw.network.add_slice(shape_layer_2.get_output(0), [0], [4], [1])

    shuffle_layer_2 = tw.network.add_shuffle(shuffle_layer.get_output(0))  # remove the tail dimension 1 to input tensor
    shuffle_layer_2.set_input(1, shape_layer_3.get_output(0))
    #shuffle_layer_2 = tw.network.add_shuffle(shuffle_layer.get_output(0))  # wrong
    #shuffle_layer_2.reshape_dims = tuple(shuffle_layer.get_output(0))[:-1]

    tw.build([shuffle_layer_2.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_static_shape():
    shape = 1, 3, 4, 5
    data = {
        "tensor": np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1) * 1000 + \
            np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1, 1) * 100 + \
            np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2], 1) * 10 + \
            np.arange(shape[3], dtype=np.float32).reshape(1, 1, 1, shape[3]) * 1,
        }
    shape_output = [1, 4, 5, 3]

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    constant_layer = tw.network.add_constant([4], np.array(shape_output, dtype=np.int32))
    layer = tw.network.add_resize(tensor)
    layer.set_input(1, constant_layer.get_output(0))

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_shape_input():
    shape = 1, 3, 4, 5
    data = {
        "tensor": np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1) * 1000 + \
            np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1, 1) * 100 + \
            np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2], 1) * 10 + \
            np.arange(shape[3], dtype=np.float32).reshape(1, 1, 1, shape[3]) * 1,
        "tensor1": np.array([1, 4, 5, 3], dtype=np.int32),
    }

    tw = TRTWrapperShapeInput()

    tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    tw.profile.set_shape_input(tensor1.name, [1 for _ in data["tensor1"]], data["tensor1"], data["tensor1"])
    tw.config.add_optimization_profile(tw.profile)

    layer = tw.network.add_shuffle(tensor0)
    layer.set_input(1, tensor1)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_zero():
    shape = 1, 3, 4, 5
    data = {
        "tensor": np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1) * 1000 + \
            np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1, 1) * 100 + \
            np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2], 1) * 10 + \
            np.arange(shape[3], dtype=np.float32).reshape(1, 1, 1, shape[3]) * 1,
        }

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_shuffle(tensor)
    layer.reshape_dims = (0, 0, -1)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_zero_is_placeholder():
    shape = 1, 3, 4, 5
    data = {
        "tensor": np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1) * 1000 + \
            np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1, 1) * 100 + \
            np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2], 1) * 10 + \
            np.arange(shape[3], dtype=np.float32).reshape(1, 1, 1, shape[3]) * 1,
        }

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_shuffle(tensor)
    layer.zero_is_placeholder = True
    layer.reshape_dims = (0, 0, 0, 0)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_zero_is_placeholder_2():
    shape = 1, 3, 4, 5
    data = {
        "tensor": np.arange(shape[0], dtype=np.float32).reshape(shape[0], 1, 1, 1) * 1000 + \
            np.arange(shape[1], dtype=np.float32).reshape(1, shape[1], 1, 1) * 100 + \
            np.arange(shape[2], dtype=np.float32).reshape(1, 1, shape[2], 1) * 10 + \
            np.arange(shape[3], dtype=np.float32).reshape(1, 1, 1, shape[3]) * 1,
        }

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    constantLayer = tw.network.add_constant([0], trt.Weights(trt.float32))
    shuffleLayer = tw.network.add_shuffle(constantLayer.get_output(0))
    shuffleLayer.zero_is_placeholder = False
    shuffleLayer.reshape_dims = (1, 3, 4, 0)
    layer = tw.network.add_concatenation([tensor, shuffleLayer.get_output(0)])
    layer.axis = 3

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using shuffle layer
    case_simple()
    #
    case_dynamic()
    #
    case_static_shape()
    #
    case_shape_input()
    #
    case_zero()

    case_zero_is_placeholder()

    case_zero_is_placeholder_2()

    print("Finish")
