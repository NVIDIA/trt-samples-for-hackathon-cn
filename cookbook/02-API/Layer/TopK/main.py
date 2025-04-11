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

from tensorrt_cookbook import (TRTWrapperDDS, TRTWrapperV1, TRTWrapperV2, case_mark, datatype_np_to_trt)

data = {"tensor": np.random.permutation(np.arange(60, dtype=np.float32)).reshape(3, 4, 5)}

@case_mark
def case_simple():
    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_topk(tensor, trt.TopKOperation.MAX, 2, 1 << 1)
    layer.op = trt.TopKOperation.MAX  # [Optional] Reset sort direction later
    layer.k = 2  # [Optional] Reset number to remain later
    layer.axes = 1 << 1  # [Optional] Reset axis to sort later

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_shape_input():
    data1 = {"tensor": data["tensor"], "tensor1": np.array([2], dtype=np.int32)}  # One more shape input tensor

    tw = TRTWrapperV2()  # Use Data-Dependent-Shape and Shape-Input mode at the same time

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), data1["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data1["tensor1"].dtype), [])
    tw.profile.set_shape_input(tensor1.name, [1], [2], [3])
    tw.config.add_optimization_profile(tw.profile)

    layer = tw.network.add_topk(tensor, trt.TopKOperation.MAX, 1, 1 << 1)
    layer.set_input(1, tensor1)

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data1)
    tw.infer()

@case_mark
def case_dds():
    data1 = {"tensor": data["tensor"], "tensor1": np.array([3, -1], dtype=np.int32)}  # tensor1 is a execution input tensor

    tw = TRTWrapperDDS()  # Use Data-Dependent-Shape mode

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), data1["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data1["tensor1"].dtype), [-1 for _ in data1["tensor1"].shape])
    tw.profile.set_shape(tensor1.name, [1 for _ in data1["tensor1"].shape], data1["tensor1"].shape, data1["tensor1"].shape)
    tw.config.add_optimization_profile(tw.profile)

    layer1 = tw.network.add_reduce(tensor1, trt.ReduceOperation.SUM, 1 << 0, False)  # Compute K from earlier layer
    layer = tw.network.add_topk(tensor, trt.TopKOperation.MAX, 1, 1 << 1)
    layer.set_input(1, layer1.get_output(0))

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data1)
    tw.infer()

if __name__ == "__main__":
    # Get Top 2 from input tensor
    case_simple()
    # Use K from shape input tensor
    case_shape_input()
    # USe K from output of earlier layer
    case_dds()

    print("Finish")
