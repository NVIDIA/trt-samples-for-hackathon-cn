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

from tensorrt_cookbook import TRTWrapperV1, case_mark, datatype_np_to_trt

shape = [1, 4, 3, 5]
data0 = np.arange(15, dtype=np.float32).reshape(1, 1, 3, 5)
data1 = 100 - data0
data2 = np.ones([1, 1, 3, 5], dtype=np.float32)
data3 = -data2
data = {"tensor": np.concatenate([data0, data1, data2, data3], axis=1)}

@case_mark
def case_layer_normalization():
    shape_scale_bias = (1, 1) + data["tensor"].shape[2:]  # [1, 1, 3, 5]

    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer1 = tw.network.add_constant(shape_scale_bias, trt.Weights(np.ones(shape_scale_bias, dtype=np.float32)))
    layer2 = tw.network.add_constant(shape_scale_bias, trt.Weights(np.zeros(shape_scale_bias, dtype=np.float32)))
    layer = tw.network.add_normalization(tensor, layer1.get_output(0), layer2.get_output(0), 1 << 2 | 1 << 3)
    layer.axes = 1 << 2 | 1 << 3  # [Optional] Reset the axes to normalize later
    layer.compute_precision = trt.float16  # [Optional] Modify the precision of accumulator
    layer.epsilon = 1e-5  # [Optional] Modify epsilon

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_group_normalization():
    n_group = 2
    shape_scale_bias = [1, n_group, 1, 1]  # [1, 2, 1, 1]

    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer1 = tw.network.add_constant(shape_scale_bias, trt.Weights(np.ones(shape_scale_bias, dtype=np.float32)))
    layer2 = tw.network.add_constant(shape_scale_bias, trt.Weights(np.zeros(shape_scale_bias, dtype=np.float32)))
    layer = tw.network.add_normalization(tensor, layer1.get_output(0), layer2.get_output(0), 1 << 2 | 1 << 3)
    layer.num_groups = n_group  # [Optional] Modify the number of groups

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_instance_normalization():
    shape_scale_bias = (1, ) + data["tensor"].shape[1:2] + (1, 1)  # [1, 4, 1, 1]

    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer1 = tw.network.add_constant(shape_scale_bias, trt.Weights(np.ones(shape_scale_bias, dtype=np.float32)))
    layer2 = tw.network.add_constant(shape_scale_bias, trt.Weights(np.zeros(shape_scale_bias, dtype=np.float32)))
    layer = tw.network.add_normalization(tensor, layer1.get_output(0), layer2.get_output(0), 1 << 2 | 1 << 3)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # Layer normalization
    case_layer_normalization()
    # Group normalization
    case_group_normalization()
    # Instance normalization
    case_instance_normalization()

    print("Finish")
