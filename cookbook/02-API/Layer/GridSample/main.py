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

shape = [1, 3, 4, 5]
shape1 = [6, 10]
data0 = np.arange(shape[0]).reshape(shape[0], 1, 1, 1) * 1000 + \
    np.arange(shape[1]).reshape(1, shape[1], 1, 1) * 100 + \
    np.arange(shape[2]).reshape(1, 1, shape[2], 1) * 10 + \
    np.arange(shape[3]).reshape(1, 1, 1, shape[3])
data0 = data0.astype(np.float32)
dataX = np.random.randint(0, shape[2], [shape[0], shape1[0], shape1[1], 1], dtype=np.int32) / (shape[2] - 1) * 2 - 1
dataY = np.random.randint(0, shape[3], [shape[0], shape1[0], shape1[1], 1], dtype=np.int32) / (shape[3] - 1) * 2 - 1
data1 = np.concatenate([dataX, dataY], axis=3).astype(np.float32)

data = {"tensor": data0, "tensor1": data1}

@case_mark
def case_simple():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    layer = tw.network.add_grid_sample(tensor, tensor1)
    layer.align_corners = False  # [Optional] Modify corner alignment algorithm after constructor
    layer.interpolation_mode = trt.InterpolationMode.LINEAR  # [Optional] Modify interpolation algorithm after constructor
    layer.sample_mode = trt.SampleMode.FILL  # # [Optional] Modify sample algorithm after constructor

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using grid sample layer
    case_simple()

    print("Finish")
