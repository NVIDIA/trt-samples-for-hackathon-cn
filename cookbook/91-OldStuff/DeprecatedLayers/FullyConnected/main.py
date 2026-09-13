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
from tensorrt_cookbook import TRTWrapperV1, case_mark, datatype_cast

@case_mark
def case_simple():
    data = {"tensor": np.ones([4, 32, 1, 1], dtype=np.float32)}
    w = np.ones([32, 64], dtype=np.float32)  # Weight, flattened input channels -> output maps
    b = np.ones([64], dtype=np.float32)  # Bias per output map

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), [-1, 32, 1, 1])
    tw.profile.set_shape(tensor.name, [1, 32, 1, 1], [4, 32, 1, 1], [16, 32, 1, 1])
    # Deprecated layer: IFullyConnectedLayer (add_fully_connected) is removed since TensorRT 10; use MatrixMultiply + ElementWise instead.
    layer = tw.network.add_fully_connected(tensor, 64, trt.Weights(w), trt.Weights(b))
    # Input: T[shape0], len(shape0) >= 4, last 3 dimensions are flattened as input channels
    # Output: T[shape1], last 3 dimensions become (num_output_maps, 1, 1)
    layer.num_output_channels = 64  # [Optional] Default: set by constructor
    layer.kernel = trt.Weights(w)  # [Optional] Default: set by constructor
    layer.bias = trt.Weights(b)  # [Optional] Default: set by constructor

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using the deprecated fully-connected layer
    case_simple()

    print("Finish")
