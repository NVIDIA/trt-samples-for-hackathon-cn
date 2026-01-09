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
from tensorrt_cookbook import TRTWrapperV1, case_mark, datatype_np_to_trt

@case_mark
def case_simple():
    data = {"tensor": (np.arange(48, dtype=np.float32)).reshape(3, 16) / 24 - 1}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    double_quantization_layer = tw.network.add_constant(shape=[], weights=np.array([1], dtype=np.float32))
    layer = tw.network.add_dynamic_quantize(tensor, 1, 16, trt.DataType.FP4, trt.DataType.FP8)
    layer.axis = 1  # [Optional] Reset axis later
    layer.block_size = 16  # [Optional] Reset block size later
    layer.to_type = trt.DataType.FP4  # [Optional] Reset target data type later
    layer.scale_type = trt.DataType.FP8  # [Optional] Reset scale data type later
    layer.set_input(1, double_quantization_layer.get_output(0))

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data)
    tw.infer()
    return

if __name__ == "__main__":
    # A simple case of using dynamic-quantize layer
    case_simple()

    print("Finish")
