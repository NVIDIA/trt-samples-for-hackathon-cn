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

data = {"tensor": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}

@case_mark
def case_simple():
    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer_axis = tw.network.add_constant(shape=(), weights=np.array([1], dtype=np.int32))
    layer = tw.network.add_cumulative(tensor, layer_axis.get_output(0), trt.CumulativeOperation.SUM, False, False)
    layer.op = trt.CumulativeOperation.SUM  # [Optional] Reset operation later
    layer.exclusive = False  # [Optional] Reset exclusive / inclusive later
    layer.reverse = False  # [Optional] Reset computation direction later

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using cumulative layer
    case_simple()

    print("Finish")
