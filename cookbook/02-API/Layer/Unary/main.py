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

data = {"tensor": np.arange(9, dtype=np.float32).reshape(3, 3) - 4}  # [0, 8] -> [-4, 4]}

@case_mark
def case_simple():
    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_unary(tensor, trt.UnaryOperation.ABS)
    layer.op = trt.UnaryOperation.ABS  # [Optional] Reset unary operator later

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # Compute absolute values of the tensor
    case_simple()

    print("Finish")
