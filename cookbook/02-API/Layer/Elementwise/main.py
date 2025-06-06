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

data0 = np.full([3, 4, 5], 2, dtype=np.float32)
data1 = np.full([3, 4, 5], 3, dtype=np.float32)
data = {"tensor": data0, "tensor1": data1}

@case_mark
def case_simple():
    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    layer = tw.network.add_elementwise(tensor, tensor1, trt.ElementWiseOperation.SUM)
    layer.op = trt.ElementWiseOperation.SUM  # [Optional] Reset operator later

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_broadcast():
    n_c, n_h, n_w = data["tensor"].shape
    data0 = np.full([n_c, 1, n_w], 1, dtype=np.float32)
    data1 = np.full([n_c, n_h, 1], 2, dtype=np.float32)
    data1 = {"tensor": data0, "tensor1": data1}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), data1["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data1["tensor1"].dtype), data1["tensor1"].shape)
    layer = tw.network.add_elementwise(tensor, tensor1, trt.ElementWiseOperation.SUM)

    tw.build([layer.get_output(0)])
    tw.setup(data1)
    tw.infer()

if __name__ == "__main__":
    # A simple case of compute elementewise addition
    case_simple()
    # Broadcast the elements while elementwise operation
    case_broadcast()

    print("Finish")
