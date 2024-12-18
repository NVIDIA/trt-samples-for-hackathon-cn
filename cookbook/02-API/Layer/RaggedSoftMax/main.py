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

from tensorrt_cookbook import TRTWrapperV1, case_mark, datatype_np_to_trt

shape = [1, 3, 4, 5]
data0 = np.ones(np.prod(shape), dtype=np.float32).reshape(shape[1:])
data1 = np.tile(2 * np.arange(shape[2], dtype=np.int32), (shape[1], 1)).reshape(shape[1], shape[2], 1)
data = {"tensor": data0, "tensor1": data1}

@case_mark
def case_simple():
    tw = TRTWrapperV1()

    tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    layer = tw.network.add_ragged_softmax(tensor0, tensor1)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using ragged softmax layer
    case_simple()

    print("Finish")
