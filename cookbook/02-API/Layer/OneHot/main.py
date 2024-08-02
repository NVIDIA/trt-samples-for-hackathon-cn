#
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

import sys

import numpy as np

sys.path.append("/trtcookbook/include")
from utils import TRTWrapperV1, case_mark, datatype_np_to_trt

data = {"tensor": np.array([[0, 1, 2, 3], [5, 4, 3, 2], [5, 7, 9, 11]], dtype=np.int32)}

@case_mark
def case_simple():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    value = tw.network.add_constant([2], np.ascontiguousarray([0, 1], dtype=np.float32))  # [offValue, onValue]
    depth = tw.network.add_constant([], np.ascontiguousarray(16, dtype=np.int32))  # Width of the embedding table, MUST be buildtime constant tensor
    layer = tw.network.add_one_hot(tensor, value.get_output(0), depth.get_output(0), 1)
    layer.axis = 1  # set axis  # [Optional] Reset axis to embed later

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # Embed a 3x4 tensor with a width 16 table
    case_simple()

    print("Finish")
