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

data = {"tensor": np.tile(np.array([1, 2, 5], dtype=np.float32).reshape(3, 1, 1), (1, 3, 3)).reshape(1, 3, 3, 3)}

@case_mark
def case_simple():
    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_lrn(tensor, 3, 1.0, 1.0, 0.0001)
    layer.window_size = 3  # [Optional]  Reset parameter later
    layer.alpha = 1.0  # [Optional]  Reset parameter later
    layer.beta = 1.0  # [Optional]  Reset parameter later
    layer.k = 0.0001  # [Optional]  Reset parameter later

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using LRN layer
    case_simple()

    print("Finish")
