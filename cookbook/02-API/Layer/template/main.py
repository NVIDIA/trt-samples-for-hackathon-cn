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
from tensorrt_cookbook import TRTWrapperV1, case_mark, check_api_coverage, datatype_cast

data = {"tensor": np.arange(9, dtype=np.float32).reshape(3, 3)}

@case_mark
def case_simple():
    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), data["tensor"].shape)
    layer = tw.network.add_identity(tensor)  # Just for placeholder

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

if __name__ == "__main__":
    #
    case_simple()

    # print_enumerated_members()

    print("Finish")
