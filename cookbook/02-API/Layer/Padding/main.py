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

data = {"tensor": np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5) + 1}

@case_mark
def case_simple():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_padding_nd(tensor, [1, 2], [3, 4])
    layer.pre_padding_nd = [1, 2]  # [Optional] Reset up and left padding later
    layer.post_padding_nd = [3, 4]  # [Optional] Reset down and right padding later

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_crop():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_padding_nd(tensor, [-1, 0], [0, -2])

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # Pad 0 around the input tensor
    case_simple()
    # Use pad layer to crop the input tensor
    case_crop()

    print("Finish")
