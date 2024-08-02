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
import tensorrt as trt

sys.path.append("/trtcookbook/include")
from utils import TRTWrapperV1, case_mark, datatype_np_to_trt

n_hk, n_wk = 2, 2
data = {"tensor": np.tile(np.arange(9, dtype=np.float32).reshape(3, 3), [1, 2, 3]) + 1}

@case_mark
def case_simple():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_pooling_nd(tensor, trt.PoolingType.AVERAGE, [n_hk, n_wk])
    layer.type = trt.PoolingType.AVERAGE  # [Optional] Reset pooling mode later
    layer.window_size_nd = [n_hk, n_wk]  # [Optional] Reset pooling widow size later
    layer.padding_nd = [1, 1]  # [Optional] Modify pooling padding
    layer.stride_nd = [1, 1]  # [Optional] Modify pooling stride
    layer.pre_padding = [1, 1]  # [Optional] Modify pooling padding
    layer.post_padding = [1, 1]  # [Optional] Modify pooling padding
    layer.padding_mode = trt.PaddingMode.SAME_UPPER  # [Optional] Modify pooling mode
    layer.average_count_excludes_padding = False  # [Optional] Modify whether to exclude padding element in average computation

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_blend_factor():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_pooling_nd(tensor, trt.PoolingType.MAX_AVERAGE_BLEND, [n_hk, n_wk])
    layer.blend_factor = 0.5  # [Optional] Modify weight of average

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_3d():
    n_ck, n_hk, n_wk = 2, 2, 2
    data1 = np.tile(data["tensor"], (2, 1, 1)).reshape([1, 2, 6, 9])
    data1[0, 0, 1] *= 10
    data1 = {"tensor": data1}

    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_pooling_nd(tensor, trt.PoolingType.MAX, [n_ck, n_hk, n_wk])

    tw.build([layer.get_output(0)])
    tw.setup(data1)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using layer
    case_simple()
    # Use blend of max and average padding
    case_blend_factor()
    # 3D pooling
    case_3d()

    print("Finish")
