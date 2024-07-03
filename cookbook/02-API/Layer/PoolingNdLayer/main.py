#
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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
#

import sys

import numpy as np
import tensorrt as trt

sys.path.append("/trtcookbook/include")
from utils import TRTWrapperV1, case_mark

shape = [1, 1, 6, 9]
nHKernel, nWKernel = 2, 2  # 池化窗口 HW
data = np.tile(np.arange(1, 1 + 9, dtype=np.float32).reshape(1, 3, 3), (shape[0], shape[1], shape[2] // 3, shape[3] // 3))
data = {"inputT0": data}

@case_mark
def case_simple():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_pooling_nd(tensor, trt.PoolingType.MAX, (nHKernel, nWKernel))

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_average_count_excludes_padding():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_pooling_nd(tensor, trt.PoolingType.AVERAGE, (nHKernel, nWKernel))
    layer.padding_nd = (1, 1)
    layer.average_count_excludes_padding = False

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_blend_factor():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_pooling_nd(tensor, trt.PoolingType.MAX_AVERAGE_BLEND, (nHKernel, nWKernel))
    layer.blend_factor = 0.5

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_blend_factor():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_pooling_nd(tensor, trt.PoolingType.MAX_AVERAGE_BLEND, (nHKernel, nWKernel))
    layer.blend_factor = 0.5

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_stride_pad():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_pooling_nd(tensor, trt.PoolingType.MAX, (nHKernel, nWKernel))
    layer.stride_nd = (1, 1)
    layer.padding_nd = (1, 1)
    layer.pre_padding = (1, 1)
    layer.post_padding = (1, 1)
    layer.padding_mode = trt.PaddingMode.SAME_UPPER

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_type_window():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_pooling_nd(tensor, trt.PoolingType.MAX, (1, 1))
    layer.type = trt.PoolingType.AVERAGE
    layer.window_size_nd = (nHKernel, nWKernel)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_3d():
    tw = TRTWrapperV1()

    shape = [1, 1, 2, 6, 9]
    cW, nHKernel, nWKernel = 2, 2, 2
    data_v2 = np.tile(data["inputT0"], (2, 1, 1, 1)).reshape(shape)
    data_v2[0, 0, 1] *= 10
    data_v2 = {"inputT0": data_v2}

    tensor = tw.network.add_input("inputT0", trt.float32, data_v2["inputT0"].shape)
    layer = tw.network.add_pooling_nd(tensor, trt.PoolingType.MAX, (cW, nHKernel, nWKernel))

    tw.build([layer.get_output(0)])
    tw.setup(data_v2)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using layer
    case_simple()
    # Whether to excludes padding elements count in denominator
    case_average_count_excludes_padding()
    # Use blend of max and average padding
    case_blend_factor()
    # Set stride and padding style
    case_stride_pad()
    #
    case_type_window()
    #
    case_3d()

    print("Finish")
