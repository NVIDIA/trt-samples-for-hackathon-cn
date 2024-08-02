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

data = {"tensor": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}

@case_mark
def case_simple():
    tw = TRTWrapperV1()
    tw.config.set_flag(trt.BuilderFlag.INT8)

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer1 = tw.network.add_constant([], np.array([60 / 127], dtype=np.float32))
    layer2 = tw.network.add_constant([], np.array([1], dtype=np.float32))
    layer3 = tw.network.add_quantize(tensor, layer1.get_output(0))
    layer3.axis = 0  # [Optional] Modify axis to quantize
    layer4 = tw.network.add_dequantize(layer3.get_output(0), layer2.get_output(0))
    layer4.axis = 0  # [Optional] Modify axis to dequantize

    tw.build([layer4.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_axis():
    tw = TRTWrapperV1()
    tw.config.set_flag(trt.BuilderFlag.INT8)

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer1 = tw.network.add_constant([3], np.array([60 / 127, 120 / 127, 240 / 127], dtype=np.float32))
    layer2 = tw.network.add_constant([], np.array([1], dtype=np.float32))
    layer3 = tw.network.add_quantize(tensor, layer1.get_output(0))
    layer3.axis = 1
    layer4 = tw.network.add_dequantize(layer3.get_output(0), layer2.get_output(0))

    tw.build([layer4.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_set_input_zero_point():
    tw = TRTWrapperV1()
    tw.config.set_flag(trt.BuilderFlag.INT8)

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)

    layer0 = tw.network.add_constant([3], np.array([20 / 127, 40 / 127, 60 / 127], dtype=np.float32))
    layer1 = tw.network.add_constant([], np.array([1], dtype=np.float32))
    layer2 = tw.network.add_constant([3], np.array([0, 0, 0], dtype=np.float32))  # [-60, -96, -106]

    layer3 = tw.network.add_quantize(tensor, layer0.get_output(0))
    layer3.axis = 1
    layer3.set_input(1, layer0.get_output(0))
    layer3.set_input(2, layer2.get_output(0))
    layer4 = tw.network.add_dequantize(layer3.get_output(0), layer1.get_output(0))
    layer4.axis = 0

    tw.build([layer4.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple QDQ structure
    case_simple()
    # Axis
    case_axis()
    # Use scale and zero point from earlier layer
    case_set_input_zero_point()
    #

    print("Finish")
