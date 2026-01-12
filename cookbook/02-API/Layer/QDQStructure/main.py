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

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, case_mark, datatype_np_to_trt

@case_mark
def case_simple():
    data = {"tensor": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}

    tw = TRTWrapperV1()
    tw.config.set_flag(trt.BuilderFlag.INT8)
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer_q_scale = tw.network.add_constant([], np.array([60 / 127], dtype=np.float32))
    layer_dq_scale = tw.network.add_constant([], np.array([1], dtype=np.float32))
    layer_q = tw.network.add_quantize(tensor, layer_q_scale.get_output(0))
    layer_q.axis = 0  # [Optional] Modify axis to quantize
    layer_dq = tw.network.add_dequantize(layer_q.get_output(0), layer_dq_scale.get_output(0))
    layer_dq.axis = 0  # [Optional] Modify axis to dequantize

    tw.build([layer_dq.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_axis():
    data = {"tensor": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}

    tw = TRTWrapperV1()
    tw.config.set_flag(trt.BuilderFlag.INT8)
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer_q_scale = tw.network.add_constant([4], np.array([40 / 127, 80 / 127, 120 / 127, 160 / 127], dtype=np.float32))
    layer_dq_scale = tw.network.add_constant([], np.array([1], dtype=np.float32))
    layer_q = tw.network.add_quantize(tensor, layer_q_scale.get_output(0))
    layer_q.axis = 1
    layer_dq = tw.network.add_dequantize(layer_q.get_output(0), layer_dq_scale.get_output(0))

    tw.build([layer_dq.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_set_input_zero_point():
    data = {"tensor": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}

    tw = TRTWrapperV1()
    tw.config.set_flag(trt.BuilderFlag.INT8)
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer_q_scale = tw.network.add_constant([4], np.array([20 / 127, 40 / 127, 60 / 127, 80 / 127], dtype=np.float32))
    layer_q_zeropoint = tw.network.add_constant([4], np.array([0, 0, 0, 0], dtype=np.float32))  # Only all-zeros is supported
    layer_dq_scale = tw.network.add_constant([], np.array([1], dtype=np.float32))
    layer_q = tw.network.add_quantize(tensor, layer_q_scale.get_output(0))
    layer_q.axis = 1
    layer_q.set_input(2, layer_q_zeropoint.get_output(0))
    layer_dq = tw.network.add_dequantize(layer_q.get_output(0), layer_dq_scale.get_output(0))
    layer_dq.axis = 0

    tw.build([layer_dq.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_three_argument():
    data = {"tensor": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}

    tw = TRTWrapperV1()
    tw.network = tw.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer_q_scale = tw.network.add_constant([], np.array([60 / 127], dtype=np.float32))
    layer_dq_scale = tw.network.add_constant([], np.array([1], dtype=np.float32))
    layer_q = tw.network.add_quantize(tensor, layer_q_scale.get_output(0), trt.DataType.FP8)
    layer_dq = tw.network.add_dequantize(layer_q.get_output(0), layer_dq_scale.get_output(0), trt.DataType.FLOAT)

    tw.build([layer_dq.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple QDQ structure
    case_simple()
    # Axis
    case_axis()
    # Use scale and zero point from earlier layer
    case_set_input_zero_point()
    # Three-argument quantization layer
    case_three_argument()

    print("Finish")
