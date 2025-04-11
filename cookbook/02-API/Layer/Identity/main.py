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

data = {"tensor": np.arange(np.prod(60), dtype=np.float32).reshape(1, 3, 4, 5)}

@case_mark
def case_simple():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_identity(tensor)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_datatype_conversion():
    tw = TRTWrapperV1()
    tw.config.set_flag(trt.BuilderFlag.FP16)  # Needed if using float16
    tw.config.set_flag(trt.BuilderFlag.BF16)  # Needed if using bfloat16
    tw.config.set_flag(trt.BuilderFlag.INT8)  # Needed if using int8
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    output_tensor_list = []
    for data_type in [trt.float16, trt.bfloat16, trt.int32, trt.int64, trt.int8, trt.uint8, trt.bool]:  # trt.int4
        # FP8 / FP4 is only supported from Plugin / Quantize / Constant / Concatenation / Shuffle layer
        layer = tw.network.add_identity(tensor)
        layer.set_output_type(0, data_type)
        if data_type == trt.int8:
            layer.get_output(0).set_dynamic_range(0, 127)  # dynamic range or calibration needed for INT8
        output_tensor_list.append(layer.get_output(0))

    tw.build(output_tensor_list)

    tw.engine = trt.Runtime(tw.logger).deserialize_cuda_engine(tw.engine_bytes)
    # Print information of input / output tensors
    for i in range(tw.engine.num_io_tensors):
        name = tw.engine.get_tensor_name(i)
        mode = tw.engine.get_tensor_mode(name)
        data_type = tw.engine.get_tensor_dtype(name)
        buildtime_shape = tw.engine.get_tensor_shape(name)
        print(f"{'Input ' if mode == trt.TensorIOMode.INPUT else 'Output'}->{data_type}, {buildtime_shape}, {name}")

if __name__ == "__main__":
    # A simple case of using Identity layer.
    case_simple()
    # Cast input tensor into FLOAT32 / FLOAT16 / INT32 / INT64 / INT8 / UINT8 / INT4 / BOOL
    case_datatype_conversion()

    print("Finish")
