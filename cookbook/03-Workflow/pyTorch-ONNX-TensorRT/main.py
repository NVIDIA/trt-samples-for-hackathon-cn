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

import os
from pathlib import Path

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import CookbookCalibratorMNIST, TRTWrapperV1, case_mark

model_path = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "model"
onnx_file = model_path / "model-trained.onnx"
onnx_file_int8qat = model_path / "model-trained-int8-qat.onnx"
data_path = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data"
data = {"x": np.load(data_path / "InferenceData.npy")}
calibration_data_file = data_path / "CalibrationData.npy"
shape = list(data["x"].shape)
trt_file = Path("model.trt")
int8_cache_file = Path("model.Int8Cache")

@case_mark
def case_normal(is_fp16: bool = False, is_int8_ptq: bool = False):
    tw = TRTWrapperV1()

    parser = trt.OnnxParser(tw.network, tw.logger)
    with open(onnx_file, "rb") as model:
        parser.parse(model.read())

    input_tensor = tw.network.get_input(0)
    tw.profile.set_shape(input_tensor.name, shape, [1] + shape[1:], [4] + shape[1:])
    tw.config.add_optimization_profile(tw.profile)

    suffix = ""
    if is_fp16:  # FP16 and INT8 can be used at the same time
        print("Using FP16")
        tw.config.set_flag(trt.BuilderFlag.FP16)
        suffix += "-fp16"
    if is_int8_ptq:
        print("Using INT8-PTQ")
        tw.config.set_flag(trt.BuilderFlag.INT8)
        input_info = {"x": [data["x"].dtype, data["x"].shape]}
        tw.config.int8_calibrator = CookbookCalibratorMNIST(input_info, calibration_data_file, int8_cache_file)
        suffix += "-int8ptq"

    tw.build()
    tw.serialize_engine(Path(str(trt_file) + suffix))

    tw.setup(data)
    tw.infer()
    return

@case_mark
def case_int8_qat():
    print("Using INT8-QAT")
    tw = TRTWrapperV1()

    parser = trt.OnnxParser(tw.network, tw.logger)
    with open(onnx_file_int8qat, "rb") as model:
        parser.parse(model.read())

    input_tensor = tw.network.get_input(0)
    tw.profile.set_shape(input_tensor.name, shape, [2] + shape[1:], [4] + shape[1:])
    tw.config.add_optimization_profile(tw.profile)

    suffix = "-int8pat"
    tw.config.set_flag(trt.BuilderFlag.INT8)  # No more work needed besides this

    tw.build()
    tw.serialize_engine(Path(str(trt_file) + suffix))

    tw.setup(data)
    tw.infer()
    return

if __name__ == "__main__":
    os.system("rm -rf *.trt* *.Int8Cache")

    case_normal()
    case_normal(is_fp16=True)
    case_normal(is_int8_ptq=True)
    case_int8_qat()

    print("Finish")
