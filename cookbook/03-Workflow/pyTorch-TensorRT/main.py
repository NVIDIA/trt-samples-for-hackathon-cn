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

import os
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

sys.path.append("/trtcookbook/include")
from utils import (MyCalibratorMNIST, TRTWrapperV1, build_mnist_network_trt, case_mark)

data_path = Path("/trtcookbook/00-Data/data")
model_path = Path("/trtcookbook/00-Data/model")
weight_file = model_path / "model-trained.npz"
calibration_data_file = data_path / "CalibrationData.npy"
data = {"x": np.load(data_path / "InferenceData.npy")}
shape = list(data["x"].shape)
trt_file = Path("model.trt")
int8_cache_file = Path("cache.Int8Cache")

@case_mark
def case_normal(is_fp16: bool = False, is_int8_ptq: bool = False):
    tw = TRTWrapperV1()

    output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile, True)

    suffix = ""
    if is_fp16:  # FP16 and INT8 can be used at the same time
        print("Using FP16")
        tw.config.set_flag(trt.BuilderFlag.FP16)
        suffix += "-fp16"
    if is_int8_ptq:
        print("Using INT8-PTQ")
        tw.config.set_flag(trt.BuilderFlag.INT8)
        input_info = {"x": [data["x"].dtype, data["x"].shape]}
        tw.config.int8_calibrator = MyCalibratorMNIST(input_info, calibration_data_file, int8_cache_file)
        suffix += "-int8ptq"

    tw.build(output_tensor_list)
    tw.serialize_engine(Path(str(trt_file) + suffix))

    tw.setup(data)
    tw.infer()
    return

if __name__ == "__main__":
    os.system("rm -rf *.trt* *.Int8Cache")

    case_normal()
    case_normal(is_fp16=True)
    case_normal(is_int8_ptq=True)

    print("Finish")
