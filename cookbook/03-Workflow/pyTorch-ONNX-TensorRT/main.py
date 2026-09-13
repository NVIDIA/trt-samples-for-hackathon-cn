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

from pathlib import Path

import numpy as np
from tensorrt_cookbook import case_mark, cookbook_path, parse_onnx, TRTWrapperV1

model_path = cookbook_path("00-Data", "model")
onnx_file = model_path / "model-trained.onnx"
onnx_file_int8qat = model_path / "model-trained-int8-qat.onnx"
data_path = cookbook_path("00-Data", "data")
data = {"x": np.load(data_path / "InferenceData.npy")}
calibration_data_file = data_path / "CalibrationData.npy"
shape = list(data["x"].shape)
trt_file = Path("model.trt")

@case_mark
def case_normal(b_int8_qat: bool = False):

    tw = TRTWrapperV1()
    parse_onnx((onnx_file_int8qat if b_int8_qat else onnx_file), tw.logger, tw.network, tw.builder_config)

    input_tensor = tw.network.get_input(0)
    tw.profile.set_shape(input_tensor.name, shape, [2] + shape[1:], [4] + shape[1:])

    tw.build()
    tw.serialize_engine(Path(str(trt_file) + ("-int8pat" if b_int8_qat else "")))

    tw.setup(data)
    tw.infer()
    return

if __name__ == "__main__":
    for pattern in ("*.trt*", ):
        for target_path in Path(".").glob(pattern):
            target_path.unlink(missing_ok=True)

    case_normal()

    case_normal(b_int8_qat=True)

    print("Finish")
