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
from tensorrt_cookbook import TRTWrapperV1, case_mark

onnx_file = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "model" / "model-trained-sparsity.onnx"
data = {"x": np.load(Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data" / "InferenceData.npy")}
shape = list(data["x"].shape)

@case_mark
def case_normal():
    tw = TRTWrapperV1(logger_level="VERBOSE")
    tw.config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

    parser = trt.OnnxParser(tw.network, tw.logger)
    with open(onnx_file, "rb") as model:
        parser.parse(model.read())

    input_tensor = tw.network.get_input(0)
    tw.profile.set_shape(input_tensor.name, shape, shape, shape)
    tw.config.add_optimization_profile(tw.profile)

    tw.build()

    return

if __name__ == "__main__":
    case_normal()

    print("Finish")
