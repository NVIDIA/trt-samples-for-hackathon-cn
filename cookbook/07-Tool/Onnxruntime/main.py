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

import os
from pathlib import Path

import numpy as np
import onnxruntime

np.random.seed(31193)

onnx_file = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "model" / "model-trained.onnx"
data_path = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data" / "InferenceData.npy"
data = {"x": np.load(data_path)}

def case_normal():
    print(f"Device: {onnxruntime.get_device()}")
    session = onnxruntime.InferenceSession(onnx_file, providers=["CUDAExecutionProvider"])

    for i, tensor in enumerate(session.get_inputs()):
        print(f"Input {i:2d}: {tensor.name}, {tensor.shape}, {tensor.type}")

    for i, tensor in enumerate(session.get_outputs()):
        print(f"Output{i:2d}: {tensor.name}, {tensor.shape}, {tensor.type}")

    output_name_list = ["y", "z"]
    output_list = session.run(output_name_list, data)

    print("Output:")
    for name, tensor in output_name_list, output_list:
        print(name, "\n", tensor)

    return

if __name__ == "__main__":
    case_normal()

    print("Finish")
