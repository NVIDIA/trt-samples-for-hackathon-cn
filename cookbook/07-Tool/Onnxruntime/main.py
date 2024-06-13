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

from pathlib import Path

import numpy as np
import onnxruntime

onnx_file = Path("/trtcookbook/00-Data/model/model-trained.onnx")
data = {"x": np.load(Path("/trtcookbook/00-Data/data/InferenceData.npy"))}

print(f"Device: {onnxruntime.get_device()}")
session = onnxruntime.InferenceSession(onnx_file)

for i, tensor in enumerate(session.get_inputs()):
    print(f"Input {i:2d}: {tensor.name}, {tensor.shape}, {tensor.type}")

for i, tensor in enumerate(session.get_outputs()):
    print(f"Output{i:2d}: {tensor.name}, {tensor.shape}, {tensor.type}")

feed_dict = {}
for i, tensor in enumerate(session.get_inputs()):
    name = session.get_inputs()[i].name
    feed_dict[name] = data[name]

output_list = session.run(None, feed_dict)

for i, tensor in enumerate(output_list):
    print(f"Output{i:2d}:\n{tensor}")

print("Finish")
