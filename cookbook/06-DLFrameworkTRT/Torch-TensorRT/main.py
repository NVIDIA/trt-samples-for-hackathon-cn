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
import torch as t
import torch.nn.functional as F
import torch_tensorrt

model_file = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "model" / "model-trained.pth"
data_file = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data" / "InferenceData.npy"
ts_model_file = "model.ts"
shape = [1, 1, 28, 28]

class Net(t.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = t.nn.Conv2d(1, 32, (5, 5), padding=(2, 2), bias=True)
        self.conv2 = t.nn.Conv2d(32, 64, (5, 5), padding=(2, 2), bias=True)
        self.gemm1 = t.nn.Linear(64 * 7 * 7, 1024, bias=True)
        self.gemm2 = t.nn.Linear(1024, 10, bias=True)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.reshape(-1, 64 * 7 * 7)
        x = F.relu(self.gemm1(x))
        y = self.gemm2(x)
        z = F.softmax(y, dim=1)
        z = t.argmax(z, dim=1)
        return y, z

model = t.load(model_file, weights_only=False)
data = np.load(data_file)

ts_model = t.jit.trace(model, t.randn(*shape, device="cuda"))  # torch script model
trt_model = torch_tensorrt.compile(
    ts_model,
    inputs=[t.randn(*shape, device="cuda")],
    enabled_precisions={t.float},
    truncate_long_and_double=True,
)

input_data = t.from_numpy(data).cuda()
output_data = trt_model(input_data)  # run inference in TensorRT
print(output_data)

t.jit.save(trt_model, ts_model_file)  # save TRT embedded Torchscript as .ts file

print("Finish")
