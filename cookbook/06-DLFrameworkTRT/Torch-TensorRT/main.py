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
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_tensorrt

MODEL_FILE = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "model" / "model-trained.pth"
DATA_FILE = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data" / "InferenceData.npy"
INPUT_SHAPE = [1, 1, 28, 28]
WARMUP = 50
STEPS = 200

class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, (5, 5), padding=(2, 2), bias=True)
        self.conv2 = torch.nn.Conv2d(32, 64, (5, 5), padding=(2, 2), bias=True)
        self.gemm1 = torch.nn.Linear(64 * 7 * 7, 1024, bias=True)
        self.gemm2 = torch.nn.Linear(1024, 10, bias=True)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.reshape(-1, 64 * 7 * 7)
        x = F.relu(self.gemm1(x))
        y = self.gemm2(x)
        z = F.softmax(y, dim=1)
        z = torch.argmax(z, dim=1)
        return y, z

def benchmark(model, input_data: torch.Tensor, warmup: int = WARMUP, steps: int = STEPS) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(input_data)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(steps):
            _ = model(input_data)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / steps

if __name__ == "__main__":

    model = torch.load(MODEL_FILE, map_location="cuda", weights_only=False).eval()
    input_data = torch.from_numpy(np.load(DATA_FILE)).to(device="cuda", dtype=torch.float32)

    trt_model = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=[torch_tensorrt.Input(shape=INPUT_SHAPE, dtype=torch.float32)],
        enabled_precisions={torch.float32},
        truncate_long_and_double=True,
    )

    torch_compile_model = torch.compile(model, mode="max-autotune")

    with torch.inference_mode():
        trt_logits, trt_pred = trt_model(input_data)
        compile_logits, compile_pred = torch_compile_model(input_data)

    torch.testing.assert_close(trt_logits, compile_logits, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(trt_pred, compile_pred, rtol=0, atol=0)

    trt_latency_ms = benchmark(trt_model, input_data)
    compile_latency_ms = benchmark(torch_compile_model, input_data)

    print(f"Torch-TensorRT latency: {trt_latency_ms:.3f} ms")
    print(f"torch.compile latency: {compile_latency_ms:.3f} ms")

    print("Finish")
