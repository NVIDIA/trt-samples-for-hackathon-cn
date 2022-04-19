#!/usr/bin/python3

#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#

import torch
import torchvision
import time
import ctypes

cudart = ctypes.CDLL('libcudart.so')

torch.manual_seed(0)

resnet50 = torchvision.models.resnet50().cuda()
resnet50.eval()

input_data = torch.randn(1, 3, 1080, 1920, dtype=torch.float32, device='cuda')

nRound = 10
cudart.cudaProfilerStart()
with torch.no_grad():
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(nRound):
        resnet50(input_data)
    torch.cuda.synchronize()
    time_pytorch = (time.time() - t0) / nRound
print('PyTorch time:', time_pytorch)
cudart.cudaProfilerStop()

