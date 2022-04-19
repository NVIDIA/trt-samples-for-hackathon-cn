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
from torchsummary import summary
import time

torch.manual_seed(0)

resnet50 = torchvision.models.resnet50().cuda()
resnet50.eval()
#summary(resnet50, (3, 1080, 1920), device='cuda')

input_data = torch.randn(1, 3, 1080, 1920, dtype=torch.float32, device='cuda')

output_data_pytorch = resnet50(input_data).cpu().detach().numpy()

nRound = 10
with torch.no_grad():
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(nRound):
        resnet50(input_data)
    torch.cuda.synchronize()
    time_pytorch = (time.time() - t0) / nRound
print('PyTorch time:', time_pytorch)

input_names = ['input']
output_names = ['output']
torch.onnx.export(resnet50, input_data, 'resnet50.onnx', input_names=input_names, output_names=output_names, verbose=False, opset_version=11)
torch.onnx.export(resnet50, input_data, 'resnet50.dynamic_shape.onnx', dynamic_axes={"input": [0, 2, 3]}, input_names=input_names, output_names=output_names, verbose=False, opset_version=11)

#继续运行python代码前，先运行如下命令
#trtexec --verbose --onnx=resnet50.onnx --saveEngine=resnet50.trt
#trtexec --verbose --onnx=resnet50.onnx --saveEngine=resnet50_fp16.trt --fp16
#以下命令不必运行，仅供参考
#trtexec --verbose --onnx=resnet50.dynamic_shape.onnx --saveEngine=resnet50.dynamic_shape.trt --optShapes=input:1x3x1080x1920 --minShapes=input:1x3x1080x1920 --maxShapes=input:1x3x1080x1920

from trt_lite2 import TrtLite
import numpy as np
import os

for engine_file_path in ['resnet50.trt', 'resnet50_fp16.trt']:
    if not os.path.exists(engine_file_path):
        print('Engine file', engine_file_path, 'doesn\'t exist. Please run trtexec and re-run this script.')
        exit(1)
    
    print('====', engine_file_path, '===')
    trt = TrtLite(engine_file_path=engine_file_path)
    trt.print_info()
    i2shape = {0: (1, 3, 1080, 1920)}
    io_info = trt.get_io_info(i2shape)
    d_buffers = trt.allocate_io_buffers(i2shape, True)
    output_data_trt = np.zeros(io_info[1][2], dtype=np.float32)

    d_buffers[0].copy_(input_data.reshape(d_buffers[0].size()))
    trt.execute([t.data_ptr() for t in d_buffers], i2shape)
    output_data_trt = d_buffers[1].cpu().numpy()

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(nRound):
        trt.execute([t.data_ptr() for t in d_buffers], i2shape)
    torch.cuda.synchronize()
    time_trt = (time.time() - t0) / nRound
    print('TensorRT time:', time_trt)

    print('Speedup:', time_pytorch / time_trt)
    print('Average diff percentage:', np.mean(np.abs(output_data_pytorch - output_data_trt) / np.abs(output_data_pytorch)))
