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

resnet50 = torchvision.models.resnet50()
summary(resnet50, (3, 128, 128), device='cpu')

dummy_input = torch.randn(1, 3, 128, 128, device='cpu')
input_names = ['input']
output_names = ['output']
torch.onnx.export(resnet50, dummy_input, 'resnet50.onnx', input_names=input_names, output_names=output_names, verbose=True, opset_version=11)
torch.onnx.export(resnet50, dummy_input, 'resnet50.dynamic_shape.onnx', dynamic_axes={"input": [0, 2, 3]}, input_names=input_names, output_names=output_names, verbose=True, opset_version=11)

#trtexec --verbose --onnx=resnet50.onnx --saveEngine=resnet50.trt
#trtexec --verbose --onnx=resnet50.dynamic_shape.onnx --saveEngine=resnet50.dynamic_shape.trt --optShapes=input:1x3x128x128 --minShapes=input:1x3x128x128 --maxShapes=input:128x3x128x128
