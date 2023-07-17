#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

import os

os.chdir("/w/gitlab/tensorrt-cookbook/08-Tool/FP16FineTuning")

import numpy as np
import onnx
import onnxruntime

onnxFile = "model.onnx"

np.random.seed(31193)
shape = [2,1,28,28]

print("Onnxruntime using device: %s" % onnxruntime.get_device())
session = onnxruntime.InferenceSession(onnxFile)

ioData = {}

for i, inputTensor in enumerate(session.get_inputs()):
    print("Input %2d: %s, %s, %s" % (i, inputTensor.name, inputTensor.shape, inputTensor.type))
    if inputTensor.type == "tensor(float)":
        dataType = np.float32
    if inputTensor.type == "tensor(int32)":
        dataType = np.int32
    data = np.random.rand(np.prod(shape)).astype(dataType).reshape(shape)
    ioData[inputTensor.name] = data

for i, outputTensor in enumerate(session.get_outputs()):
    print("Output%2d: %s, %s, %s" % (i, outputTensor.name, outputTensor.shape, outputTensor.type))

outputList = session.run(None, ioData)

for i, outputTensor in enumerate(session.get_outputs()):
    print(outputList[i])
    ioData[outputTensor.name] = outputList[i]

np.savez("IOData.npz", **ioData)
