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

import numpy as np
import onnxruntime

np.random.seed(31193)
data = np.random.rand(1, 1, 28, 28).astype(np.float32) * 2 - 1
onnxFile = "modelA.onnx"

# Run the model in Onnx Runtime ------------------------------------------------
print("Onnxruntime using device: %s" % onnxruntime.get_device())
session = onnxruntime.InferenceSession(onnxFile, providers=["CPUExecutionProvider"])

for i, inputTensor in enumerate(session.get_inputs()):
    print("Input %2d: %s, %s, %s" % (i, inputTensor.name, inputTensor.shape, inputTensor.type))

for i, outputTensor in enumerate(session.get_outputs()):
    print("Output%2d: %s, %s, %s" % (i, outputTensor.name, outputTensor.shape, outputTensor.type))

inputDict = {}
inputDict[session.get_inputs()[0].name] = data

outputList = session.run(None, inputDict)

for i, outputTensor in enumerate(outputList):
    print("Output%2d:\n%s" % (i, outputTensor))

print("Succeeding running model in OnnxRuntime!")
