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

from cuda import cudart
import cv2
from datetime import datetime as dt
from glob import glob
import numpy as np
import os
import tensorrt as trt
import torch as t
import torch.nn.functional as F
from torch.autograd import Variable

import calibrator

np.random.seed(31193)
t.manual_seed(97)
t.cuda.manual_seed_all(97)
t.backends.cudnn.deterministic = True
nTrainBatchSize = 128
nHeight = 28
nWidth = 28
#ptFile = "./model.pt"
onnxFile1 = "./encoderV3.onnx"
trtFile = "./model.plan"
dataPath = os.path.dirname(os.path.realpath(__file__)) + "/../../00-MNISTData/"
trainFileList = sorted(glob(dataPath + "train/*.jpg"))
testFileList = sorted(glob(dataPath + "test/*.jpg"))
inferenceImage = dataPath + "8.png"

# for FP16 mode
bUseFP16Mode = False
# for INT8 model
bUseINT8Mode = False
nCalibration = 1
cacheFile = "./int8.cache"
calibrationDataPath = dataPath + "test/"

#os.system("rm -rf ./*.onnx ./*.plan ./*.cache")
np.set_printoptions(precision=3, linewidth=100, suppress=True)
cudart.cudaDeviceSynchronize()

# TensorRT 中加载 .onnx 创建 engine ----------------------------------------------
logger = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
if bUseFP16Mode:
    config.set_flag(trt.BuilderFlag.FP16)
if bUseINT8Mode:
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, nCalibration, (1, 1, nHeight, nWidth), cacheFile)

parser = trt.OnnxParser(network, logger)
if not os.path.exists(onnxFile1):
    print("Failed finding ONNX file!")
    exit()
print("Succeeded finding ONNX file!")
with open(onnxFile1, "rb") as model:
    if not parser.parse(model.read()):
        print("Failed parsing .onnx file!")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
    print("Succeeded parsing .onnx file!")

inputT0 = network.get_input(0)
inputT0.shape = [-1, -1, 80]
profile.set_shape(inputT0.name, [1, 16, 80], [4, 64, 80], [16, 256, 80])
inputT1 = network.get_input(1)
inputT1.shape = [-1]
profile.set_shape(inputT1.name, [
    1,
], [
    4,
], [
    16,
])

config.add_optimization_profile(profile)

#-------------------------------------------------------------------------------
print("Succeeded building network!")
#del engineString, engine, context
from NetworkInspector import inspectNetwork
from NetworkRebuilder import rebuildNetwork

if "profile" not in locals().keys():
    inspectNetwork(builder, config, network)
else:
    inspectNetwork(builder, config, network, [profile])  # seems ugly if we can not get optimization profile from BuilderConfig

engineString = rebuildNetwork(logger)
with open(trtFile, "wb") as f:
    f.write(engineString)
    print("Succeeded saving .plan file!")

engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
