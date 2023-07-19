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
import re

information = os.popen("nvidia-smi").read().split("\n")[2]
driverV = re.search(r"Driver Version: \d+\.\d+(\.\d+)?", information)
cudaV = re.search(r"CUDA Version: \d+\.\d+(\.\d+)?", information)
driverV = "None" if driverV is None else driverV.group().split(": ")[-1]
cudaV = "None" if cudaV is None else cudaV.group().split(": ")[-1]
print("Driver:          %s" % driverV)
print("CUDA:            %s" % cudaV)

information = os.popen(r"cat /usr/include/x86_64-linux-gnu/cudnn_version_v*.h").read()
cudnnMajorV = re.search(r"CUDNN_MAJOR \d+", information)
cudnnMinorV = re.search(r"CUDNN_MINOR \d+", information)
cudnnPatchV = re.search(r"CUDNN_PATCHLEVEL \d+", information)
cudnnMajorV = "None" if cudnnMajorV is None else cudnnMajorV.group().split(" ")[-1]
cudnnMinorV = "None" if cudnnMinorV is None else cudnnMinorV.group().split(" ")[-1]
cudnnPatchV = "None" if cudnnPatchV is None else cudnnPatchV.group().split(" ")[-1]
print("cuDNN:           %s" % cudnnMajorV + "." + cudnnMinorV + "." + cudnnPatchV)

information = os.popen(r"cat /usr/local/cuda/include/cublas_api.h").read()
cublasMajorV = re.search(r"CUBLAS_VER_MAJOR \d+", information)
cublasMinorV = re.search(r"CUBLAS_VER_MINOR \d+", information)
cublasPatchV = re.search(r"CUBLAS_VER_PATCH \d+", information)
cublasBuildV = re.search(r"CUBLAS_VER_BUILD \d+", information)
cublasMajorV = "None" if cublasMajorV is None else cublasMajorV.group().split(" ")[-1]
cublasMinorV = "None" if cublasMinorV is None else cublasMinorV.group().split(" ")[-1]
cublasPatchV = "None" if cublasPatchV is None else cublasPatchV.group().split(" ")[-1]
cublasBuildV = "None" if cublasBuildV is None else cublasBuildV.group().split(" ")[-1]
print("cuBLAS:          %s" % cublasMajorV + "." + cublasMinorV + "." + cublasPatchV + "." + cublasBuildV)

information = os.popen(r"cat /usr/include/x86_64-linux-gnu/NvInferVersion.h").read()
tensorrtMajorV = re.search(r"NV_TENSORRT_MAJOR \d+", information)
tensorrtMinorV = re.search(r"NV_TENSORRT_MINOR \d+", information)
tensorrtPatchV = re.search(r"NV_TENSORRT_PATCH \d+", information)
tensorrtBuildV = re.search(r"NV_TENSORRT_BUILD \d+", information)
tensorrtMajorV = "None" if tensorrtMajorV is None else tensorrtMajorV.group().split(" ")[-1]
tensorrtMinorV = "None" if tensorrtMinorV is None else tensorrtMinorV.group().split(" ")[-1]
tensorrtPatchV = "None" if tensorrtPatchV is None else tensorrtPatchV.group().split(" ")[-1]
tensorrtBuildV = "None" if tensorrtBuildV is None else tensorrtBuildV.group().split(" ")[-1]
print("TensorRT:        %s" % tensorrtMajorV + "." + tensorrtMinorV + "." + tensorrtPatchV + "." + tensorrtBuildV)

information = os.popen(r"pip list").read()
pyTorchV = re.search(r"torch .+", information)
pyTorchV = "None" if pyTorchV is None else pyTorchV.group().split(" ")[-1]
tensorflowV = re.search(r"tensorflow .+", information)
tensorflowV = "None" if tensorflowV is None else tensorflowV.group().split(" ")[-1]
tensorrtV = re.search(r"tensorrt .+", information)
tensorrtV = "None" if tensorrtV is None else tensorrtV.group().split(" ")[-1]
print("pyTorch:         %s" % pyTorchV)
print("TensorFlow:      %s" % tensorflowV)
print("TensorRT(python):%s" % tensorrtV)
