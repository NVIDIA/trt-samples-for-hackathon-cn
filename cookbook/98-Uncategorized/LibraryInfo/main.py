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
import re

info = os.popen("nvidia-smi").read().split("\n")[2]
v_driver = re.search(r"Driver Version: \d+\.\d+(\.\d+)?", info)
v_cuda = re.search(r"CUDA Version: \d+\.\d+(\.\d+)?", info)
v_driver = "None" if v_driver is None else v_driver.group().split(": ")[-1]
v_cuda = "None" if v_cuda is None else v_cuda.group().split(": ")[-1]
print(f"Driver:          {v_driver}")
print(f"CUDA:            {v_cuda}")

info = os.popen(r"cat /usr/include/x86_64-linux-gnu/cudnn_version_v*.h").read()
v_major = re.search(r"CUDNN_MAJOR \d+", info)
v_minor = re.search(r"CUDNN_MINOR \d+", info)
v_patch = re.search(r"CUDNN_PATCHLEVEL \d+", info)
v_major = "None" if v_major is None else v_major.group().split(" ")[-1]
v_minor = "None" if v_minor is None else v_minor.group().split(" ")[-1]
v_patch = "None" if v_patch is None else v_patch.group().split(" ")[-1]
print(f"cuDNN:           {'.'.join([v_major, v_minor, v_patch])}")

info = os.popen(r"cat /usr/local/cuda/include/cublas_api.h").read()
v_major = re.search(r"CUBLAS_VER_MAJOR \d+", info)
v_minor = re.search(r"CUBLAS_VER_MINOR \d+", info)
v_patch = re.search(r"CUBLAS_VER_PATCH \d+", info)
v_build = re.search(r"CUBLAS_VER_BUILD \d+", info)
v_major = "None" if v_major is None else v_major.group().split(" ")[-1]
v_minor = "None" if v_minor is None else v_minor.group().split(" ")[-1]
v_patch = "None" if v_patch is None else v_patch.group().split(" ")[-1]
v_build = "None" if v_build is None else v_build.group().split(" ")[-1]
print(f"cuBLAS:          {'.'.join([v_major, v_minor, v_patch, v_build])}")

info = os.popen(r"cat /usr/include/x86_64-linux-gnu/NvInferVersion.h").read()
v_major = re.search(r"NV_TENSORRT_MAJOR \d+", info)
v_minor = re.search(r"NV_TENSORRT_MINOR \d+", info)
v_patch = re.search(r"NV_TENSORRT_PATCH \d+", info)
v_build = re.search(r"NV_TENSORRT_BUILD \d+", info)
v_major = "None" if v_major is None else v_major.group().split(" ")[-1]
v_minor = "None" if v_minor is None else v_minor.group().split(" ")[-1]
v_patch = "None" if v_patch is None else v_patch.group().split(" ")[-1]
v_build = "None" if v_build is None else v_build.group().split(" ")[-1]
print(f"TensorRT:        {'.'.join([v_major, v_minor, v_patch, v_build])}")

info = os.popen(r"pip list").read()
v_pytorch = re.search(r"torch .+", info)
v_pytorch = "None" if v_pytorch is None else v_pytorch.group().split(" ")[-1]
v_tf = re.search(r"tensorflow .+", info)
v_tf = "None" if v_tf is None else v_tf.group().split(" ")[-1]
v_trt_py = re.search(r"tensorrt .+", info)
v_trt_py = "None" if v_trt_py is None else v_trt_py.group().split(" ")[-1]
print(f"TensorRT(python):{v_trt_py}")
print(f"pyTorch:         {v_pytorch}")
print(f"TensorFlow:      {v_tf}")
