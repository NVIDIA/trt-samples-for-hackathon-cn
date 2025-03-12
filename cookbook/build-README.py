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

from tensorrt_cookbook import build_readme

outline = \
"""
+ This repository is presented for NVIDIA TensorRT beginners and developers, which provides TensorRT-related learning and reference materials, as well as code examples.

+ Related materials (slices, datasets, models and PDF files): [Baidu Netdisk](https://pan.baidu.com/s/14HNCFbySLXndumicFPD-Ww) (Extraction code: gpq2)

+ Related video tutorial on Bilibili website:
  + [TensorRT-8.6.1](https://www.bilibili.com/video/BV1jj411Z7wG/)
  + [TensorRT-8.2.3](https://www.bilibili.com/video/BV15Y4y1W73E)
  + [Hackathon2022](https://www.bilibili.com/video/BV1i3411G7vN)
  + [Hackathon2023](https://www.bilibili.com/video/BV1dw411r7X8/)

+ Recommend order to read the subtopics if you are a freshman to the TensorRT:
  + 01-SimpleDemo/TensorRT10.0
  + 04-BuildEngineByONNXParser/pyTorch-ONNX-TensorRT
  + 07-Tool/Netron
  + 07-Tool/trtexec
  + 05-Plugin/BasicExample
  + 05-Plugin/UseONNXParserAndPlugin-pyTorch
  + ...

+ Chinese contents will be all translated into English in the future.

## Steps to setup

+ We recommend to use NVIDIA-optimized Docker to run the examples: [Steps to install](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
+ The information of the docker images can be found from [here](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
+ The packages of pyTorch and TensorFlow in the Docker is somewhere different from the version installed directly by `pip install`, especially quantization related features. So I recommend you use them inside the docker image rather than install by yourself.
+ Recommended docer images

|            Name of Docker Image             | python |    CUDA    | TensorRT  | Nsight-Systems | Lowest  Driver |            Comment             |
| :-----------------------------------------: | :----: | :--------: | :-------: | :------------: | :------------: | :----------------------------: |
|    **nvcr.io/nvidia/tensorrt:19.12-py3**    |  3.6   |  10.2.89   |   6.0.1   |    2019.6.1    |   440.33.01    |  Last version with TensorRT 6  |
|    **nvcr.io/nvidia/tensorrt:21.06-py3**    |  3.8   |   11.3.1   |  7.2.3.4  |  2021.2.1.58   |   465.19.01    |  Last version with TensorRT 7  |
|    **nvcr.io/nvidia/pytorch:23.02-py3**     |  3.8   |   12.0.1   |   8.5.3   |    2022.5.1    |      525       |  Last version with pyTorch 1   |
| **nvcr.io/nvidia/tensorflow:23.03-tf1-py3** |  3.8   |   12.1.0   |   8.5.3   |  2023.1.1.127  |      530       | Last version with TensorFlow 1 |
|    **nvcr.io/nvidia/pytorch:24.04-py3**     |  3.10  |   12.3.2   |  8.6.1.6  |  2023.4.1.97   |      545       | Last version with TensorRT 8.6 |
|    **nvcr.io/nvidia/pytorch:25.02-py3**     |  3.10  | 12.8.0.038 | 10.8.0.43 |  2025.1.1.65   |      570       |       **prefer version**       |

+ Start the container

```bash
docker run \
    -it -e NVIDIA_VISIBLE_DEVICES=0 --gpus "device=0" \
    --shm-size 16G --ulimit memlock=-1 --ulimit stack=67108864 \
    --name trt-cookbook \
    -v <PathToRepo>:/trtcookbook \
    nvcr.io/nvidia/pytorch:25.02-py3 \
    /bin/bash
```

+ Inside the container

```bash
cd <Path to the cookbook repo>
export TRT_COOKBOOK_PATH=$(pwd)  # NECESSARY!
pip install -r requirements.txt  # Add "-i https://pypi.tuna.tsinghua.edu.cn/simple" to accelerate downloading for Chinese users.

# Fore release usage:
rm -rf build dist
python3 setup.py bdist_wheel
pip install dist/*.whl

# For developer usage:
pip install -e .
```

+ \[Optional\] Prepare the dataset (following the steps in 00-Data/README.md) which some examples need.

+ Now it's OK to go through other directories and enjoy the examples.

## Important update of the repository

+ **1st November 2024**. Update to TensorRT 10.6, package `tensorrt_cookbook.whl` for installation.

+ **20th September 2024**. Update to TensorRT 10.5.

+ **1st May 2024**. Update to TensorRT 10.0 GA.
  + From branch TensorRT-10.0, we will discard several examples in older TensorRT versions.

+ **12th February 2024**. Update to TensorRT 9.X GA.

+ **18th June 2023**. Update to TensorRT 8.6 GA. Finish TensorRT tutorial (slice + audio) for Bilibili.

+ **17th March 2023**. Freeze code of branch TensorRT-8.5
  + Translate almost all contents into English (except 02-API/Layer/\*.md)
  + Come to development work of TensorRT 8.6 EA

+ **10th October 2022**. Update to TensorRT 8.5 GA. Cookbook with TensorRT 8.4 is remained in branch old/TensorRT-8.4.

+ **15th July 2022**. Update to TensorRT 8.4 GA. Cookbook with older version of TensorRT is remained in branch old/TensorRT-8.2.

## Useful Links

+ [**Download**](https://developer.nvidia.com/nvidia-tensorrt-download)
+ [TensorRT Open Source Software (TRTOSS on GitHub)](https://github.com/NVIDIA/TensorRT)
+ [Document Index](https://docs.nvidia.com/deeplearning/tensorrt/) and [Document Archives](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html)
+ [**Developer Guide**](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
+ [Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
+ [Supporting Matrix](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html) and [Supporting Matrix (old version)](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-843/support-matrix/index.html)
+ [**C++ API**](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api) and [**Python API**](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/) and [API Migration Guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1001/migration-guide/index.html)
+ [**Operators Document**](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/)

+ Provided by TensorRT document
  + [Install Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
  + [Quick Start Guide](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html)
  + [Sample Support Guide](https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html)
  + [Container Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html)
  + [ONNX GraphSurgeon API Reference](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html)
  + [Polygraphy document](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/polygraphy)
  + [Polygraphy API Reference](https://docs.nvidia.com/deeplearning/tensorrt/polygraphy/docs/index.html)
  + [PyTorch-Quantization Toolkit User Guide](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/index.html)
  + [TensorFlow Quantization Toolkit User Guide](https://docs.nvidia.com/deeplearning/tensorrt/tensorflow-quantization-toolkit/docs/index.html)

+ Others
  + [CUDA](https://developer.nvidia.com/cuda-zone)
  + [ONNX](https://github.com/onnx/onnx)
  + [ONNX Operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
  + [ONNX Model Zoo](https://github.com/onnx/models)
  + [ONNX Runtime](https://github.com/microsoft/onnxruntime)
  + [TREX](https://github.com/NVIDIA/TensorRT/tree/main/tools/experimental/trt-engine-explorer)
  + [tensorrtx (Network API building)](https://github.com/wang-xinyu/tensorrtx)
  + [TF-TRT](https://github.com/tensorflow/tensorrt)
  + [Torch-TensorRT](https://pytorch.org/TensorRT/)
"""

build_readme(__file__, outline)
