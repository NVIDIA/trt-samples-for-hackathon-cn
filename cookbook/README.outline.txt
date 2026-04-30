+ **This README.md is automatically generated from `build-Copyright-and-README.py`, changes should be done in `README.outline.txt`.**

<p align="center">
<img src="TensorRTCookbook-ico.png" width="200px" height="200px" alt="description">
</p>

+ This repository is presented for NVIDIA TensorRT beginners and developers, which provides TensorRT-related learning and reference materials, as well as code examples.

+ Recommend order to read the subtopics if you are a freshman to the TensorRT:
  + 01-SimpleDemo/TensorRT10
  + 03-Workflow/pyTorch-ONNX-TensorRT
  + 07-Tool/Netron
  + 07-Tool/trtexec
  + 05-Plugin/BasicExample
  + 05-Plugin/ONNXParserWithPlugin
  + ...

## Steps to setup

+ We recommend to use NVIDIA-optimized Docker to run the examples: [Steps to install](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
+ The information of the docker images can be found from [here](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
+ The packages of pyTorch and TensorFlow in the Docker is somewhere different from the version installed directly by `pip install`, especially quantization related features. So I recommend you use them inside the docker image rather than install by yourself.
+ Recommended docker images

|            Name of Docker Image             | python |  Driver   |    CUDA    |  TensorRT  | Nsight-Systems |            Comment             |
| :-----------------------------------------: | :----: | :-------: | :--------: | :--------: | :------------: | :----------------------------: |
|    **nvcr.io/nvidia/tensorrt:19.12-py3**    |  3.6   | 440.33.01 |  10.2.89   |   6.0.1    |    2019.6.1    |  Last version with TensorRT 6  |
|    **nvcr.io/nvidia/tensorrt:21.06-py3**    |  3.8   | 465.19.01 |   11.3.1   |  7.2.3.4   |  2021.2.1.58   |  Last version with TensorRT 7  |
|    **nvcr.io/nvidia/pytorch:23.02-py3**     |  3.8   |    525    |   12.0.1   |   8.5.3    |    2022.5.1    |  Last version with pyTorch 1   |
| **nvcr.io/nvidia/tensorflow:23.03-tf1-py3** |  3.8   |    530    |   12.1.0   |   8.5.3    |  2023.1.1.127  | Last version with TensorFlow 1 |
|    **nvcr.io/nvidia/pytorch:24.04-py3**     |  3.10  |    545    |   12.3.2   |  8.6.1.6   |  2023.4.1.97   |  Last version with TensorRT 8  |
|    **nvcr.io/nvidia/pytorch:25.06-py3**     |  3.12  |     /     | 12.9.1.010 | 10.11.0.33 |  2025.5.1.121  |   Last version with CUDA 12    |
|    **nvcr.io/nvidia/pytorch:26.05-py3**     |  3.12  |     /     |   13.2.1   | 10.16.1.11 |  2026.2.1.210  | Last version with TensorRT 10  |
|    **nvcr.io/nvidia/pytorch:26.06-py3**     |  3.12  |     /     |   13.3.0   | 11.0.0.114 |       ?        |       **prefer version**       |

+ Start the container

```bash
docker run \
    -it -e NVIDIA_VISIBLE_DEVICES=0 --gpus "device=0" \
    --shm-size 16G --ulimit memlock=-1 --ulimit stack=67108864 \
    --name trt-cookbook \
    -v <PathToTheCookbookRepo>:/trtcookbook \
    nvcr.io/nvidia/pytorch:26.03-py3 \
    /bin/bash
```

+ Inside the container

```bash
cd <PathToTheCookbookRepo>
export TRT_COOKBOOK_PATH=$(pwd)  # Optional: package now auto-discovers this path in most cases.
pip install -r requirements.txt  # Add "-i https://pypi.tuna.tsinghua.edu.cn/simple" to accelerate downloading for Chinese users.

# For release usage:
python -m pip install -U build
rm -rf build dist
python -m build
pip install dist/*.whl

# For developer usage:
pip install -e .
```

+ [Optional] Prepare the dataset (following the steps in 00-Data/README.md) which some examples need.

+ Now it's OK to go through other directories and enjoy the examples.

+ Related materials (slices, datasets, models and PDF files): [Baidu Netdisk](https://pan.baidu.com/s/14HNCFbySLXndumicFPD-Ww?pwd=gpq2).

+ Related video tutorial on Bilibili website:
  + [TensorRT-8.6.1](https://www.bilibili.com/video/BV1jj411Z7wG/)
  + [TensorRT-8.2.3](https://www.bilibili.com/video/BV15Y4y1W73E)
  + [Hackathon2022](https://www.bilibili.com/video/BV1i3411G7vN)
  + [Hackathon2023](https://www.bilibili.com/video/BV1dw411r7X8/)

## Important update of the repository

+ **1st April 2026**. Release of tensorrt-cookbook v1.0.0-trt-10.16. Update to TensorRT 10.16.

+ **20th November 2025**. Update to TensorRT 10.13, APIs of `cuda-python` change in new version.

+ **18th June 2025**. Update to TensorRT 10.12, much stuff is deprecated.

+ **1st November 2024**. Update to TensorRT 10.6, package `tensorrt_cookbook.whl` for installation.

+ **20th September 2024**. Update to TensorRT 10.5.

+ **1st May 2024**. Update to TensorRT 10.0 GA.
  + From branch TensorRT-10.0, we will discard several examples in older TensorRT versions.

+ **12th February 2024**. Update to TensorRT 9.X GA.

+ **18th June 2023**. Update to TensorRT 8.6 GA. Finish TensorRT tutorial (slice + audio) for Bilibili.

+ **17th March 2023**. Freeze code of branch TensorRT-8.5
  + Translate almost all contents into English.
  + Come to development work of TensorRT 8.6 EA

+ **10th October 2022**. Update to TensorRT 8.5 GA. Cookbook with TensorRT 8.4 is remained in branch old/TensorRT-8.4.

+ **15th July 2022**. Update to TensorRT 8.4 GA. Cookbook with older version of TensorRT is remained in branch old/TensorRT-8.2.

## Useful Links

+ [Git address of this repo](https://github.com/wili-65535/trt-samples-for-hackathon-cn.git)
+ [**TensorRT Download**](https://developer.nvidia.com/nvidia-tensorrt-download)
+ [TensorRT Open Source Software (TRTOSS) on GitHub](https://github.com/NVIDIA/TensorRT)
+ [Document](https://docs.nvidia.com/deeplearning/tensorrt/)
+ [C++ API](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/index.html)
+ [Python API](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/index.html)
+ [Operators Document](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/)

+ Others
  + [CUDA](https://developer.nvidia.com/cuda-zone)
  + [ONNX Model Zoo](https://github.com/onnx/models)
  + [ONNX Operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
  + [ONNX Runtime](https://github.com/microsoft/onnxruntime)
  + [ONNX](https://github.com/onnx/onnx)
  + [Polygraphy API Reference](https://docs.nvidia.com/deeplearning/tensorrt/polygraphy/docs/index.html)
  + [Polygraphy document](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/polygraphy)
  + [tensorrtx (Network API building)](https://github.com/wang-xinyu/tensorrtx)
  + [TF-TRT](https://github.com/tensorflow/tensorrt)
  + [Torch-TensorRT](https://pytorch.org/TensorRT/)
  + [TREX](https://github.com/NVIDIA/TensorRT/tree/main/tools/experimental/trt-engine-explorer)
