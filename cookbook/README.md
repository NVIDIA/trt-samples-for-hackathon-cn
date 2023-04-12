# TensorRT Cookbook

## General Introduction

+ This repository is presented for NVIDIA TensorRT beginners and developers, which provides TensorRT-related learning and reference materials, as well as code examples.

+ This README.md contains catalogue of the cookbook, you can search your interested subtopics and go to the corresponding directory to read.

+ Recommend order to read (and try) the subtopics if you are freshman to the TensorRT:
  + 01-SimpleDemo/TensorRT8.5
  + 04-Parser/pyTorch-ONNX-TensorRT
  + 08-Tool/Netron
  + 08-Tool/trtexec
  + 05-Plugin/UsePluginV2DynamicExt
  + 06-PluginAndParser/pyTorch-AddScalar
  + ...

+ Steps to setup
  + We recommend to use NVIDIA-optimized Docker to run the examples. [Steps to install](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

```shell
# start the container
docker run -it -e NVIDIA_VISIBLE_DEVICES=0 --gpus "device=0" --name trt-cookbook \
--shm-size 32G --ulimit memlock=-1 --ulimit stack=67108864 \
-v ~:/work \
nvcr.io/nvidia/pytorch:23.03-py3 /bin/bas

# inside thecontainer
# go to directory of cookbook firstly
pip3 install -r requirement.txt

# prepare the dataset we may need in some examples
# download MNIST dataset (see the "00-MNISTData -- Related dataset" part below)
cd 00-MNISTData
python3 extractMnistData.py

# now we can go to other directory to try the examples we are interested and run the examples
```

+ Table of tested docker images. Note that pyTorch and TensorFlow1 attached in the following docker images contain some changes by NVIDIA, which is different from the version installed with *pip*.

|            Name of Docker Image             | python |  CUDA   |  cuDNN   | TensorRT |     Framework      |           Comment           |
| :-----------------------------------------: | :----: | :-----: | :------: | :------: | :----------------: | :-------------------------: |
|    **nvcr.io/nvidia/tensorrt:19.12-py3**    | 3.6.9  | 10.2.89 |  7.6.5   |  6.0.1   |     TensorRT 6     | Last version of TensorRT 6  |
|    **nvcr.io/nvidia/tensorrt:21.06-py3**    | 3.8.5  | 11.3.1  |  8.2.1   | 7.2.3.4  |     TensorRT 7     | Last version of TensorRT 7  |
|    **nvcr.io/nvidia/tensorrt:22.04-py3**    | 3.8.10 | 11.6.2  | 8.4.0.27 | 8.2.4.2  |   TensorRT 8.2.5   | Last version of TensorRT8.2 |
|    **nvcr.io/nvidia/tensorrt:22.08-py3**    | 3.8.13 | 11.7U1  | 8.5.0.96 |  8.4.1   |  TensorRT 8.4.2.4  | Last version of TensorRT8.4 |
|    **nvcr.io/nvidia/tensorrt:23.01-py3**    | 3.8.10 | 12.0.1  |  8.7.0   | 8.5.0.12 |  TensorRT 8.5.2.2  |         TensorRT8.5         |
| **nvcr.io/nvidia/tensorflow:23.01-tf1-py3** | 3.8.10 | 12.0.1  |  8.7.0   | 8.5.0.12 | TensorFlow 1.15.5  |      TensorFlow 1 LTS       |
| **nvcr.io/nvidia/tensorflow:23.01-tf2-py3** | 3.8.10 | 12.0.1  |  8.7.0   | 8.5.0.12 | TensorFlow 2.11.0  |                             |
|    **nvcr.io/nvidia/pytorch:23.01-py3**     | 3.8.10 | 12.0.1  |  8.7.0   | 8.5.0.12 | pyTorch 1.14.0a0+  |                             |
|  **nvcr.io/nvidia/paddlepaddle:23.01-py3**  | 3.8.10 | 12.0.1  |  8.7.0   | 8.5.0.12 | PaddlePaddle 2.3.2 |                             |

+ Important update of the repository

  + Chinese contents will be translated into English in the future ([\u4E00-\u9FA5]+)

  + **17th March 2023**. Freeze code of branch TensorRT-8.5
    + Translate almost all contents into English (except 02-API/Layer/\*.md)
    + Come to development of TensorRT 8.6 EA

  + **10th October 2022**. Updated to TensorRT 8.5 GA. Cookbook with TensorRT 8.4 is remained in branch old/TensorRT8.4. Using the older version of TensorRT to run the examples may need to modify some of the code, for example:
    + Modify `context.set_input_shape` back to `context.set_binding_shape`, etc.

  + **15th July 2022** Updated to TensorRT 8.4 GA. Cookbook with older version of TensorRT is remained in branch old/TensorRT\*. Using the older version of TensorRT to run the examples may need to modify some of the code, for example:
    + Modify `config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)` back to `config.max_workspace_size = 1 << 30`.

+ Useful Links
  + [Download](https://developer.nvidia.com/nvidia-tensorrt-download)
  + [Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
  + [Document](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
  + [Document Archives](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html)
  + [C++ API Document](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api)
  + [Python API Document](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)
  + [API Change](https://docs.nvidia.com/deeplearning/tensorrt/api/index.html)
  + [Operator algorithm](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/)
  + [Onnx Operator Supporting Matrix](https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md)
  + [Supporting Matrix](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html)
  + [Supporting Matrix (Old version)](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-843/support-matrix/index.html)

  + [ONNX model zoo](https://github.com/onnx/models)
  + [ONNX-TensorRT](https://github.com/onnx/onnx-tensorrt)
  + [TF-TRT Guide](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html)
  + [NSight- Systems](https://developer.nvidia.com/nsight-systems)
  + [DL Prof](DLProf)
  + [TensorRT Open Source Software](https://github.com/NVIDIA/TensorRT)


---

## Catalogue and introduction of the files

+ **clean.sh**: script to clean all files produced by the example codes in the cookbook.

+ **cloc.txt**: word statistics of the cookbook.

+ **testAll.sh**: run all the example code (certain ML framework may be needed in some examples) **\[Not finished\]**.

## include

+ Common files used by some examples in the cookobook.

+ **Makefile.inc**: common part of Makefile used by the cookbook, especially the compute capability and the path to the CUDA / TensorRT).

+ **cookbookHelper.cuh**: common head file used by the C++ example code in the cookbook, including classes (such as Logger) and functions (for error check, print array, print information, convert datatype and debug).

## 00-MNISTData -- Related dataset

+ The MNIST dataset used by Cookbook, which needs to be downloaded and preprocessed before running other example code

+ We can get the dataset from [Link](http://yann.lecun.com/exdb/mnist/) or [Link](https://storage.googleapis.com/cvdf-datasets/mnist/) or [Baidu Netdisk](https://pan.baidu.com/s/14HNCFbySLXndumicFPD-Ww)(Extraction code: gpq2)

+ The dataset should be put into this directory as *00-MNISTData/\*.gz*, 4 files in total.

+ Run the following command to extract XXX training pictures to./train and YYY pictures to./test (there are 60000 training pictures and 10000 test pictures in total. If no parameter provided, 3000 training pictures and 500 test pictures are extracted as JPEG format by default.

```shell
python3 extractMnistData.py XXX YYY
```

+ There is also an *8.png* from test set of the dataset in this directory, which is used as the input data of TensorRT

---

## 01-SimpleDemo -- Examples of a complete program using TensorRT

+ Basic steps to use TensorRT, including network construction, engine construction, serialization, deserialization, inference computation, etc.

+ Examples cover different versions of TensorRT, all with equivalent implementations in C++ and python

### TensorRT6

+ **TensorRT6** + **Implicit Batch** mode + **Static Shape** mode + **Builder API** + **pycuda** library.

+ **Implicit Batch** mode is only for the backward compatibility, and may lack support for new features and performance optimization. Please try to use Explicit Batch mode in newer version of TensorRT.

+ **Builder API** has been deprecated, please use BuilderConfig API in newer versions of TensorRT.

+ **pycuda** library is relatively old, and there may be some problems of thread safety and interaction with CUDA Context in other machine learning frameworks. It has been deprecated. Please use cuda-python library in newer version of TensorRT.

### TensorRT7

+ **TensorRT7** + **Implicit Batch** mode + **Static Shape** mode + **Builder API** + Runtime API of **cuda-python** library.

### TensorRT8

+ **TensorRT8** + **Explicit Batch** mode + **Dynamic Shape** mode + **BuilderConfig API** + Driver API / Runtime API of **cuda-python** library.

### TensorRT8.5

+ Based on TensorRT8, add new APIs in TensorRT8.5, including set_memory_pool_limit and APIs of I/O Tensor (Binding), only CUDA runtime API of cuda-python library used.

---

## 02-API -- Examples of TesnorRT core objects and API

+ API usage of TensorRT objects, mainly to show the functions and the effect of their parameters.

### Builder

+ Common members and methods of the **Builder** class

### BuilderConfig

+ Common members and methods of **BuilderConfig** class

### CudaEngine

+ Common members and methods of **CudaEngine** class

### ExecutionContext

+ Common members and methods of **ExecutionContext** class

### HostMemory

+ Common members and methods of **HostMemory** class

### Int8-PTQ

+ General process of using TensorRT INT8-PTQ mode, and related methods of various classes involved.

+ See 03-APIModel/MNISTExample-\* for examples of using INT8-PTQ mode in the actual model. There is little different work to be done when exporting models into TensorRT from ONNX.

### Int8-QDQ

+ Example of a TensorRT network with Quantize and Dequantize layers.

+ See 04-Parser/\*-QAT for the examples of using INT8-QAT mode in the actual model, which involves the modification of the model training framework, but still little different work to be done when exporting models into TensorRT.

### Layer

+ Common members and methods of **Layer** class

+ The examples of each layer and its parameters

+ TensorRT8 + Explicit Batch mode is used except some special circumstances.

+ Each \*Layer/\*.md describe the usage of all layer, its corresponding parameters, the output and algorithm explanations of the layer in detail, as well as the reasults of the example code \*Layer/\*.py.

+ Latex is not supported by the the Markdown of GitLab/Github, so the formula in the \*Layer/\*.md can not be rendered during online preview. You can download the file and use the markdown software with Latex supported (such as Typora) to read it. Meanwhile, we provide PDF format of each. md (in 50-Resource/Layer) for reading directly.

+ Dimension support matrix of each layer  [Link](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#layers-matrix)

+ Precision support matrix of each layer [Link](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#layers-precision-matrix)

+ Flow control support matrix of each layer [Link](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#layers-flow-control-constructs)

### Network

+ Common members and methods of the **Network** class

### OnnxParser

+ Common members and methods of the **OnnxParser** class

### OptimizationProfile

+ Common members and methods of the **OptimizationProfile** class

### Runtime

+ Common members and methods of the **Runtime** class

### Tensor

+ Common members and methods of the **Tensor** class

---

## 03-APIModel -- Examples of rebuilding a model using layer API

+ Taking a complete handwritten digit recognition model based on MNIST dataset as an example, the process of rebuilding the trained models layer by layer from various machine learning frameworks in TensorRT with layer API is demonstrated, including the weights extraction from the original model, the model rebuilding and weights loading etc.

+ All examples based on MNIST dataset can run in FP16 mode or INT8-PTQ mode.

### MNISTExample-Paddlepaddle

+ Use PaddlePaddle framework to train handwritten digit recognition model, then rebuild the model and do inference in TensorRT.

### MNISTExample-pyTorch

+ Use pyTorch framework to train handwritten digit recognition model, then rebuild the model and do inference in TensorRT.

+ **Here contains an example of using TensorRT C++ API to rebuild the model layer by layer.**

### MNISTExample-TensorFlow1

+ Use TensorFlow1 framework to train handwritten digit recognition model, then rebuild the model and do inference in TensorRT.

### MNISTExample-TensorFlow2

+ Use TensorFlow2 framework to train handwritten digit recognition model, then rebuild the model and do inference in TensorRT.

### TypicalaPI-Paddlepaddle[TODO]

+ Weights extraction, layer rebuilding and weights loading of various typical layers from PaddlePaddle to TensorRT.

### TypicalaPI-pyTorch[TODO]

+ Weights extraction, layer rebuilding and weights loading of various typical layers from pyTorch to TensorRT.

### TypicalaPI-TensorFlow1

+ Weights extraction, layer rebuilding and weights loading of various typical layers from TensorFlow1 to TensorRT.

+ Till now, convolution, Fully Connection and LSTM are included.

### TypicalaPI-TensorFlow2[TODO]

   Weights extraction, layer rebuilding and weights loading of various typical layers from TensorFlow2 to TensorRT.

---

## 04-Parser -- Examples of rebuilding a model using Parser

+ Taking a complete handwritten digit recognition model based on MNIST dataset as an example, the process of rebuilding the trained models from various machine learning frameworks in TensorRT with build-in parser is demonstrated, including the process of exporting the model into ONNX format and then into TensorRT.

+ Introduction of ONNX operator [Link](https://github.com/onnx/onnx/blob/main/docs/Operators.md)

+ Support matrix of ONNX operator in TensorRT parser [Link](https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md)

+ All examples based on MNIST dataset can run in FP16 mode or INT8-PTQ mode (except the examples using INT8-QAT already).

### Paddlepaddle-ONNX-TensorRT

+ Use PaddlePaddle framework to train handwritten digit recognition model, then export the model into ONNX and into TensorRT by parser to do inference.

### pyTorch-ONNX-TensorRT

+ Use pyTorch framework to train handwritten digit recognition model, then export the model into ONNX and into TensorRT by parser to do inference.

### pyTorch-ONNX-TensorRT-QAT

+ Use pyTorch framework to train handwritten digit recognition model with Quantization Aware Training (QAT), then export the model into ONNX and into TensorRT by parser to do inference.

+ Original example code for reference [Link](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization)

+ In the original example, the process of calibration and fine tuning depends on docker image *nvcr.io/nvidia/pytoch:20.08-py3*, but the process of exporting model into ONNX depends on docker image *nvcr.io/nvidia/pytoch:21.12-py3* or newer ones. The reason is the example needs the file */opt/pythoch/vision/references/classification/train.py* which changes a lot in newer version of docker images, meanwhile exporting QAT model into ONNX is not support in the *torch.onnx* module of the old version of image.

+ The example code here in the cookbook uses a fully localized model, which removes all the above dependencies, so it can run independently.

### TensorFlow1-Caffe-TensorRT

+ **This workflow has been deprecated. Please use ONNX instead, and this example is for reference only.**

+ Use TensorFlow1 framework to train handwritten digit recognition model, then export the model into .prototxt and.caffemodel file and into TensorRT by parser to do inference.

### TensorFlow1-ONNX-TensorRT

+ Use TensorFlow1 framework to train handwritten digit recognition model, then export the model into ONNX and into TensorRT by parser to do inference.

### TensorFlow1-ONNX-TensorRT-QAT

+ Use TensorFlow1 framework to train handwritten digit recognition model with Quantization Aware Training (QAT), then export the model into ONNX and into TensorRT by parser to do inference.

+ Original example code for reference [Link](https://github.com/shiyongming/QAT_demo)

### TensorFlowF1-UFF-TensorRT

+ **This workflow has been deprecated. Please use ONNX instead, and this example is for reference only.**

+ Use TensorFlow1 framework to train handwritten digit recognition model, then export the model into .uff file and into TensorRT by parser to do inference.

### TensorFlow2-ONNX-TensorRT

+ Use TensorFlow2 framework to train handwritten digit recognition model, then export the model into ONNX and into TensorRT by parser to do inference.

### TensorFlow2-ONNX-TensorRT-QAT[TODO]

+ Use TensorFlow2 framework to train handwritten digit recognition model with Quantization Aware Training (QAT), then export the model into ONNX and into TensorRT by parser to do inference.

---

## 05-Plugin -- Examples of writing customized plugins

+ Example of implementing customized plugins and applying them to TensorRT network.

+ The core example is **usePluginV2DynamicExt**， which represents the most basic usage of the mainstream plugin in TensorRT 8. All other examples are extended from this one. Please read this example first if you are fresh to write Plugin.

+ There attached a file **test\*Plugin.py** in each example for unit test. It is stongly recommended to conduct unit test after writing plugin but before integrating it into the model.

### API

+ Common members and methods of the **Plugin** class in Python.

### C++-UsePluginInside

+ Build a TensorRT engine with plugins inside the executable file

### C++-UsePluginOutside

+ Build a TensorRT engine with plugins outside the executable file

### LoadNpz

+ Example to read data from .npz file (may contains weights or other information and is provided by python) and use in plugin.

+ Show the method of reading data from .npz file in C++ with cnpy library.

+ Origial repository of cnpy [Link](https://github.com/rogersce/cnpy)

### multipleVersion

+ Example of writing and using different versions of the plugin sharing the same name.

+ When using the built-in plugins of TensorRT, you also need to confirm the version plugin in this way.

### PluginPrecess

+ Example of the calling sequence of the member methods of the plugin when using Multiple-Optimization-Profiles in the network with a plugin.

+ The example tells us which and the order of the member methods are called on each steps (building step or runtime step).

### PluginReposity

+ A small repository of some wide-used plugins

+ The correctness of the plugins are guaranteed, but the performance may not be fully optimized.

+ The feature (including input / output tensor or parameters) may be different among various versions of the same plugin, but we retain them all together here.

+ Plugins with the suffix "-TRT8" indicate that the format requirements of TensorRT8 have been aligned. Many other plugins are written based on TensorRT6 or TensorRT7, and some member methods need to be modified when compiling on newer versions of TensorRT (see *usePluginV2Ext/AddScalarPlugin.h* and *usePluginV2Ext/AddScalarPlugin.cu* to find the differences).

### PluginSerialize

+ TODO

### UseCuBLAS

+ Example of using cuBLAS to calculate matrix multiplication in Plugin

+ Here attached an standalone example *useCuBLASAlone.cu* to use cuBLAS alone in CUDA C++ to calculate GEMM.

### UseFP16

+ Example of using FP16 data type as input / output tensor in Plugin.

+ The example has the same function as *usePluginV2DynamicExt*.

### UseINT8-PTQ

+ Example of using INT8 data type as input / output tensor in Plugin.

+ The example has the same function as *usePluginV2DynamicExt*.

+ Note that the data layout ("format" in TensorRT's API) of input / output tensor may not be linear.

### UseINT8-QDQ

+ Example of using plugin in QAT network (with quantize layer and dequantize layer).

+ The example has the same function as *usePluginV2DynamicExt*.

### UsePluginV2DynamicExt

+ Use the **IPluginV2DynamicExt** class to implement a plugin to add the a scalar value to all elements of the input tensor.

+ Features:
  + The declarations of the member methods align with requirement of TensorRT8.
  + Use Explicit Batch mode + Dynamic Shape mode.
  + The shape of input tensor is dynnamic (inferences of input tensor with the same number of dimensions but different shapes share one TensorRT engine).
  + Scalar addends are determined during construction (cannot be changed between multiple references)
  + Serialization and deserialization supported.

### UsePluginV2Ext

+ Use the **IPluginV2Ext** class to implement a Plugin to to add the a scalar value to all elements of the input tensor.

+ Features:
  + Using Implicit Batch mode
  + The shape of input tensor is static (inferences of input tensor with different shapes must use distinctive engines).
  + Scalar addends are determined during construction (cannot be changed between multiple references)
  + Serialization and deserialization supported.
  + **Show the difference of plugin between TensorRT 7 and TensorRT 8**, especially the declaration and definition of the member function "queue".

### UsePluginV2IOExt

+ Use the **IPluginV2IOExt** class to implement a Plugin to to add the a scalar value to all elements of the input tensor.

+ Features:
  + Using Implicit Batch mode
  + Different data types and layouts for each output tensor supported (there is only one output tensor in this example though).

---

## 06-PluginAndParser -- Examples of combining the usage of parser and plugin

+ For the model using any operator which is supported in ONNX but not in TensorRT, we can combine to use parser and plugin to export the model into TensorRT.

+ Please refer to 08-Tool/onnxGraphSurgeon to learn the method of using onnx-graphsurgeon to edit the graph of ONNX.

### pyTorch-FailConvertNonZero

+ A Nonzero operator is used in the trained model in pyTorch which is not natively supported before TensorRT 8.5, so it fails to parse from ONNX into  TensorRT.

### pyTorch-AddScalar

+ Replace a subgraph as AddScalar Plugin during exporting model from pyTorch to ONNX to TensorRT.

### pyTorch-BigNode

+ Replace a big submodule in pyTorch graph and replace it as plugins during exporting model from pyTorch to ONNX to TensorRT.

---

## 07-FameworkTRT -- Examples of using build-in TensorRT API in Machine Learning Frameworks

### TensorFlow1-TFTRT

+ Use TFTRT in TensorFlow1 to run a trained TF model.

### TensorFlow2-TFTRT

+ Use TFTRT in TensorFlow2 to run a trained TF model.

### Torch-TensorRT

+ Use Torch-TensorRT (deprecated name: TRTorch) to run a trained pyTorch model.

---

## 08-Tool -- Assistant tools for development

+ Introduction of some assistant tools for model development in TensorRT, including the help information and usage examples.

### Netron

+ A visualization tool of neural network, which supports **ONNX**, TensorFlow Lite, Caffe, Keras, Darknet, PaddlePaddle, ncnn, MNN, Core ML, RKNN, MXNet, MindSpore Lite, TNN, Barracuda, Tengine, CNTK, TensorFlow.js, Caffe2, UFF, and experimental supports for PyTorch, TensorFlow, TorchScript, OpenVINO, Torch, Vitis AI, kmodel, Arm NN, BigDL, Chainer, Deeplearning4j, MediaPipe, MegEngine, ML.NET, scikit-learn

+ Original repository: [Link](https://github.com/lutzroeder/Netron)

+ Installation: please refer to the original repository of the tool.

+ We usually use this tool to check metadata of compute graph, structure of network, node information and tensor information of the model.

+ Here is a ONNX model *model.onnx* in the subdirectory which can be opened with Netron.

+ Here is another interesting tool **onnx-modifier** [Link](https://github.com/ZhangGe6/onnx-modifier). Its appearance is similar to Netron and ONNX compute graph can be edited directly by the user interface. However, it becomes much slower during opening or editing large models, meanwhile the type of graph edition are limited.

### NetworkInspector

+ Network serialize / deserialize to / from JSON.

### NetworkPrinter

+ Print the network onto standard output.

### Nsight systems

+ Program performance analysis tool (replacing the old performance analysis tools nvprof and nvvp).

+ Original website [Link](https://developer.nvidia.com/nsight-systems).

+ Installation:
  + atteched with CUDA installation [Link](https://developer.nvidia.com/cuda-zone), the executable program are */usr/local/cuda/bin/nsys* and */usr/local/cuda/bin/nsys*
  + standalone installation [Link](https://developer.nvidia.com/nsight-systems), the executable program are *?/bin/nsys* and *?/bin/nsys*

+ Document [Link](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).

+ Note: please update nsight systems to the latest version. Nsight Systems of older version is not able to open .qdrep or .qdrep-nsys or .nsys-rep file generated by newer version of Nsight Systems.

+ Usage: run `nsys profile ./a.exe` in the command line to obtain a .qdrep or .qdrep-nsys or .nsys-rep file, then open nsys-ui, drag the above file into it, so we can observe the timeline in the user interface.

+ Suggestions for using Nsight Systems in model development in TensorRT:
  + Use the switch *ProfilingVerbosity* while building serialized network so that we can get more information about the layer of the model in the timeline of Nsight Systems (see 09-Advance/ProfilingVerbosity).
  + Only analyze phase of inference, not phase of building.
  + Nsight Systems can be used with trtexec (for example `nsys profile -o myProfile -f true trtexec --loadEngine=model.plan`) or your own script (for example `nsys profile -o myProfile -f true python3 myScript.py`).

### nvtx

+ Use NVIDIA®Tools Extension SDK to add mark in timeline of Nsight systems.

### OnnxGraphSurgeon

+ A python library for ONNX compute graph edition, which different from the library *onnx*.

+ Installation: `pip install nvidia-pyindex onnx-graphsungeon`

+ Document [Link](https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html)

+ The example code here refers to the NVIDIA official repository about TensorRT tools [Link](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/examples).

+ Function:
  + Modify metadata/node / tensor / weight data of compute graph.
  + Modify subgraph: Add / Delete / Replace / Isolate
  + Optimize: constant folding / topological sorting / removing useless layers.

### Onnxruntime

+ A library to run the ONNX model using different backends. Here we usually use it for accuracy test to check the correctness of the exported model from the training framework.

+ Installation: `pip install runtime-gpu -i https://pypi.ngc.nvidia.com`

+ Document [Link](https://onnxruntime.ai/)

### Polygraphy-API

+ API tool of polygraphy

### Polygraphy-CLI

+ CLI tool of polygraphy

+ Deep learning model debugger, including equivalent python APIs and command-line tools.

+ Installation `pip install polygraph`

+ Document [Link](https://docs.nvidia.com/deeplearning/tensorrt/polygraphy/docs/index.html) and a tutorial video [Link](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31695/).

+ Function:
  + Do inference computation using multiple backends, including TensorRT, onnxruntime, TensorFlow etc.
  + Compare results of computation layer by layer among different backends.
  + Generate TensorRT engine from model file and serialize it as .plan file.
  + Print the detailed information of model.
  + Modify ONNX model, such as extracting subgraph, simplifying computation graph.
  + Analyze the failure of parsing ONNX model into TensorRT, and save the subgrapha that can / cannot be converted to TensorRT.

### trex

+ A vidualization tool for TensorRT engine.

+ Original website [Link](https://github.com/NVIDIA/TensorRT/tree/main/tools/experimental/trt-engine-explorer).

+ Function
  + Generate JSON files while buildtime and runtime of TensoRT and analyze the information of the TensorRT engine.
  + Draw the serialized network in the TensorRT engine layer by layer.
  + Provide the statistic report of the time / memory cost during the infernece computation.

### trtexec

+ Command-line tool of TensorRT, attached with an end-to-end performance test tool.

+ Installation: attached with TensorRT, the executable program is *?/bin/trtexec.*

+ Function
  + Generate TensorRT engine from model file and serialize it as .plan file.
  + Print the detailed information of model.
  + End-to-end performance test on the model.

---

## 09-Advance -- Examples of advanced features in TensorRT

### AlgorithmSelector

+ Example of printing, saving and filtering the tactics of each layer of the network manually during the automatic optimization stage of buildtime.

### AuxStream

+ Use aux CUDA stream to do inference.

### CreateExecutionContextWithoutDeviceMemory

+ Example of using CreateExecutionContextWithoutDeviceMemory method of the Execution Context object.

### CudaGraph

+ Example of using CUDA Graph with TensorRT to solve the problem of launch bound.

### DataFormat

+ Different data format and corresponding information

### DyanmicShapeOutput

+ Data-Dependent network and various output shape by the values of the input data.

### EmptyTensor[TODO]

+ Example of using TensorRT empty tensor.

### EngineInspector

+ Example of printing the structure of serialized network layer by layer after automatically optimization of TensorRT.

### ErrorRecoder

+ Example of providing a customizied error log in TensorRT.

### Event

+ Use CUDA Event to control the flow of the inference process.

### GPUAllocator

+ Example of using a customized GPU memory allocator during the buildtime and runtime in TensorRT.

### LabeledDimension

+ this feature is introduced since TensorRT 8.4.

+ Example of input tensors with named dimensions, which is advantageous for TensorRT to perform relevant optimization.

+ For example, in NLP models like Wenet, we have two input tensors "speech" (shape: \[nBatchSize,nSequenceLength,nEmbedding\]) and "speech_length" (shape: \[nBatchSize\]). Without the named dimension, TensorRT will give warning below because both the two "nBatchSize" and one "nSequenceLength" dimensions are all -1 in the viewpoint of TensorRT, and Myelin system will refuse to try some paths of optimization due to uncertain equality of these shapes. But once we name them (especially the same name "nBatchSize" between two input tensors), the warning disappears and more paths of optimization will be performed.

```shell
[W] [TRT] Myelin graph with multiple dynamic values may have poor performance if they differ. Dynamic values are: 
[W] [TRT]  (# 1 (SHAPE speech))
[W] [TRT]  (# 0 (SHAPE speech))
```

### Logger

+ Example of using a customized information logger during the buildtime and runtime in TensorRT.

### MultiContext

+ Examples of doing inference with more than one Excution Context on one TensorRT engine.

### MultiOptimizationProfile

+ Examples of doing inference with more than one Optimization Profile on one TensorRT engine.

+ During automatic optimization of TensorRT, the beat kernels are chosed based on the opt shape of dynamic shape, and the compatibility between min shape and max shape is guaranteed. Only one optimization profile may sacrifice performance for compatibility when the range of dynamic shape is large. So we can split the range of dynamic shape into more pieces, and choose the best suitable one when the actual input tensor is comming, please refer to 10-BestPractice/UsingMultiOptimizationProfile to see the effect.

### MultiStream

+ Examples of doing inference with asynchronization API and more than one CUDA stream on one TensorRT engine.

+ This example only introduces the use of CUDA stream in Python. Please refer 09-Advance/StreamAndAsync to see the use of page locked memory, especially copying data between page locked memory and numpy array objects, which is necessary for asynchronization.

+ Using CUDA streams, the time of data copy and GPU computation can overlap together, saving time while a plenty of infernece should be done. It is especially effective if cost of I/O time of a model is obvious compared to the computation time.

### OutputAllocator

+ GPU memory allocator for Data-Dependent network.

### Profiler

+ Example of using a customized repoter to note the time spent by each layers in the network during the buildtime in TensorRT.

+ It provides data support for subsequent manual optimization.

### ProfilingVerbosity

+ A switch of TensorRT to record detailed journal, which can be used by Nsight Sysytems to provide more information about the model. network and layers.

### Refit

+ Examples of updating the weights in TensorRT engine without rebuilding serialized network.

### Safety

+ Examples of using safety module of TensorRT to build and run a model.

+ Safety module is only applicable to Drive Platform (QNX) [Link](https://github.com/NVIDIA/TensorRT/issues/2156).

### Sparsity

+ Examples of using structured sparsity feature of TensorRT to build and run a model.

### StreamAndAsync

+ Examples of using CUDA stream and asychronization API to do inference in TensorRT.

+ The example also contains the use of page locked memory, especially copying data between page locked memory and numpy array objects, which is necessary for asynchronization.

### StrictType

+ Examples of specify the accumulate data type of each layer of the network manually.

### TacticSource

+ Examples of specify the scope of alternative kernels in the automatic optimization phase of buildtime in TensorRT manually.

+ Tactic sources includes cuBLAS, cuBLASLt, cuDNN, edge_mask_convolutions etc. Better performance may be achieve if more tactic sources are chose, but more time and memory during automatic optimization will be cost in the same time.

+ cuDNN could be considered to forbid if much GPU memory cost need to reduce during both buildtime and runtime.

### TensorRTGraphSurgeon

+ Do graph edition using TensorRT API.

### TimingCache

+ Examples of using reusable timing cache for building similar engine repeatedly in TensorRT.

+ Without specify manually, timing cache is still utilized during single build time (for example there are several convolution layers with identical parameters in the network) but deleted once the serialized network is produced. If we specify it manually, the timing cache can be saved as file and used among building serialized network for more times.

### TorchOperation

+ Use CUDA runtime API in pyTorch rather than cudart library.

+ This is usually needed when some parts of our pipeline are in pyTorch.

---

## 10-BestPractice -- Examples of interesting TensorRT optimization

+ Some examples of manual optimization in TensorRT. Please Dettailed description and result of performance test

+ Some references [Link](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#optimize-layer).

### AdjustReduceLayer

+ Optimization of the reduce layer. It usually appeares higher performance to compute at the last dimension of the input tensor, rather then at the other dimensions.

### AlignSize

+ Optimization the matrix multiplication layer. It usually appeares higher performance if data alignment is guaranteed.

### ComputationInAdvance

+ Optimization of computation of buildtime in advance to reduce the computation during infernece.

+ This example is from the model of Wenet.

### Convert3DMMTo2DMM (?)

+ Optimize the matrix multiplication layer. In some occasion, matrix multiplication of two dimension appears higher performance than the three dimension ones.

### ConvertTranposeMultiplicationToConvolution

+ Optimization a combination of Transpose and Matrix Multiplication layers into a combination of Convolution and Shuffle layers.

### EliminateSqueezeUnsqueezeTranspose

+ Optimization of removing some Squeeze / Unsqueeze /T ranspose layers for further layer fusion by TensorRT.

### IncreaseBatchSize

+ Optimization of increasing batch size during inference to improve the overall throughput in TensorRT.

### UsingMultiOptimizationProfile

+ Optimization of using more than one Optimization Profile to improve the overall performance in TensorRT.

### UsingMultiStream

+ Optimization of using asynchronization API and more than one CUDA stream to overlap the time of data copy and computation.

### WorkFlowOnONNXModel[TODO]

+ A overall workflow of optimization from ONNX model to TensorRT engine, including:
  + Save input / output data as .npz file for reference from ONNX model.
  + Test performance of ONNX model using onnxruntime within specified range of input size.
  + Simplify the ONNX mdoel using polygraphy.
  + Do optimization manually using onnx-graphsurgeon in Python script.
  + Parse the adjusted ONNX model into TensorRT and build a engine.
  + Test the accurracy of TensorRT model, using the input data from .npz file and comparing the result between onnxruntime and TEnsorRT.
  + Test performance of TensorRT model  within specified range of input size.
  + Produce a report about the all workflow.

---

## 11-ProblemSolving[TODO]

+ Some error messages and corresponding solutions when deploying models to TensorRT.

### Parameter Check Failed

### Slice Node With Bool IO

### Weights Are Not Permitted Since They Are Of Type Int32

### Weights Has Count X But Y Was Expected

---

## 50-Resource -- Resource of document

+ PDF version of the slides of TensorRT tutorial (in Chinese), as well as some other useful reference materials.

+ Since Github's markdown does not natively support Latex,  *. of Latex is used in cookie book documents Md will export a PDF file to this directory for easy browsing

+ Latex is not supported by the the Markdown of GitLab/Github, so the formula in the \*.md can not be rendered during online preview. All the \*.md with Latex are exported as PDF and saved one copy in this directory.

+ Contents included
  + 02-API-Layer/\*.pdf, exported from cookbook/02-API/Layer/\*Layer/*\.md, which describes the usage and corresponding parameters of each layer API.
  + number.pdf, exported from cookbook/51-Uncategorized/number.md, which notes the data range of both floating point and integer types, including FP64, FP32, TF32, FP16, BF16, FP8e5m2, FP8e4m3, INT64, INT32, INT16, INT8, INT4.
  + TensorRTTutorial-TRT8.2.3-V1.1.pdf, exported from 51-Uncategorized/TensorRTTutorial-TRT8.2.3-V1.1.pptx, which contains a tutorial slides of TensorRT 8.2.3 with audio in Chinese.
  + Hackathon2022-PreliminarySummary-WenetOptimization-V1.1.pdf, exported from 51-Uncategorized/Hackathon2022-PreliminarySummary-WenetOptimization-V1.1.pptx, which contains a lecture of optimizing the Wenet model in TensorRT using as the competition question in Hackathon 2022.

---

## 51-Uncategorized

+ Unclassified things.

+ Contents included
  + getTensorRTVersion.sh, shows the version of CUDA, cuDNN, TensorRT of the current environment.
  + number.md, notes the data range of both floating point and integer types, including FP64, FP32, TF32, FP16, BF16, FP8e5m2, FP8e4m3, INT64, INT32, INT16, INT8, INT4.
  + TensorRTTutorial-TRT8.2.3-V1.1.pptx, contains a tutorial slides of TensorRT 8.2.3 with audio in Chinese.
  + Hackathon2022-PreliminarySummary-WenetOptimization-V1.1.pptx, contains a lecture of optimizing the Wenet model in TensorRT using as the competition question in Hackathon 2022.

---

## 52-Deprecated

+ Some APIs and usages in TensorRT which have been deprecated or removed in the newer version of TensorRT. If you run them directly, you will get error messages.

---

## 99-NotFinish

+ Incomplete example code and plans of new example codes proposed by our readers.
