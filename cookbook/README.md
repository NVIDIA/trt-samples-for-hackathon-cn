# TensorRT Cookbook

## General Introduction

+ This repository is presented for NVIDIA TensorRT beginners and developers, which provides TensorRT-related learning and reference materials, as well as code examples.

+ This README.md contains catalogue of the cookbook, you can search your interested subtopics and go to the corresponding directory to read. (not finished)

+ Related materials (slices, datasets, models and PDF files): [Baidu Netdisk](https://pan.baidu.com/s/14HNCFbySLXndumicFPD-Ww) (Extraction code: gpq2)

+ Related video tutorial on Bilibili website: [TensorRT-8.6.1](https://www.bilibili.com/video/BV1jj411Z7wG/), [TensorRT-8.2.3](https://www.bilibili.com/video/BV15Y4y1W73E), [Hackathon2022](https://www.bilibili.com/video/BV1i3411G7vN)

+ Chinese contents will be translated into English in the future ([\u4E00-\u9FA5]+)

+ Recommend order to read (and try) the subtopics if you are freshman to the TensorRT:
  + 01-SimpleDemo/TensorRT8.6
  + 04-BuildEngineByONNXParser/pyTorch-ONNX-TensorRT
  + 07-Tool/Netron
  + 07-Tool/trtexec
  + 05-Plugin/UsePluginV2DynamicExt
  + 05-Plugin/UseONNXParserAndPlugin-pyTorch
  + ...

## Steps to setup

+ We recommend to use NVIDIA-optimized Docker to run the examples. [Steps to install](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

+ Start the container

```shell
docker run -it -e NVIDIA_VISIBLE_DEVICES=0 --gpus "device=0" --name trt-cookbook \
--shm-size 16G --ulimit memlock=-1 --ulimit stack=67108864 \
-v ~:/work \
nvcr.io/nvidia/pytorch:23.04-py3 /bin/bash
```

+ Inside thecontainer, go to directory of cookbook firstly

```shell
cd <PathToCookBook>
pip3 install -r requirements.txt
```

+ prepare the dataset we may need in some examples, they can be found from [Link](http://yann.lecun.com/exdb/mnist/) or [Link](https://storage.googleapis.com/cvdf-datasets/mnist/) or the Baidu Netdisk above.

```shell
cd 00-MNISTData

# download dataset (4 gz files in total) and put them here as <PathToCookBook>/00-MNISTData/*.gz

python3 extractMnistData.py  # extract some of the dataset into pictures for later use
```

+ There is also an *8.png* from test set of the dataset in this directory, which is used as the input data of TensorRT
+ now we can go to other directory to try the examples we are interested and run the examples

## Table of tested docker images

+ Notice that pyTorch and TensorFlow in NVIDIA-optimized Docker is somewhere different from the version installed directly by *pip install*.
+ The data can be found from [Link](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)

|            Name of Docker Image             | python |  CUDA   |   cuBLAS   |  cuDNN  | TensorRT | Nsight-Systems | Lowest  Driver |            Comment             |
| :-----------------------------------------: | :----: | :-----: | :--------: | :-----: | :------: | :------------: | :------------: | :----------------------------: |
|    **nvcr.io/nvidia/tensorrt:19.12-py3**    |  3.6   | 10.2.89 | 10.2.2.89  |  7.6.5  |  6.0.1   |    2019.6.1    |   440.33.01    |  Last version with TensorRT 6  |
|    **nvcr.io/nvidia/tensorrt:21.06-py3**    |  3.8   | 11.3.1  | 11.5.1.109 |  8.2.1  | 7.2.3.4  |  2021.2.1.58   |   465.19.01    |  Last version with TensorRT 7  |
|    **nvcr.io/nvidia/pytorch:23.02-py3**     |  3.8   | 12.0.1  |   12.0.2   |  8.7.0  |  8.5.3   |    2022.5.1    |      525       |  Last version with pyTorch 1   |
| **nvcr.io/nvidia/tensorflow:23.03-tf1-py3** |  3.8   | 12.1.0  |   12.1.0   | 8.8.1.3 |  8.5.3   |  2023.1.1.127  |      530       | Last version with TensorFlow 1 |
|    **nvcr.io/nvidia/tensorrt:23.03-py3**    |  3.8   | 12.1.0  |   12.1.0   | 8.8.1.3 |  8.5.3   |  2023.1.1.127  |      530       | Last version with TensorRT 8.5 |
|    **nvcr.io/nvidia/pytorch:23.06-py3**     |  3.10  | 12.1.1  |  12.1.3.1  |  8.9.2  | 8.6.1.6  |    2023.2.3    |      530       |  **cookbook prefer version**   |
| **nvcr.io/nvidia/tensorflow:23.06-tf2-py3** |  3.10  | 12.1.1  |  12.1.3.1  |  8.9.2  | 8.6.1.6  |    2023.2.3    |      530       |                                |
|  **nvcr.io/nvidia/paddlepaddle:23.06-py3**  |  3.10  | 12.1.1  |  12.1.3.1  |  8.9.2  | 8.6.1.6  |    2023.2.3    |      530       |                                |
|    **nvcr.io/nvidia/tensorrt:23.06-py3**    |  3.10  | 12.1.1  |  12.1.3.1  |  8.9.2  | 8.6.1.6  |    2023.2.3    |      530       |                                |

## Important update of the repository

+ **18th June 2023**. Updated to TensorRT 8.6 GA. Finish TensorRT tutorial (slice + audio) for Bilibili.

+ **17th March 2023**. Freeze code of branch TensorRT-8.5
  + Translate almost all contents into English (except 02-API/Layer/\*.md)
  + Come to development work of TensorRT 8.6 EA

+ **10th October 2022**. Updated to TensorRT 8.5 GA. Cookbook with TensorRT 8.4 is remained in branch old/TensorRT-8.4.

+ **15th July 2022**. Updated to TensorRT 8.4 GA. Cookbook with older version of TensorRT is remained in branch old/TensorRT-8.2.

## Useful Links

+ [TensorRT Download](https://developer.nvidia.com/nvidia-tensorrt-download)
+ [TensorRT Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)
+ [Document](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) and [Document Archives](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html)
+ [Supporting Matrix](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html) and [Supporting Matrix (Old version)](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-843/support-matrix/index.html)
+ [C++ API Document](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api) and [Python API Document](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/) and [API Change](https://docs.nvidia.com/deeplearning/tensorrt/api/index.html)
+ [Operator algorithm](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/)
+ [Onnx Operator Supporting Matrix](https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md)
+ [TensorRT Open Source Software](https://github.com/NVIDIA/TensorRT)
+ [NSight- Systems](https://developer.nvidia.com/nsight-systems)

+ Others:
  + [ONNX-TensorRT](https://github.com/onnx/onnx-tensorrt)
  + [TF-TRT](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html)
  + [Torch-TensorRT](https://pytorch.org/TensorRT/)
  + [ONNX model zoo](https://github.com/onnx/models)
  + [tensorrtx (build engine by API)](https://github.com/wang-xinyu/tensorrtx)
  
---

## Catalogue and introduction of the files

### include

+ Common files used by some examples in the cookobook.

### 00-MNISTData -- Related dataset

+ The MNIST dataset used by Cookbook, which needs to be downloaded and preprocessed before running other example code

### 01-SimpleDemo -- Examples of a complete program using TensorRT

+ Basic steps to use TensorRT, including network construction, engine construction, serialization/deserialization, inference computation, etc.

+ Examples cover different versions of TensorRT, all with equivalent implementations in C++ and python

### 02-API -- Examples of TesnorRT core objects and API

+ API usage of TensorRT objects, mainly to show the functions and the effect of their parameters.

### 03-BuildEngineByTensorRTAPI -- Examples of rebuilding a model using layer API

+ Taking a complete handwritten digit recognition model based on MNIST dataset as an example, the process of rebuilding the trained models from various machine learning frameworks in TensorRT with layer APIs is demonstrated, including the weights extraction from the original model, the model rebuilding and weights loading etc.

+ All examples based on MNIST dataset can run in FP16 mode or INT8-PTQ mode.

+ Some examples of importing typical layers from various machine learning frameworks to TensorRT.

### 04-BuildEngineByONNXParser -- Examples of rebuilding a model using ONNX Parser

+ Taking a complete handwritten digit recognition model based on MNIST dataset as an example, the process of rebuilding the trained models from various machine learning frameworks in TensorRT with build-in ONNX parser is demonstrated, including the process of exporting the model into ONNX format and then into TensorRT.

+ All examples based on MNIST dataset can run in FP16 mode or INT8-PTQ/INT8-QAT mode.

### 05-Plugin -- Examples of writing customized plugins

+ Example of implementing customized plugins and applying them to TensorRT network.

+ The core example is **usePluginV2DynamicExt**， which represents the most basic usage of the mainstream plugin in TensorRT 8. All other examples are extended from this one. Please read this example first if you are fresh to write Plugin.

+ There attached a file **test\*Plugin.py** in each example for unit test. It is stongly recommended to conduct unit test after writing plugin but before integrating it into the model.

+ Here are examples of combining ONNX parser and plugin to build the model in TensorRT.

### 06-FameworkTRT -- Examples of using build-in TensorRT API in Machine Learning Frameworks

+ Examples of using build-in TensorRT API in various machine learning frameworks.

### 07-Tool -- Assistant tools for development

### 08-Advance -- Examples of advanced features in TensorRT

### 09-BestPractice -- Examples of interesting TensorRT optimization

### 10-ProblemSolving[TODO]

+ Some error messages and corresponding solutions when deploying models to TensorRT.

### 50-Resource -- Resource of document

+ PDF version of the slides of TensorRT tutorial (in Chinese), as well as some other useful reference materials.

+ Latex is not supported by the the Markdown of GitLab/Github, so the formula in the \*.md can not be rendered during online preview. All the \*.md with Latex are exported as PDF and saved one copy in this directory.

### 51-Uncategorized

+ Unclassified things.

### 52-Deprecated

+ Some APIs and usages in TensorRT which have been deprecated or removed in the newer version of TensorRT. If you run them directly, you will get error messages.

### 99-NotFinish

+ Incomplete example code and plans of new example codes proposed by our readers.

## Tree diagram of the repo

```txt
├── 00-MNISTData
│   ├── test
│   └── train
├── 01-SimpleDemo
│   ├── TensorRT6
│   ├── TensorRT7
│   ├── TensorRT8.0
│   └── TensorRT8.5
├── 02-API
│   ├── AlgorithmSelector
│   ├── AuxStream
│   ├── Builder
│   ├── BuilderConfig
│   ├── CudaEngine
│   ├── EngineInspector
│   ├── ErrorRecoder
│   ├── ExecutionContext
│   ├── GPUAllocator
│   ├── HostMemory
│   ├── INT8-PTQ
│   │   └── C++
│   ├── Layer
│   │   ├── ActivationLayer
│   │   ├── AssertionLayer
│   │   ├── CastLayer
│   │   ├── ConcatenationLayer
│   │   ├── ConstantLayer
│   │   ├── ConvolutionNdLayer
│   │   ├── DeconvolutionNdLayer
│   │   ├── EinsumLayer
│   │   ├── ElementwiseLayer
│   │   ├── FillLayer
│   │   ├── FullyConnectedLayer
│   │   ├── GatherLayer
│   │   ├── GridSampleLayer
│   │   ├── IdentityLayer
│   │   ├── IfConditionStructure
│   │   ├── LoopStructure
│   │   ├── LRNLayer
│   │   ├── MatrixMultiplyLayer
│   │   ├── NMSLayer
│   │   ├── NonZeroLayer
│   │   ├── NormalizationLayer
│   │   ├── OneHotLayer
│   │   ├── PaddingNdLayer
│   │   ├── ParametricReLULayer
│   │   ├── PluginV2Layer
│   │   ├── PoolingNdLayer
│   │   ├── QuantizeDequantizeLayer
│   │   ├── RaggedSoftMaxLayer
│   │   ├── ReduceLayer
│   │   ├── ResizeLayer
│   │   ├── ReverseSequenceLayer
│   │   ├── RNNv2Layer
│   │   ├── ScaleLayer
│   │   ├── ScatterLayer
│   │   ├── SelectLayer
│   │   ├── ShapeLayer
│   │   ├── ShuffleLayer
│   │   ├── SliceLayer
│   │   ├── SoftmaxLayer
│   │   ├── TopKLayer
│   │   └── UnaryLayer
│   ├── Logger
│   ├── Network
│   ├── ONNXParser
│   ├── OptimizationProfile
│   ├── OutputAllocator
│   ├── Profiler
│   ├── ProfilingVerbosity
│   ├── Refit
│   ├── Runtime
│   ├── TacticSource
│   ├── Tensor
│   └── TimingCache
├── 03-BuildEngineByTensorRTAPI
│   ├── MNISTExample-Paddlepaddle
│   ├── MNISTExample-pyTorch
│   │   └── C++
│   ├── MNISTExample-TensorFlow1
│   ├── MNISTExample-TensorFlow2
│   ├── TypicalAPI-Paddlepaddle
│   ├── TypicalAPI-pyTorch
│   ├── TypicalAPI-TensorFlow1
│   └── TypicalAPI-TensorFlow2
├── 04-BuildEngineByONNXParser
│   ├── Paddlepaddle-ONNX-TensorRT
│   ├── pyTorch-ONNX-TensorRT
│   │   └── C++
│   ├── pyTorch-ONNX-TensorRT-QAT
│   ├── TensorFlow1-Caffe-TensorRT
│   ├── TensorFlow1-ONNX-TensorRT
│   ├── TensorFlow1-ONNX-TensorRT-QAT
│   ├── TensorFlow1-UFF-TensorRT
│   ├── TensorFlow2-ONNX-TensorRT
│   └── TensorFlow2-ONNX-TensorRT-QAT-TODO
├── 05-Plugin
│   ├── API
│   ├── C++-UsePluginInside
│   ├── C++-UsePluginOutside
│   ├── LoadDataFromNpz
│   ├── MultipleVersion
│   ├── PluginProcess
│   ├── PluginReposity
│   │   ├── AddScalarPlugin-TRT8
│   │   ├── BatchedNMS_TRTPlugin-TRT8
│   │   ├── CCLPlugin-TRT6-StaticShape
│   │   ├── CCLPlugin-TRT7-DynamicShape
│   │   ├── CumSumPlugin-V2.1-TRT8
│   │   ├── GruPlugin
│   │   ├── LayerNormPlugin-TRT8
│   │   ├── Mask2DPlugin
│   │   ├── MaskPugin
│   │   ├── MaxPlugin
│   │   ├── MMTPlugin
│   │   ├── MultinomialDistributionPlugin-cuRAND-TRT8
│   │   ├── MultinomialDistributionPlugin-thrust-TRT8
│   │   ├── OneHotPlugin-TRT8
│   │   ├── ReducePlugin
│   │   ├── Resize2DPlugin-TRT8
│   │   ├── ReversePlugin
│   │   ├── SignPlugin
│   │   ├── SortPlugin-V0.0-useCubAlone
│   │   ├── SortPlugin-V1.0-float
│   │   ├── SortPlugin-V2.0-float4
│   │   ├── TopKAveragePlugin
│   │   └── WherePlugin
│   ├── PluginSerialize-TODO
│   ├── PythonPlugin
│   │   └── circ_plugin_cpp
│   ├── UseCuBLAS
│   ├── UseFP16
│   ├── UseINT8-PTQ
│   ├── UseINT8-QDQ
│   ├── UseONNXParserAndPlugin-pyTorch
│   ├── UsePluginV2DynamicExt
│   ├── UsePluginV2Ext
│   └── UsePluginV2IOExt
├── 06-UseFrameworkTRT
│   ├── TensorFlow1-TFTRT
│   ├── TensorFlow2-TFTRT
│   └── Torch-TensorRT
├── 07-Tool
│   ├── FP16FineTuning
│   ├── Netron
│   ├── NetworkInspector
│   │   └── C++
│   ├── NetworkPrinter
│   ├── NsightSystems
│   ├── nvtx
│   ├── OnnxGraphSurgeon
│   │   └── API
│   ├── Onnxruntime
│   ├── Polygraphy-API
│   ├── Polygraphy-CLI
│   │   ├── convertExample
│   │   ├── dataExample
│   │   ├── debugExample
│   │   ├── HelpInformation
│   │   ├── inspectExample
│   │   ├── runExample
│   │   ├── surgeonExample
│   │   └── templateExample
│   ├── trex
│   │   ├── model
│   │   ├── trex
│   │   └── trex.egg-info
│   └── trtexec
├── 08-Advance
│   ├── BuilderOptimizationLevel
│   ├── CreateExecutionContextWithoutDeviceMemory
│   ├── C++StaticCompilation
│   ├── CudaGraph
│   ├── DataFormat
│   ├── DynamicShapeOutput
│   ├── EmptyTensor
│   ├── Event
│   ├── ExternalSource
│   ├── HardwareCompatibility
│   ├── LabeledDimension
│   ├── MultiContext
│   ├── MultiOptimizationProfile
│   ├── MultiStream
│   ├── Safety-TODO
│   ├── Sparsity
│   │   └── pyTorch-ONNX-TensorRT-ASP
│   ├── StreamAndAsync
│   ├── StrictType
│   ├── TensorRTGraphSurgeon
│   ├── TorchOperation
│   └── VersionCompatibility
├── 09-BestPractice
│   ├── AdjustReduceLayer
│   ├── AlignSize
│   ├── ComputationInAdvance
│   │   └── Convert3DMMTo2DMM
│   ├── ConvertTranposeMultiplicationToConvolution
│   ├── EliminateSqueezeUnsqueezeTranspose
│   ├── IncreaseBatchSize
│   ├── UsingMultiOptimizationProfile
│   ├── UsingMultiStream
│   └── WorkFlowOnONNXModel
├── 10-ProblemSolving
│   ├── ParameterCheckFailed
│   ├── SliceNodeWithBoolIO
│   ├── WeightsAreNotPermittedSinceTheyAreOfTypeInt32
│   └── WeightsHasCountXButYWasExpected
├── 50-Resource
│   └── 02-API-Layer
├── 51-Uncategorized
├── 52-Deprecated
│   ├── BindingEliminate-TRT8
│   ├── ConcatenationLayerBUG-TRT8.4
│   ├── ErrorWhenParsePadNode-TRT-8.4
│   ├── FullyConnectedLayer-TRT8.4
│   ├── FullyConnectedLayerWhenUsingParserTRT-8.4
│   ├── MatrixMultiplyDeprecatedLayer-TRT8
│   ├── max_workspace_size-TRT8.4
│   ├── MultiContext-TRT8
│   ├── ResizeLayer-TRT8
│   ├── RNNLayer-TRT8
│   └── ShapeLayer-TRT8
├── 99-NotFinish
│   └── TensorRTElementwiseBug
└── include
```
