# TensorRT Cookbook in Chinese

## 总体介绍

+ 下列各章节的目录内包含了该章节的详细 README.md

+ 初次使用，请在起 docker container 之后参考 requirements.txt 自选安装需要的库 `pip install -r requirement.txt`

+ 有用的参考文档：

  + TensorRT 下载 [link](https://developer.nvidia.com/nvidia-tensorrt-download)

  + TensorRT 版本变化 [link](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html)

  + TensorRT 文档 [link](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)

  + TensorRT 归档文档 [link](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html)

  + TensorRT 特性支持列表 [link](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html)

  + TensorRT C++ API [link](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api)

  + TensorRT Python API [link](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)

## 注意事项

+ 【**2022.7.15**】Cookbook 内容随 TensorRT 更新到 8.4 GA 版本，部分使用最新版中才有的 API，同学们使用较旧版本的 TensorRT 运行 Cookbook 中的代码时，可能需要修改部分代码才能运行，例如：
  + `config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)` 改回 `config.max_workspace_size = 1 << 30`

+ 【**2022.8.4**】常用的 docker image
  + TensorRT6：**nvcr.io/nvidia/tensorrt:19.12-py3**（包含 python 3.6，CUDA 10.2.89，cuDNN 7.6.5，TensoRT 6.0.1）
  + TensorRT7：**nvcr.io/nvidia/tensorrt:21.06-py3**（包含 python 3.8.5，CUDA 11.3.1，cuDNN 8.2.1，TensoRT 7.2.3.4）
  + TensorRT8.2：**nvcr.io/nvidia/tensorrt:22.04-py3**（包含 python 3.8.10，CUDA 11.6.2，cuDNN 8.4.0.27，TensoRT 8.2.4.2）
  + TensorRT8.4：**nvcr.io/nvidia/tensorrt:22.07-py3**（包含 python 3.8.13，CUDA 11.7U1，cuDNN 8.4.1，TensoRT 8.4.1）
  + TensorFlow1：**nvcr.io/nvidia/tensorflow:22.07-tf1-py3**（包含 python 3.8.13，CUDA 11.7U1，cuDNN 8.4.1，TensoRT 8.4.1，TensorFlow 1.15.5）
  + TensorFlow2：**nvcr.io/nvidia/tensorflow:22.07-tf2-py3**（包含 python 3.8.13，CUDA 11.7U1，cuDNN 8.4.1，TensoRT 8.4.1，TensorFlow 2.9.1）
  + pyTorch：**nvcr.io/nvidia/pytorch:22.07-py3**（包含 python 3.8.13，CUDA 11.7U1，cuDNN 8.4.1，TensoRT 8.4.1，1.13.0a0+08820cb）
  + PaddlePaddle：**nvcr.io/nvidia/paddlepaddle:22.07-py3**（包含 python 3.8.13，CUDA 11.7U1，cuDNN 8.4.1，TensoRT 8.4.1，PaddlePaddle 2.3.0）

  + 推荐配置方法：使用 ```nvcr.io/nvidia/pytorch:22.07-py3```，然后依照 requirements.txt 在其中安装 TensorFlow2 等其他库，因为该 docker image 中的 pyTorch 包含部分改动（尤其是 QAT 相关内容），与单独 ```pip install torch``` 的效果不同

+ 【**2022.8.10**】添加 ```testAllExceptTF1.sh``` 和 ```testTF1.sh``` 用于运行所有范例代码并生成相应的输出结果
  + 可在包含 pyTorch 和 TF2 的环境中运行 testAllExceptTF1.sh，执行除了 TensorFlow1 以外的所有范例代码
  + 可在包含 TF1 的环境中运行 testTF1.sh，执行 TensorFlow1 相关的所有范例代码

---

## 00-MNISTData

+ Cookbook 用到的 MNIST 数据集，运行其他示例代码前，需要下载到本地并做一些预处理

---

## 01-SimpleDemo

+ 使用 TensorRT 的基本步骤，包括网络搭建、引擎构建、序列化与反序列化、推理计算
+ 示例涵盖 TensorRT6，TensorRT7，TensorRT8（部分 API 有差异），均包含 C++ 和 python 的等价实现

---

## 02-API

+ TensorRT 各 API 的用法，包括各层详细用法、打印网络信息、Dynamicshape 模式的 Shape 操作示例、Int8-QDQ 网络

---

## 03-APIModel

+ 采用 TensorRT API 搭建方式重建来自各种 ML 框架中模型的关键步骤，包括原模型权重提取，TensorRT 中典型层的搭建和权重加载
+ 一个完整的、基于 MNIST 数据集的、手写数字识别模型的示例，该模型在 TensorFlow / pyTorch 中训练好之后在 TensotRT 中重建并推理

---

## 04-Parser

+ 不同 ML 框架中训练好的模型使用 Parser 迁移到 TensorRT 中并推理的样例
+ 示例以基于 MNIST 数据集的、手写数字识别模型为例，使用 TensorRT8 python 代码实现

---

## 05-Plugin

+ 实现 Plugin 并运用到 TensorRT 推理中的样例，使用 TensorRT8 版本 python 实现

---

## 06-PluginAndParser

+ 结合使用 Paser 和 Plugin 来转化模型并在 TensorRT 中推理

---

## 07-FameworkTRT

+ 使用各 ML 框架内置的接口来使用 TensorRT 的样例

---

## 08-Tool

+ 开发辅助工具的使用示例，包括 Netron，onnx-graphsurgeon，nsight system，trtexec，Polygraphy，trex

---

## 09-Advance

+ TensorRT 高级用法

---

## 10-BestPractice

+ 有趣的 TensorRT 优化样例

---

## 11-ProblemSolving

+ 在将模型部署到 TensorRT 上时常见的报错信息及其解决办法

---

## 50-Resource

+ TensorRT 教程的 .pptx 或对应的 .pdf，以及一些其他有用的参考资料

---

## 51-Uncategorized

+ 未分类的一些东西

---

## 52-Deprecated

+ 旧版本 TensorRT 中的一些 API 和用法，他们在较新版本 TensorRT 中已被废弃，直接使用会报错退出

---

## 99-NotFinish

+ 施工中……没有完成的范例代码，以及同学们提出的新的范例需求
