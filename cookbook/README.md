# TensorRT Cookbook in Chinese

+ TensorRT 文档 [link](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
+ TensorRT C++ API [link](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api)
+ TensorRT Python API [link](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)
+ TensorRT 版本支持列表 [link](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html)

## 00-MNISTData —— 用到的数据
+ Cookbook 用到的 MNIST 数据集，运行其他示例代码前，需要下载到本地并做一些预处理

---
## 01-SimpleDemo
+ 使用 TensorRT 的基本步骤，包括网络搭建、引擎生成、序列化与反序列化、推理计算
+ 示例涵盖 TensorRT6，TensorRT7，TensorRT8（部分 API 有差异），均包含 C++ 和 python 的等价实现

---
## 02-API
+ TensorRT 各 API 的用法，使用 TensorRT8，python 代码实现

---
## 03-APIModel
+ 采用 TensorRT API 搭建方式重建来自各种 ML 框架中模型的关键步骤，包括原模型权重提取，TensorRT 中典型层的搭建和权重加载
+ 包含一个完整的、基于 MNIST 数据集的、手写数字识别模型的示例，该模型在 TensorFlow 中训练好之后在 TensotRT 中重建并推理

---
## 04-Parser
+ 不同 ML 框架中训练好的模型使用 Parser 迁移到 TensorRT 中并推理的样例
+ 样例以基于 MNIST 数据集的、手写数字识别模型为例，使用 TensorRT8，python 代码实现

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
+ 一些开发辅助工具的使用范例，包括 onnx-graphsurgeon，nsight system，trtexec，Polygraphy

---
## 09-Advance
+ 一些 TensorRT 高级用法

---
## 10-BestPractice
+ 一些有趣的 TensorRT 优化样例

---
## 11-Uncategorized
+ 未分类的一些东西
+ Throughput 测量
+ TRTLite
+ QAT int8 例子
+ 空张量


