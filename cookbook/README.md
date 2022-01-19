# TensorRT Cookbook in Chinese【alpha版】

## 00-MNISTData —— 用到的数据
+ 示例项目使用的 MNIST 数据，可以从[这里](http://yann.lecun.com/exdb/mnist/)或者[这里](https://storage.googleapis.com/cvdf-datasets/mnist/)下载
+ 数据下载后放进该目录，形如 `./00-MNISTData/*.gz`，一共 4 个文件
+ 需要提取部分图片到 `./train` 和 `./test` 中备用（可以用 extractMnistData.py 提取，提取出来的图片是 .jpg 格式）
+ 该目录下有一张 8.png 为提取出来的图片示例，单独作为 TensorRT 的输入数据使用

### 运行方法
```python 
cd ./00-MNISTData
pip install -r ./requirements.txt
python extractMnistData.py XXX YYY # 提取 XXX（默认值 600）张训练图，YYY（默认值 100）张测试图
```

---
## 01-APISimple —— TensorRT API 的简单示例
+ 使用 TensorRT 的基本操作，包括网络搭建、引擎生成、序列化与反序列化、数据拷贝、推理计算
+ 示例涵盖 TensorRT6（子目录 01-TensorRT6），TensorRT7（子目录 02-TensorRT7）和 TensorRT8（子目录 03-TensorRT8）版本（部分 API 有差异），包含 C++ 和 python 的等价实现
+ 01-TensorRT6，采用 TensorRT6 + implicit batch 模式 + static shape 模式 + builder API，使用较旧的 pycuda 库
+ 02-TensorRT7，采用 TensorRT7 + implicit batch 模式 + static shape 模式 + builder API，此后均使用较新的 cuda-python 库
+ 03-TensorRT8，采用 TensorRT8 + explicit batch 模式 + dynamic shape 模式 + builderConfig API（推荐使用，灵活性和应用场景更广）

### 01-TensorRT6
+ 环境：nvcr.io/nvidia/tensorrt:19.12-py3（包含 python 3.6, CUDA 10.2.89, cuDNN 7.6.5, TensoRT 6.0.1）
+ 运行方法
```shell
cd ./01-APISimple/01-TensorRT6
pip install -r ./requirements.txt
# 修改 Makefile 中的 SM 等参数
make
make test
```
+ 输出结果，见 ./01-APISimple/01-TensorRT6/result.txt

### 02-TensorRT7
+ 环境：nvcr.io/nvidia/tensorrt:21.06-py3（包含 python 3.8.5, CUDA 11.3.1, cuDNN 8.2.1, TensoRT 7.2.3.4）
+ 运行方法
```shell
cd ./01-APISimple/02-TensorRT7
pip install -r ./requirements.txt
# 修改 Makefile 中的 SM 等参数
make
make test
```
+ 输出结果，见 ./01-APISimple/02-TensorRT7/result.txt

### 03-TensorRT8
+ 环境：nvcr.io/nvidia/tensorrt:21.09-py3（包含 python 3.8.5, CUDA 11.3.1, cuDNN 8.2.1, **TensoRT 8.0.3**）
    - 也可使用更新的 nvcr.io/nvidia/tensorrt:21.12-py3（包含 python 3.8.10, CUDA 11.5.50, cuDNN 8.3.1, **TensoRT 8.2.3**）
+ 运行方法
```shell
cd ./01-APISimple/03-TensorRT8
pip install -r ./requirements.txt
# 修改 Makefile 中的 SM 等参数
make
make test
```
+ 输出结果，见 ./01-APISimple/03-TensorRT8/result.txt

---
## 02-API —— TesnorRT API 用法示例
+ TensorRT 的各种 API 及其参数
+ 示例代码为 python 实现
+ 01-Layer，各种 Layer 的用法和示例，无特殊情况均采用 TensorRT8 + explicit batch 模式
+ 02-???[TODO]，其他 API 及其参数的使用方法

### 01-Layer
+ 详见各 ./02-API01-Layer/*.md

### 02-???[TODO]
+ TODO

---
## 03-APIModel —— TensorRT API 搭建模型样例
+ 在 TensorRT 中采用 API 搭建的方式重建来自各种 ML 框架中的模型的关键步骤，包括原模型权重提取，TensorRT 中典型层的搭建和权重加载
+ 主要包括卷积层、池化层、全连接层、循环神经网络层等常用层的重建
+ 01-TensoFlowToTensorRT，TensorFlow 常见层的重建
+ 02-pyTorchToTensorRT，pyTorch 常见层的重建
+ 03-PaddlepaddleToTensorRT，Paddlepaddle  常见层的重建
+ 04-DynamicShape+FCLayer，TensorRT 中 dynamic shape 模式 + fully Connected 层的一个例子，在许多模型中经常出现
+ 05-OCR，完整的 OCR 模型迁移到 TEnsorRT 的过程代码

### 01-TensorFlowToTrnsorRT[TODO]
+ TODO

### 02-pyTorchToTensorRT[TODO]
+ TODO

### 03-PaddlepaddleToTensorRT[TODO]
+ TODO

### 04-DynamicShape+FCLayer[TODO]
+ TODO

### 05-OCR[TODO]
+ TODO

---
## 04-Parser —— 使用 Parser 转换模型到 TensorRT 中的简单样例
+ 样例代码均以 MNIST 项目为例，使用 TensorRT8 版本 python 实现
+ 01-TF-ONNX-TensorRT，使用 TensorFlow (.pb) 转 ONNX (.onnx) 转 TensorRT (.trt)，并在 TensorRT 中使用 float16 模式
+ 02-pyTorch-ONNX-TensorRT，使用 pyTorch (.pt) 转 ONNX (.onnx) 转 TensorRT (.trt)，并在 TensorRT 中使用 int8 模式
+ 03-pyTorch-ONNX-TensorRT-Split，使用 pyTorch (.pt) 转 ONNX (.onnx) 转 TensorRT (.trt)，并将模型进行分割，在 TensorRT 中插入一些层后再缝合成一个模型[TODO]
+ 04-TensorFlowF-UFF-TensorRT，使用 TensorFlow (.pt) 转 UFF (.uff) 转 TensorRT (.trt)

### 01-TensorFlow-ONNX-TensorRT
+ 环境：nvcr.io/nvidia/tensorflow:21.10-tf1-py3（包含 python 3.8.10, CUDA 11.4.2, cuBLAS 11.6.5.2, cuDNN 8.2.4.15, TensoFlow 1.15.5, TensorRT 8.0.3.4）
+ 运行方法
```python
cd ./04-Parser/01-TensorFlow-ONNX-TensorRT
pip install -r requirments.txt
python TensorFlowToTensorRT.py
```
+ 输出结果，见 ./04-Parser/01-TensorFlowToTensorRT/result.txt

### 02-pyTorch-ONNX-TensorRT pyTorch 转 ONNX 转 TensorRT
+ 环境：nvcr.io/nvidia/pytorch:21.10-py3（包含 python 3.8.12, CUDA 11.4.2, cuBLAS 11.6.5.2, cuDNN 8.2.4.15, pyTorch 1.10.0a0+0aef44c, TensorRT 8.0.3.4）
+ 运行方法
```python
cd ./04-Parser/02-pyTorch-ONNX-TensorRT
pip install -r requirments.txt
python pyTorchToTensorRT.py
```
+ 输出结果，见 ./04-Parser/02-pyTorch-ONNX-TensorRT/result.txt

### 03-pyTorch-ONNX-TensorRT-Split[TODO]
+ TODO

### 04-TensorFlowF-UFF-TensorRT[TODO]
+ TODO

---
## 05-Plugin —— 书写 TensorRT Plugin 样例
+ 样例代码使用 TensorRT8 版本 python 实现
+ 01-usePluginV2Ext
    - 使用 IPluginV2Ext 类实现一个 plugin，功能是给输入张量所有元素加上同一个标量值，然后输出
    - 输入张量形状不可变（不同形状的输入需要建立不同的 TensorRT engine）
    - 标量加数在创建计算网络时确定（多次 inference 之间不能更改）
    - 支持序列化和反序列化
    - 支持 TensorRT7 和 TensorRT8 版本，但是需要修改 AddScalarPlugin.h 和 AddScalarPlugin.cu 中 enqueue 函数的声明和定义
+ 02-usePluginV2DynamicExt
    - 使用 IPluginV2DynamicExt 类实现一个 plugin，功能是给输入张量所有元素加上同一个标量值，然后输出
    - 输入张量形状可变（相同维度的输入可以使用同一个 TensorRT engine）
    - 标量加数在创建计算网络时确定（多次 inference 之间不能更改）
    - 支持序列化和反序列化
    - 支持 TensorRT7 和 TensorRT8 版本
+ 03-useCublasAndSaveWeight[TODO]
    - 在 plugin 中使用 cuBLAS 计算矩阵乘法，并在 plguin 之间利用指针共享乘数权重
+ 04-useFP16[TODO]
    - 在 plugin 中使用 float16
+ 05-useInt8[TODO]
    - 在 plugin 中使用 int8
+ 06-pluginReposity，常见 Plugin 小仓库[TODO]
    - https://gitlab-master.nvidia.com/wili/tensorrt-plugin

### 01-usePluginV2Ext
+ 环境：nvcr.io/nvidia/tensorrt:21.09-py3（包含 python 3.8.5, CUDA 11.3.1, cuDNN 8.2.1, TensoRT 8.0.3）
+ 运行方法
```shell
cd ./05-Plugin/01-usePluginV2Ext
# 修改 Makefile 中的 SM 等参数
make
make test
```
+ 输出结果，见 ./05-Plugin/01-usePluginV2Ext/result.txt

### 02-usePluginV2DynamicExt
+ 环境：nvcr.io/nvidia/tensorrt:21.09-py3（包含 python 3.8.5, CUDA 11.3.1, cuDNN 8.2.1, TensoRT 8.0.3）
+ 运行方法
```shell
cd ./05-Plugin/02-usePluginV2DynamicExt
# 修改 Makefile 中的 SM 等参数
make
make test
```
+ 输出结果，见 ./05-Plugin/02-usePluginV2DynamicExt/result.txt

### 03-useCublasAndSaveWeight[TODO]
+ TODO

### 04-useFP16[TODO]
+ TODO

### 05-useInt8[TODO]
+ TODO

### 06-pluginReposity[TODO]
+ TODO

---
## 06-PluginAndParser —— 结合使用 Parser 与 Plugin 的样例
+ onnx-gaphsurgeon 的基本操作示例（调整网络输入输出，添加删除节点，添加删除输入输出张量，自定义节点）
+ 一个 pyTorch-ONNX-TensorRT 的例子，手动替换一个 op 为 AddOnePlugin
+ 01-onnx-graphsurgeon[TODO]，onnx-graphsurgeon 基本操作
+ 02-pyTorch-ONNX-TensorRT[TODO]，导出 pyTorch 模型的过程中插入一个 TensorRT 不能原生支持的算子（05-Plugin 中的 AddScalar）

### 01-onnx-graphsurgeon[TODO]
+ TODO

### 02-pyTorch-ONNX-TensorRT[TODO]
+ TODO

---
## 07-Advance —— 其他话题
+ Refit 例子
+ trtexec 的使用例子
+ 多 context 例子（之前 dota2 模型的例子）
+ Throughput 测量
+ TRTLite
+ QAT int8 例子
+ plugin 中各接口函数解读
+ EinsumLayer 例子
+ Nsight system 与优化示例（用之前 dota2 模型中的几个小 plugin 优化例子）
+ polygraphy

+ 空张量






