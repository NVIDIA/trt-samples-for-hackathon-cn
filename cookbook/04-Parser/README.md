# TensorRT Cookbook in Chinese

## 04-Parser —— 使用 Parser 转换模型到 TensorRT 中的简单样例
+ Onnx 的算子说明 [link](https://github.com/onnx/onnx/blob/main/docs/Operators.md)

### pyTorch-ONNX-TensorRT
+ .pt 转 .onnx 转 .plan，并在 TensorRT 中使用 int8 模式
+ 环境：nvcr.io/nvidia/pytorch:21.12-py3（包含 python 3.8.10，CUDA 11.5.0，cuBLAS 11.7.3.1，cuDNN 8.3.1.22，pyTorch 1.11.0a0+b6df043，TensorRT 8.2.1.8）
+ 运行方法
```shell
cd ./pyTorch-ONNX-TensorRT
python3 pyTorchToTensorRT.py
```
+ 参考输出结果，见 ./pyTorch-ONNX-TensorRT/result.txt

### pyTorch-ONNX-TensorRT-QAT
+ 量化感知训练的 .pt 转 .onnx 转 .plan，并在 TensorRT 中使用 int8 模式
+ 参考 https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization
+ 原始例子中，校正和精调要跑在 nvcr.io/nvidia/pytorch:20.08-py3 上，导出 .onnx 的部分要跑在 21.12-py3 及之后。因为两个 image 中 /opt/pytorch/vision/references/classification/train.py 文件发生了较大变动，代码依赖该文件但是旧版 image 的 torch.onnx 不支持 QAT 导出
+ 本例子使用了完全本地化的模型，移除了上述依赖，可以独立运行
+ 需要 pytorch-quantization 库
+ 环境：nvcr.io/nvidia/pytorch:21.12-py3（包含 python 3.8.10，CUDA 11.5.0，cuBLAS 11.7.3.1，cuDNN 8.3.1.22，pyTorch 1.11.0a0+b6df043，TensorRT 8.2.1.8）
+ 运行方法
```shell
cd ./pyTorch-ONNX-TensorRT-QAT
python3 pyTorchToTensorRT.py
```
+ 参考输出结果，见 ./pyTorch-ONNX-TensorRT/result.txt

### pyTorch-PadNode
+ 在 pyTorch 转 onnx 转 TensorRT 的过程中，pyTorch 的 Pad 节点不能被 TensorRT 正确解析
+ Workaround：在 pyTorch 中将该 Pad 节点替换为 interpolate 节点，然后在 TensorRT 中解析后替换回 Slice 层，从而实现原本的功能
+ 环境：nvcr.io/nvidia/pytorch:21.12-py3（包含 python 3.8.10，CUDA 11.5.0，cuBLAS 11.7.3.1，cuDNN 8.3.1.22，pyTorch 1.11.0a0+b6df043，TensorRT 8.2.1.8）
+ 运行方法
```shell
cd ./pyTorch-PadNode
python3 padNode.py
```
+ 参考输出结果，见 ./pyTorch-PadNode/result.txt

### TensorFlowF-Caffe-TensorRT
+ .ckpt 转 .prototxt/.caffemodel 转 .plan，**该 Workflow 已废弃，本示例仅做参考**
+ 环境：使用 conda 搭建环境，包含 python 3.8.10，CUDA 11.5.0，cuBLAS 11.7.3.1，cuDNN 8.3.1.22，TensorRT 8.2.1.8
+ 运行方法
```shell
cd ./TensorFlow-Caffe-TensorRT
conda install caffe # pip install 装不了
python3 buildModelInTensorFlow.py
mmconvert -sf tensorflow -in ./model.ckpt.meta -iw ./model.ckpt --dstNodeName y -df caffe -om model
# 修改 model.prototxt 第 91 行附近，“dim: 1 dim: 3136”之间插入两行“dim: 1”，变成“dim: 1 dim: 1 dim: 1 dim: 3136”（不添加或者只添加一行的报错见 result-Dim2.txt 和 result-Dim3.txt）
python3 runModelInTensorRT.py
```
+ 参考输出结果，见 ./TensorFlow-Caffe-TensorRT/result.txt

### TensorFlow-ONNX-TensorRT
+ .pb 转 .onnx 转 .plan，并在 TensorRT 中使用 float16 模式
+ 环境：nvcr.io/nvidia/tensorflow:21.12-tf1-py3（包含 python 3.8.10，CUDA 11.5.0，cuBLAS 11.7.3.1，cuDNN 8.3.1.22，TensoFlow 1.15.5，TensorRT 8.2.1.8）
+ 运行方法
```shell
cd ./TensorFlow-ONNX-TensorRT
python3 TensorFlowToTensorRT-NHWC.py
python3 TensorFlowToTensorRT-NHWC(C=2).py
python3 TensorFlowToTensorRT-NCHW.py
```
+ 参考输出结果，见 ./TensorFlow-ONNX-TensorRT/result.txt

### TensorFlow-ONNX-TensorRT-QAT
+ 量化感知训练的 .pb 转 .onnx 转 .plan，并在 TensorRT 中使用 int8 模式
+ 参考 https://github.com/shiyongming/QAT_demo
+ 环境：nvcr.io/nvidia/tensorflow:21.12-tf1-py3（包含 python 3.8.10，CUDA 11.5.0，cuBLAS 11.7.3.1，cuDNN 8.3.1.22，TensoFlow 1.15.5，TensorRT 8.2.1.8）
+ 运行方法
```shell
cd ./TensorFlow-ONNX-TensorRT-QAT

python3 TensorFlowToTensorRT-QAT.py
```
+ 参考输出结果，见 ./TensorFlow-ONNX-TensorRT-QAT/result.txt

### TensorFlowF-UFF-TensorRT
+ .pb 转 .uff 转 .plan，**该 Workflow 已废弃，本示例仅做参考**
+ 环境：nvcr.io/nvidia/tensorflow:21.12-tf1-py3（包含 python 3.8.10，CUDA 11.5.0，cuBLAS 11.7.3.1，cuDNN 8.3.1.22，TensoFlow 1.15.5，TensorRT 8.2.1.8）
+ 运行方法
```shell
cd ./TensorFlow-UFF-TensorRT
python3 TensorFlowToTensorRT.py
```
+ 参考输出结果，见 ./TensorFlow-UFF-TensorRT/result.txt

