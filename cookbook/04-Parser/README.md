# TensorRT Cookbook in Chinese

## 04-Parser —— 使用 Parser 转换模型到 TensorRT 中的简单样例
+ pyTorch-ONNX-TensorRT，使用 pyTorch (.pt) 转 ONNX (.onnx) 转 TensorRT (.plan)，并在 TensorRT 中使用 int8 模式
+ TF-ONNX-TensorRT，使用 TensorFlow (.pb) 转 ONNX (.onnx) 转 TensorRT (.plan)，并在 TensorRT 中使用 float16 模式
+ TensorFlowF-UFF-TensorRT，使用 TensorFlow (.pt) 转 UFF (.uff) 转 TensorRT (.plan)，UFF 已废弃，仅做参考

+ Onnx 的算子说明 [link](https://github.com/onnx/onnx/blob/main/docs/Operators.md)

### pyTorch-ONNX-TensorRT
+ 环境：nvcr.io/nvidia/pytorch:21.10-py3（包含 python 3.8.12，CUDA 11.4.2，cuBLAS 11.6.5.2，cuDNN 8.2.4.15，pyTorch 1.10.0a0+0aef44c，TensorRT 8.0.3.4）
+ 运行方法
```python
cd ./pyTorch-ONNX-TensorRT
pip install -r requirments.txt
python pyTorchToTensorRT.py
```
+ 参考输出结果，见 ./pyTorch-ONNX-TensorRT/result.txt

### TensorFlow-ONNX-TensorRT
+ 环境：nvcr.io/nvidia/tensorflow:21.10-tf1-py3（包含 python 3.8.10，CUDA 11.4.2，cuBLAS 11.6.5.2，cuDNN 8.2.4.15，TensoFlow 1.15.5，TensorRT 8.0.3.4）
+ 运行方法
```python
cd ./TensorFlow-ONNX-TensorRT
pip install -r requirments.txt
python TensorFlowToTensorRT-NHWC.py
python TensorFlowToTensorRT-NHWC(C=2).py
python TensorFlowToTensorRT-NCHW.py
```
+ 参考输出结果，见 ./TensorFlow-ONNX-TensorRT/result.txt

### TensorFlowF-UFF-TensorRT
+ 环境：nvcr.io/nvidia/tensorflow:21.10-tf1-py3（包含 python 3.8.10，CUDA 11.4.2，cuBLAS 11.6.5.2，cuDNN 8.2.4.15，TensoFlow 1.15.5，TensorRT 8.0.3.4）
+ 运行方法
```python
cd ./TensorFlow-UFF-TensorRT
pip install -r requirments.txt
python TensorFlowToTensorRT.py
```
+ 参考输出结果，见 ./TensorFlow-UFF-TensorRT/result.txt

