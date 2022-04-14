# TensorRT Cookbook in Chinese

## 06-PluginAndParser —— 结合使用 Parser 与 Plugin 的样例
+ 需要用到 onnx-graphsurgeon，可以参考 Cookbook 的 08-Tool/onnxGraphSurgeon
+ TensorFlow-addScalar，在 TensorFlow 转 onnx 转 TensorRT 的过程中替换一个算子为我们自己实现的 Plugin（05-Plugin 中的 AddScalar）
+ pyTorch-LayerNorm，在 pyTorch 转 onnx 转 TensorRT 的过程中替换一个模块（LayerNorm）以提高效率
+ pyTorch-FailConvertNonZero，在 pyTorch 转 onnx 转 TensorRT 的过程中转换 NonZero 节点失败的例子（TensorRT 不原生支持该算子）

### TensorFlow-AddScalar
+ 环境：nvcr.io/nvidia/tensorflow:21.10-tf1-py3（包含 python 3.8.10, CUDA 11.4.2, cuBLAS 11.6.5.2, cuDNN 8.2.4.15, TensoFlow 1.15.5, TensorRT 8.0.3.4）
+ 运行方法
```python
cd ./TensorFlow-addScalar
pip install -r requirments.txt
python TensorFlowToTensorRT.py
```
+ 参考输出结果，见 ./TensorFlow-addScalar/result.txt

### 02-pyTorch-LayerNorm
+ 环境：nvcr.io/nvidia/pytorch:21.10-py3（包含 python 3.8.12，CUDA 11.4.2，cuBLAS 11.6.5.2，cuDNN 8.2.4.15，pyTorch 1.10.0a0+0aef44c，TensorRT 8.0.3.4）
+ 运行方法
```python
cd ./pyTorch-LayerNorm
pip install -r requirments.txt
python pyTorchToTensorRT.py
```
+ 参考输出结果，见 ./pyTorch-LayerNorm/result.txt

### pyTorch-FailConvertNonZero
+ 环境：nvcr.io/nvidia/pytorch:21.10-py3（包含 python 3.8.12，CUDA 11.4.2，cuBLAS 11.6.5.2，cuDNN 8.2.4.15，pyTorch 1.10.0a0+0aef44c，TensorRT 8.0.3.4）
+ 运行方法
```python
cd ./pyTorch-FailConvertNonZero
pip install -r requirments.txt
python pyTorchToTensorRT.py
```
+ 参考输出结果，见 ./pyTorch-FailConvertNonZero/result.txt

