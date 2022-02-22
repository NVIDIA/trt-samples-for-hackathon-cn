# TensorRT Cookbook in Chinese

## 03-APIModel —— TensorRT API 搭建模型样例
+ DynamicShape+Shuffle，使用 TensorRT 的 Dynamic shape 模式时一种常见的、需要 shuffle 的情景
+ MNISTExample，一个完整的、基于 MNIST 数据集的、手写数字识别模型的示例，该模型在 TensorFlow 中训练好之后在 TensotRT 中重建并推理
+ TensorFlow，TensorFlow 中卷积、全连接、LSTM结构的不同实现在 TensorRT 中重建的样例
+ pyTorch，pyTorch 中各种结构的不同实现在 TensorRT 中重建的样例
+ Paddlepaddle，Paddlepaddle 中各种结构的不同实现在 TensorRT 中重建的样例

### DynamicShape+Shuffle
+ 环境：nvcr.io/nvidia/tensorrt:21.12-py3（包含 python 3.8.10，CUDA 11.5.50，cuDNN 8.3.1，TensoRT 8.2.3）
+ 运行方法
```python
cd ./DynamicShape+FCLayer
python dynamicShape+FCLayer.py
```
+ 参考输出结果，见 ./DynamicShape+FCLayer/result.txt

### MNISTExample
+ 环境：nvcr.io/nvidia/tensorflow:21.10-tf1-py3（包含 python 3.8.10，CUDA 11.4.2，cuBLAS 11.6.5.2，cuDNN 8.2.4.15，TensoFlow 1.15.5，TensorRT 8.0.3.4）
+ 运行方法
```python
cd ./MNISTExample
pip install -r requirments.txt
python MNISTExample.py
```
+ 参考输出结果，见 ./MNISTExample/result.txt

### TensorFlow
+ 环境：nvcr.io/nvidia/tensorflow:21.10-tf1-py3（包含 python 3.8.10, CUDA 11.4.2, cuBLAS 11.6.5.2, cuDNN 8.2.4.15, TensoFlow 1.15.5, TensorRT 8.0.3.4）
+ 运行方法
```python
cd ./TensoFlowToTensorRT
python Convolution.py
python FullyConnected.py
python RNN-LSTM.py.py
```
+ 参考输出结果，见 ./TensorFlowToTensorRT/result-*.txt

### pyTorch
+ TODO

### Paddlepaddle
+ TODO

