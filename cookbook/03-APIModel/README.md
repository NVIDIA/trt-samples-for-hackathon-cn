# TensorRT Cookbook in Chinese

## 03-APIModel —— TensorRT API 搭建模型样例

### MNISTExample
+ 一个完整的、基于 MNIST 数据集的、手写数字识别模型的示例，该模型在 TensorFlow 中训练好之后在 TensotRT 中重建并推理
+ 环境：nvcr.io/nvidia/tensorflow:21.12-tf1-py3（包含 python 3.8.10，CUDA 11.5.0，cuBLAS 11.7.3.1，cuDNN 8.3.1.22，TensoFlow 1.15.5，TensorRT 8.2.1.8）
+ 运行方法
```shell
cd ./MNISTExample
pip install -r requirements.txt
python MNISTExample.py
```
+ 参考输出结果，见 ./MNISTExample/result.txt

### Paddlepaddle
+ Paddlepaddle 中各种结构的不同实现在 TensorRT 中重建的样例
+ **TODO**

### pyTorch
+ pyTorch 中各种结构的不同实现在 TensorRT 中重建的样例
+ **TODO**

### TensorFlow
+ TensorFlow 中卷积、全连接、LSTM 结构的不同实现在 TensorRT 中重建的样例
+ 环境：nvcr.io/nvidia/tensorflow:21.12-tf1-py3（包含 python 3.8.10，CUDA 11.5.0，cuBLAS 11.7.3.1，cuDNN 8.3.1.22，TensoFlow 1.15.5，TensorRT 8.2.1.8）
+ 运行方法
```shell
cd ./TensoFlowToTensorRT
python Convolution.py
python FullyConnected.py
python RNN-LSTM.py.py
```
+ 参考输出结果，见 ./TensorFlow/result-*.txt

