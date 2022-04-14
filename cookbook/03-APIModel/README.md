# TensorRT Cookbook in Chinese

## 03-APIModel —— TensorRT API 搭建模型样例
+ MNISTExample，一个完整的、基于 MNIST 数据集的、手写数字识别模型的示例，该模型在 TensorFlow 中训练好之后在 TensotRT 中重建并推理
+ Paddlepaddle，Paddlepaddle 中各种结构的不同实现在 TensorRT 中重建的样例
+ pyTorch，pyTorch 中各种结构的不同实现在 TensorRT 中重建的样例
+ TensorFlow，TensorFlow 中卷积、全连接、LSTM结构的不同实现在 TensorRT 中重建的样例

### MNISTExample
+ 环境：nvcr.io/nvidia/tensorflow:21.10-tf1-py3（包含 python 3.8.10，CUDA 11.4.2，cuBLAS 11.6.5.2，cuDNN 8.2.4.15，TensoFlow 1.15.5，TensorRT 8.0.3.4）
+ 运行方法
```python
cd ./MNISTExample
pip install -r requirments.txt
python MNISTExample.py
```
+ 参考输出结果，见 ./MNISTExample/result.txt

### Paddlepaddle
+ TODO

### pyTorch
+ TODO

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

