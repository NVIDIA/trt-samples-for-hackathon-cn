# TensorRT Cookbook in Chinese

## 03-APIModel —— TensorRT API 搭建模型样例

### MNISTExample-pyTorch
+ 一个完整的、基于 MNIST 数据集的、手写数字识别模型的示例，该模型在 pyTorch 中训练好之后在 TensotRT 中重建并推理
+ 运行方法
```shell
cd ./MNISTExample-pyTorch
python3 main.py
```
+ 参考输出结果，见 ./MNISTExample-pyTorch/result.md

### MNISTExample-TensorFlow1
+ 一个完整的、基于 MNIST 数据集的、手写数字识别模型的示例，该模型在 TensorFlow1 中训练好之后在 TensotRT 中重建并推理
+ 运行方法
```shell
cd ./MNISTExample-TensorFlow1
python3 main.py
```
+ 参考输出结果，见 ./MNISTExample-TensorFlow1/result.md

### MNISTExample-TensorFlow2
+ 一个完整的、基于 MNIST 数据集的、手写数字识别模型的示例，该模型在 TensorFlow2 中训练好之后在 TensotRT 中重建并推理
+ 运行方法
```shell
cd ./MNISTExample-TensorFlow2
python3 main.py
```
+ 参考输出结果，见 ./MNISTExample-TensorFlow2/result.md

### PaddlePaddle
+ PaddlePaddle 中各种结构的不同实现在 TensorRT 中重建的样例
+ **TODO**

### pyTorch
+ pyTorch 中各种结构的不同实现在 TensorRT 中重建的样例
+ **TODO**

### TensorFlow1
+ TensorFlow1 中卷积、全连接、LSTM 结构的不同实现在 TensorRT 中重建的样例
+ 运行方法
```shell
cd ./TensoFlow1
python3 Convolution.py
python3 FullyConnected.py
python3 RNN-LSTM.py.py
```
+ 参考输出结果，见 ./TensorFlow/result-*.md

