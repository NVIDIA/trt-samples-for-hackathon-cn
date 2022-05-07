# TensorRT Cookbook in Chinese

## 07-FameworkTRT —— 使用 ML 框架内置的接口来使用 TensorRT

### TFTRT
+ 使用 TFTRT 来运行一个训练好的 TF 模型
+ 环境：nvcr.io/nvidia/tensorflow:21.12-tf1-py3（包含 python 3.8.10，CUDA 11.5.0，cuBLAS 11.7.3.1，cuDNN 8.3.1.22，TensoFlow 1.15.5，TensorRT 8.2.1.8）
+ 运行方法
```shell
cd ./TFTRT
python main.py
```
+ 参考输出结果，见 ./TFTRT/result.txt

### Torch-TensorRT
+ 使用 Torch-TensorRT 来运行一个训练好的 pyTorch 模型
+ 环境：nvcr.io/nvidia/pytorch:21.12-py3（包含 python 3.8.10，CUDA 11.5.0，cuBLAS 11.7.3.1，cuDNN 8.3.1.22，pyTorch 1.11.0a0+b6df043，TensorRT 8.2.1.8）
+ 运行方法
```shell
cd ./Torch-TensorRT
python main.py
```
+ 参考输出结果，见 ./Torch-TensorRT/result.txt
