# TensorRT Cookbook in Chinese

## 01-SimpleDemo —— TensorRT API 的简单示例
+ TensorRT6，采用 TensorRT6 + implicit batch 模式 + static shape 模式 + builder API + 较旧的 pycuda 库
+ TensorRT7，采用 TensorRT7 + implicit batch 模式 + static shape 模式 + builder API + 较新的 cuda-python 库的 driver API
+ TensorRT8，采用 TensorRT8 + explicit batch 模式 + dynamic shape 模式 + builderConfig API（推荐使用，灵活性和应用场景更广）+ cuda-python 库的 diver API /runtime API 双版本

### TensorRT6
+ 环境：nvcr.io/nvidia/tensorrt:19.12-py3（包含 python 3.6，CUDA 10.2.89，cuDNN 7.6.5，TensoRT 6.0.1）
+ 运行方法
```shell
cd ./01-TensorRT6
pip install -r ./requirements.txt
# 修改 Makefile 中的 SM 等参数
make
make test
```
+ 参考输出结果，见 ./01-TensorRT6/result.txt

### TensorRT7
+ 环境：nvcr.io/nvidia/tensorrt:21.06-py3（包含 python 3.8.5，CUDA 11.3.1，cuDNN 8.2.1，TensoRT 7.2.3.4）
+ 运行方法
```shell
cd ./02-TensorRT7
pip install -r ./requirements.txt
# 修改 Makefile 中的 SM 等参数
make
make test
```
+ 参考输出结果，见 ./02-TensorRT7/result.txt

### TensorRT8
+ 环境：
    - nvcr.io/nvidia/tensorrt:21.09-py3（包含 python 3.8.5，CUDA 11.3.1，cuDNN 8.2.1，**TensoRT 8.0.3**）
    - nvcr.io/nvidia/tensorrt:21.12-py3（包含 python 3.8.10，CUDA 11.5.50，cuDNN 8.3.1，**TensoRT 8.2.3**）
+ 运行方法
```shell
cd ./03-TensorRT8
pip install -r ./requirements.txt
# 修改 Makefile 中的 SM 等参数
make
make test
```
+ 参考输出结果，见 ./03-TensorRT8/result.txt

