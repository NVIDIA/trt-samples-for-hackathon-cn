# TensorRT Cookbook in Chinese

## 01-SimpleDemo —— TensorRT API 的简单示例

### TensorRT6
+ 采用 TensorRT6 + Implicit batch 模式 + Static Shape 模式 + Builder API + 较旧的 pycuda 库
+ 环境：nvcr.io/nvidia/tensorrt:19.12-py3（包含 python 3.6，CUDA 10.2.89，cuDNN 7.6.5，TensoRT 6.0.1）
+ 需要 pycuda 库
+ 运行方法
```shell
cd ./TensorRT6
make test
```
+ 参考输出结果，见 ./TensorRT6/result.txt

### TensorRT7
+ 采用 TensorRT7 + Implicit batch 模式 + Static Shape 模式 + Builder API + 较新的 cuda-python 库的 Runtime API
+ 环境：nvcr.io/nvidia/tensorrt:21.06-py3（包含 python 3.8.5，CUDA 11.3.1，cuDNN 8.2.1，TensoRT 7.2.3.4）
+ 运行方法
```shell
cd ./TensorRT7
make test
```
+ 参考输出结果，见 ./TensorRT7/result.txt

### TensorRT8
+ 采用 TensorRT8 + Explicit batch 模式 + Dynamic Shape 模式 + BuilderConfig API + cuda-python 库的 Driver API / Runtime API 双版本
+ 环境：nvcr.io/nvidia/tensorrt:21.12-py3（包含 python 3.8.10，CUDA 11.5.50，cuDNN 8.3.1，**TensoRT 8.2.3**）
+ 运行方法
```shell
cd ./TensorRT8
make test
```
+ 参考输出结果，见 ./TensorRT8/result.tx
