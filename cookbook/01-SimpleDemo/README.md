# TensorRT Cookbook in Chinese

## 01-SimpleDemo —— TensorRT API 的简单示例

### TensorRT6
+ 采用 TensorRT6 + Implicit batch 模式 + Static Shape 模式 + Builder API + 较旧的 pycuda 库
+ 需要 pycuda 库
+ 运行方法
```shell
cd ./TensorRT6
make test
```
+ 参考输出结果，见 ./TensorRT6/result.log

### TensorRT7
+ 采用 TensorRT7 + Implicit batch 模式 + Static Shape 模式 + Builder API + 较新的 cuda-python 库的 Runtime API
+ 运行方法
```shell
cd ./TensorRT7
make test
```
+ 参考输出结果，见 ./TensorRT7/result.log

### TensorRT8
+ 采用 TensorRT8 + Explicit batch 模式 + Dynamic Shape 模式 + BuilderConfig API + cuda-python 库的 Driver API / Runtime API 双版本
+ 运行方法
```shell
cd ./TensorRT8
make test
```
+ 参考输出结果，见 ./TensorRT8/result.log

### TensorRT8.4
+ 基本同 TensorRT8，使用了 TensorRT8.4 中新引入的 API，python 代码仅保存了 cuda-python 库的 Runtime API 版本
+ 运行方法
```shell
cd ./TensorRT8.4
make test
```
+ 参考输出结果，见 ./TensorRT8.4/result.log
