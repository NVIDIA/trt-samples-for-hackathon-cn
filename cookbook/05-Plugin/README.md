# TensorRT Cookbook in Chinese

## 05-Plugin
+ usePluginV2Ext
    - 使用 IPluginV2Ext 类实现一个 Plugin，功能是给输入张量所有元素加上同一个标量值，然后输出
    - 使用 Explicit Batch 模式
    - 输入张量形状不可变（不同形状的输入要建立不同的 TensorRT engine）
    - 标量加数在 build 期确定（多次 inference 之间不能更改）
    - 支持序列化和反序列化
    - 支持 TensorRT7 和 TensorRT8 版本（但是需要修改 AddScalarPlugin.h 和 AddScalarPlugin.cu 中 enqueue 函数的声明和定义）
+ usePluginV2DynamicExt
    - 使用 IPluginV2DynamicExt 类实现一个 Plugin，功能同 IPluginV2Ext
    - 不同点：输入张量形状可变（相同维度不同形状的输入共享同一个 TensorRT engine）
+ usePluginV2IOExt [TODO]
    - 使用 IPluginV2IOExt 类实现一个 Plugin，功能同 IPluginV2Ext
    - 不同点：使用 Implicit Batch 模式
+ cuBLASPlugin
    - 在 Plugin 中使用 cuBLAS 计算矩阵乘法，并在 plguin 之间利用指针共享乘数权重
+ fp16
    - 在 Plugin 中使用 float16 数据类型，功能同 IPluginV2Ext
+ pluginReposity
    - 常见 Plugin 小仓库，收集各类常见 Plugin
    - 仅保证计算结果正确，不保证性能最优化

### usePluginV2Ext
+ 环境：nvcr.io/nvidia/tensorrt:21.09-py3（包含 python 3.8.5，CUDA 11.3.1，cuDNN 8.2.1，TensoRT 8.0.3）
+ 运行方法
```shell
cd ./usePluginV2Ext
# 修改 Makefile 中的 SM 等参数
make
make test
```
+ 参考输出结果，见 ./usePluginV2Ext/result.txt

### usePluginV2DynamicExt
+ 环境：nvcr.io/nvidia/tensorrt:21.09-py3（包含 python 3.8.5，CUDA 11.3.1，cuDNN 8.2.1，TensoRT 8.0.3）
+ 运行方法
```shell
cd ./usePluginV2DynamicExt
# 修改 Makefile 中的 SM 等参数
make
make test
```
+ 参考输出结果，见 ./usePluginV2DynamicExt/result.txt

### usePluginV2IOExt
+ 环境：nvcr.io/nvidia/tensorrt:21.09-py3（包含 python 3.8.5，CUDA 11.3.1，cuDNN 8.2.1，TensoRT 8.0.3）
+ 运行方法
```shell
cd ./usePluginV2IOExt
# 修改 Makefile 中的 SM 等参数
make
make test
```
+ 参考输出结果，见 ./usePluginV2IO/result.txt

### cuBLASPlugin
+ 环境：nvcr.io/nvidia/tensorrt:21.09-py3（包含 python 3.8.5，CUDA 11.3.1，cuDNN 8.2.1，TensoRT 8.0.3）
+ 运行方法
```shell
cd ./cuBLASPlugin
# 修改 Makefile 中的 SM 等参数
make
make test
```
+ 参考输出结果，见 ./cuBLASPlugin/result.txt
+ 另含一个 useCuBLASAlone.cu 生成的 useCuBLASAlone.exe，为单独使用 cuBLAS 计算 GEMM 的例子可以通过 ```./useCuBLASAlone.exe``` 运行

### fp16
+ 环境：nvcr.io/nvidia/tensorrt:21.09-py3（包含 python 3.8.5，CUDA 11.3.1，cuDNN 8.2.1，TensoRT 8.0.3）
+ 运行方法
```shell
cd ./fp16
# 修改 Makefile 中的 SM 等参数
make
make test
```
+ 参考输出结果，见 ./FP16AndInt8/result.txt

### pluginReposity
+ 环境和运行方法同 usePluginV2DynamicExt

