# TensorRT Cookbook in Chinese

## 05-Plugin —— 自定义插件书写

### loadNpz
+ 从 .npz 中读取数据并被 Plugin 使用，展示了在 Plugin 中读取权重数据的方法
+ 运行方法
```shell
cd ./loadNpz
make
make test
```
+ 参考输出结果，见 ./loadNpz/result.txt

### multipleVersion
+ 书写和使用同一个 Plugin 的不同版本（使用 TensorRT 内建的 Plugin 时也需要如此确认 Plugin 的版本号）
+ 运行方法
```shell
cd ./multipleVersion
make
make test
```
+ 参考输出结果，见 ./multipleVersion/result.txt

### PluginPrecess
+ 使用多 OptimizationProfile 的情境下一个含有 Plugin 的网络中，Plugin 的各成员函数调用顺序
+ 运行方法
```shell
cd ./PluginPrecess
make
make test
```
+ 参考输出结果，见 ./PluginPrecess/result.txt，注意，如果直接使用 make test > XXX.txt 来导出标准输出，则主程序和 Plugin 的结果会相互抢占

### PluginReposity
+ 常见 Plugin 小仓库，收集各类常见 Plugin，仅保证计算结果正确，不保证性能最优化
+ 环境和运行方法同 usePluginV2DynamicExt
+ 很多 Plugin 基于 TensorRT6 或 TensorRT7 书写，在更新版本的 TensorRT 上编译时需要修改部分成员函数

### useCuBLASPlugin
+ 在 Plugin 中使用 cuBLAS 计算矩阵乘法
+ 运行方法
```shell
cd ./useCuBLAS
make
make test
```
+ 参考输出结果，见 ./useCuBLAS/result.txt
+ 内含一个 useCuBLASAlone.cu 生成的 useCuBLASAlone.exe，为单独使用 cuBLAS 计算 GEMM 的例子，可以通过 ```./useCuBLASAlone.exe``` 运行

### useFP16
+ 在 Plugin 中使用 float16 数据类型，功能同 usePluginV2Ext
+ 运行方法
```shell
cd ./useFP16
make
make test
```
+ 参考输出结果，见 ./useFP16/result.txt

### useInt8
+ 在 Plugin 中使用 int8 数据类型，功能同 usePluginV2Ext
+ 在使用 Plugin 中使用 int8 时要注意输入输出的数据排布可能不是 Linear 型的
+ 运行方法
```shell
cd ./useINT8
make
make test
```
+ 参考输出结果，见 ./useINT8/result.txt

### usePluginV2DynamicExt
+ 特性
    - 使用 IPluginV2DynamicExt 类实现一个 Plugin，功能同 usePluginV2Ext
    - 不同点：输入张量形状可变（相同维度不同形状的输入共享同一个 TensorRT engine）
+ 运行方法
```shell
cd ./usePluginV2DynamicExt
make
make test
```
+ 参考输出结果，见 ./usePluginV2DynamicExt/result.txt

### usePluginV2Ext
+ 特性
    - 使用 IPluginV2Ext 类实现一个 Plugin，功能是给输入张量所有元素加上同一个标量值，然后输出
    - 使用 Explicit Batch 模式
    - 输入张量形状不可变（不同形状的输入要建立不同的 TensorRT engine）
    - 标量加数在构建期确定（多次 inference 之间不能更改）
    - 支持序列化和反序列化
    - 支持 TensorRT7 和 TensorRT8 版本（但是需要修改 AddScalarPlugin.h 和 AddScalarPlugin.cu 中 enqueue 函数的声明和定义）
+ 运行方法
```shell
cd ./usePluginV2Ext
make
make test
```
+ 参考输出结果，见 ./usePluginV2Ext/result.txt

### usePluginV2IOExt
+ 特性
    - 使用 IPluginV2IOExt 类实现一个 Plugin，功能同 usePluginV2Ext
    - 不同点：使用 Implicit Batch 模式
+ 运行方法
```shell
cd ./usePluginV2IOExt
make
make test
```
+ 参考输出结果，见 ./usePluginV2IO/result.txt

