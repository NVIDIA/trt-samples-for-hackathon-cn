# TensorRT Cookbook in Chinese

## 08-Tool
+ 一些开发辅助工具的使用示例，包括 Netron，nsight system，onnx-graphsurgeon，Polygraphy，trtexec

### Netron
+ 模型可视化工具
+ 下载：[这里](https://github.com/lutzroeder/Netron)
+ 目录中有一个 model.onnx-save 可供参考，打开前把后缀名的 -save 去掉

### Nsight systems
+ 程序系统优化工具
+ 注意将 ns 更新到最新版本，较老的 ns 可能打不开较新 ns 生成的 .qdrep 或 .nsys-rep
+ 运行方法
```shell
cd ./NsightSystems
./command.sh
```
+ 参考输出结果，见 ./NsightSystems/MNISTModel.qdrep 或 ./NsightSystems/MNISTModel.nsys-rep
+ 使用 ```/usr/local/cuda/Nsight*/bin/nsys-ui``` 打开上述文件查看

### OnnxGraphsurgeon
+ .onnx 模型计算图编辑库
+ 每个样例代码的说明参见 ```./OnnxGraphSurgeon/command.sh``` 中的注释
+ 这部分代码参考了 NVIDIA 官方仓库关于 onnx-graphsurgeon 的样例代码 [link](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon/examples)
+ 运行方法
```shell
cd ./OnnxGraphSurgeon
./command.sh
```
+ 参考输出结果，见 ./OnnxGraphSurgeon/*.txt

### Polygraphy
+ DL 模型调试器
+ 每个样例代码的说明参见 ```./Polygraphy/*Example/command.sh``` 中的注释
+ 运行方法
```shell
cd ./Polygraphy
cd ./*Example
./command.sh
```
+ 参考输出结果，见 ./Polygraphy/*Example/*.txt

### trex
+ TensorRT Engine 可视化和分析工具
+ 原仓库：https://github.com/NVIDIA/TensorRT/tree/main/tools/experimental/trt-engine-explorer
+ 样例代码的说明参见 ```./trex/README.md```

### trtexec
+ TensorRT 命令行工具
+ 每个样例代码的说明参见 ```./trtexec/command.sh``` 中的注释
+ 运行方法
```shell
cd ./trtexec
./command.sh
```
+ 参考输出结果，见 ./trtexec/*.txt

