# TensorRT Cookbook in Chinese

## 06-PluginAndParser —— 结合使用 Parser 与 Plugin 的样例
+ 需要用到 onnx-graphsurgeon，可参考 08-Tool/onnxGraphSurgeon

### pyTorch-FailConvertNonZero
+ 在 .pt 转 .onnx 转 .plan 的过程中，转换 NonZero 节点失败的例子（TensorRT 不原生支持该算子）
+ 运行方法
```shell
cd ./pyTorch-FailConvertNonZero
python3 main.py
```
+ 参考输出结果，见 ./pyTorch-FailConvertNonZero/result.log

### pyTorch-LayerNorm
+ 在 pyTorch 转 onnx 转 TensorRT 的过程中，替换一个 LayerNorm 以提高效率
+ 运行方法
```shell
cd ./pyTorch-LayerNorm
python3 main.py
```
+ 参考输出结果，见 ./pyTorch-LayerNorm/result.log

### TensorFlow-AddScalar
+ 在 .pb 转 .onnx 转 .plan 的过程中，将 Add 算子替换为 Plugin 的例子（05-Plugin 中的 AddScalarPlugin）
+ 运行方法
```shell
cd ./TensorFlow-addScalar
python3 TensorFlowToTensorRT.py
```
+ 参考输出结果，见 ./TensorFlow-addScalar/result.log

