# NetworkInspector

## Features

- [x] Simple layers

- [x] Simple layers unit test

- [x] Dynamic Shape mode

- [x] ShapeTensor Input

- [x] IfCondition structure

- [x] Loop structure

- [ ] INT8-PTQ

- [x] INT8-QDQ

- [ ] Plugin Layer

- [x] RNNV2 Layer

- [ ] Advanced feature

- [x] Test with parser

- [x] Test model based on MNIST of cookbook

- [x] Test wenet model of Hackathon 2022

## Problems

1. （已解决，WAR）不能从 BuilderConfig 中获取已经加入其中的 Optimization Profile，导致序列化 Network 的时候要带着 OP 一起传入 extractModel，不够简洁

2. （已解决，报 bug） 默认状态下所有新层的输出张量的 allowed_formats 都是 4095（支持所有格式），但是直接把 4095 赋值给重建张量的 allowed_formats，会因为 TensorRT 乱选 format 而报错
    比如 TensorRT 会选 HWC4 形 format 然后才发现该张量维度不足 3，报 [TRT] [E] 2: [optimizer.cpp::symbolicFormat::4473] Error Code 2: Internal Error (Assertion dims.nbDims >= 3 failed. )

3. （已解决，报 bug）INT8-QDQ，默认状态下 QuantizeLayer/ DequantizeLayer 的 axis 值为 -1（表示 per-tensor quantize），但在重建该层的时候 axis 参数不接受负值（文档中负值表示从末维开始算，但是代码不支持），想要 per-TensorRTquantize 要改成 0

4. （已解决，报 bug）Resize 层的 ResizeCoordinateTransformation 参数，在线性插值模式下，默认状态该参数值是 ASYMMETRIC 但是 TensorRT 的行为是 HALF_PIXIEL，想要 ASYMMETRIC 模式需要再手动设置一次该值为 ASYMMETRIC

5. （未解决）含构建期参数的 Plugin Layer 无法解析，因为没有 API 可以通过 Plugin Layer 获取其 PluginV2 里的参数

6. （已解决）从 ONNX parse 进来的模型没有完全拓扑排序，出现了靠前 Layer 的Input tensor来自靠后 Layer 的输出张量的现象（用 04-Parser/pyTorch-ONNX-TensorRT 的例子或者 Wenet 的例子）
