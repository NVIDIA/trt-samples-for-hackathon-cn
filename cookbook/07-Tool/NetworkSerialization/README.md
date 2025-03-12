# Network Serialization and Deserialization

+ Serialize a network into a json file, and deserialize it back into a INetwork.

+ NOT finish yet!!

## Features

- [x] Simple layers
- [x] Simple layers unit test

- [ ] INT8-PTQ
- [x] INT8-QDQ

- [x] Dynamic Shape mode
- [x] ShapeTensor Input

- [x] IfCondition structure
- [x] Loop structure
- [ ] Plugin Layer
- [ ] Advanced feature

- [x] Test with parser
- [x] Test model based on MNIST of cookbook
- [x] Test wenet model from Hackathon 2022
- [ ] Timing cache
- [ ] Calibration cache
- [ ] Refit
- [ ] Constant Layer + Int4 weight (can not get value from `trt.Weights`)

## Issues and suggestions

1. We have no API to get Optimization Profile from a BuidlerConfig (such as `get_optimization_profile`).

2. （已解决，报 bug） 默认状态下所有新层的输出张量的 allowed_formats 都是 4095（支持所有格式），但是直接把 4095 赋值给重建张量的 allowed_formats，会因为 TensorRT 乱选 format 而报错
    比如 TensorRT 会选 HWC4 形 format 然后才发现该张量维度不足 3，报 [TRT] [E] 2: [optimizer.cpp::symbolicFormat::4473] Error Code 2: Internal Error (Assertion dims.nbDims >= 3 failed. )

3. （已解决，报 bug）INT8-QDQ，默认状态下 QuantizeLayer/ DequantizeLayer 的 axis 值为 -1（表示 per-tensor quantize），但在重建该层的时候 axis 参数不接受负值（文档中负值表示从末维开始算，但是代码不支持），想要 per-TensorRTquantize 要改成 0

4. （已解决，报 bug）Resize 层的 ResizeCoordinateTransformation 参数，在线性插值模式下，默认状态该参数值是 ASYMMETRIC 但是 TensorRT 的行为是 HALF_PIXIEL，想要 ASYMMETRIC 模式需要再手动设置一次该值为 ASYMMETRIC

5. （未解决）含构建期参数的 Plugin Layer 无法解析，因为没有 API 可以通过 Plugin Layer 获取其 PluginV2 里的参数

6. We can not get value from a `trt.Weights`, since type of `trt.Weights(aa)` and `trt.Weights(aa).numpy()` are both `trt.Weights`.

7. For shape layer, `layer.shape = [3,4,5]` is useless after `add_fill()`.

8. Fill layer 随机模式，每次 build 得到的随机数都一样

9. Non-Zeros Layer + Shuffle Layer，刚加上 shuffle layer 还没设置 reshape_dims 的时候，其中就会有随机值，影响后续使用

10. Resize Layer，只设置 scale_factor 没有设置 shape 时 shape 上就偶遇随机值，影响后续使用

11. Slice Layer，axes 参数随机值
