# Network Serialization and Deserialization

+ Serialize a network into a json file, and deserialize it back into a INetwork.

+ See more examples in tests/test*.py

+ Steps to run.

```bash
python3 main.py
```

## TODO

- [ ] INT8-PTQ
- [ ] Plugin Layer
- [ ] Advanced feature
- [ ] Timing cache
- [ ] Calibration cache
- [ ] Refit
- [ ] callback object dict
- [ ] Skipped cases

## Issues and suggestions

1. We have no API to get Optimization Profile from a BuidlerConfig (such as `get_optimization_profile`).

2. （已解决，报 bug） 默认状态下所有新层的输出张量的 allowed_formats 都是 4095（支持所有格式），但是直接把 4095 赋值给重建张量的 allowed_formats，会因为 TensorRT 乱选 format 而报错
    比如 TensorRT 会选 HWC4 形 format 然后才发现该张量维度不足 3，报 [TRT] [E] 2: [optimizer.cpp::symbolicFormat::4473] Error Code 2: Internal Error (Assertion dims.nbDims >= 3 failed. )

3. （已解决，报 bug）INT8-QDQ，默认状态下 QuantizeLayer/ DequantizeLayer 的 axis 值为 -1（表示 per-tensor quantize），但在重建该层的时候 axis 参数不接受负值（文档中负值表示从末维开始算，但是代码不支持），想要 per-TensorRTquantize 要改成 0

4. （已解决，报 bug）Resize 层的 ResizeCoordinateTransformation 参数，在线性插值模式下，默认状态该参数值是 ASYMMETRIC 但是 TensorRT 的行为是 HALF_PIXIEL，想要 ASYMMETRIC 模式需要再手动设置一次该值为 ASYMMETRIC

5. We can not get value from a `trt.Weights`, since type of `trt.Weights(aa)` and `trt.Weights(aa).numpy()` are both `trt.Weights`.

6. For shape layer, `layer.shape = [3,4,5]` is useless after `add_fill()`.

7. Fill layer 随机模式，每次 build 得到的随机数都一样

8. Non-Zeros Layer + Shuffle Layer，刚加上 shuffle layer 还没设置 reshape_dims 的时候，其中就会有随机值，影响后续使用

9.  Resize Layer，只设置 scale_factor 没有设置 shape 时 shape 上就偶遇随机值，影响后续使用

10. Slice Layer，axes 参数随机值

11. 使用 logging 模块来打印日志

```Python
# Special cases for some layers
# TODO: test cases in unit test:
# 1. Plugin_V2
# 2. static Shuffle without setting reshape_dims
# 3. Slice with different count of input tensors, especially Fill mode with 5 input tensors
# 4. Whether "shape" or "scales" is parsed correctly in Resize layer
# 5. How to check layer.shape in static resize mode + use scale mode in Resize layer
# 6. Resize layer:
#if layer.resize_mode == trt.ResizeMode.LINEAR and layer.coordinate_transformation == trt.ResizeCoordinateTransformation.ASYMMETRIC:
#        print("[Warning from NetworkInspector]: ResizeCoordinateTransformation of Resize Layer %s is set as HALF_PIXEL though default behaviour or your explicit set is ASYMMETRIC mode, please refer to the source code of NetworkInspector if you insist to use ASYMMETRIC mode!" % layer.name)
#        layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
#        #layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC # uncomment this line if you want to use ASYMMETRIC mode
# 7. Fill layer, "shape" member in dynamic / static fill mode
# 8. Convolution layer, no bias
# 9. Constant layer with 0 / 1 dimension
```

改进建议：
### 1. **代码结构优化**

### 2. **代码可读性提升**
- **注释和文档**：
  - 虽然代码中有一些注释，但整体注释密度仍然较低，尤其是对于复杂的逻辑和特殊处理。建议增加更多详细的注释，解释每一部分代码的作用和实现细节。
  - 为每个函数和类添加文档字符串，说明其功能、输入参数、返回值和可能的异常。

- **变量命名**：
  - 一些变量命名较为模糊，例如 `c`、`op`、`dump` 等。建议使用更具描述性的命名，例如 `constants`、`optimization_profile`、`layer_dump` 等。
  - 避免使用缩写，除非这些缩写在团队中被广泛接受。


### 3. **性能优化**
- **减少重复计算**：
  - 在 `dump_layers` 和 `build_layers` 中，多次调用了 `layer.type` 和其他重复的属性访问。建议将这些值缓存起来，避免重复计算。
  - 例如，可以将 `layer.type` 的值存储在一个变量中，避免多次调用。


### 4. **错误处理和日志**
- **错误处理**：
  - 当前代码中对错误的处理较为简单，建议增加更详细的错误信息和异常处理逻辑。例如，在 `dump_member` 和 `build_member` 中，当遇到未知属性时，可以记录详细的错误信息。
  - 对于可能引发错误的操作（如文件读写、网络构建等），建议使用 `try-except` 块来捕获异常，并提供友好的错误提示。
