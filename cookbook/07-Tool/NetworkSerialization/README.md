# Network Serialization and Deserialization

+ Serialize a network into a json file, and deserialize it back into a INetwork.

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
- [ ] callback_object_dict

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

- **优化数据结构**：
  - 当前代码中使用了大量的 `OrderedDict`，但 `OrderedDict` 的性能不如普通字典。如果顺序不是必须的，可以考虑使用普通字典。
  - 对于频繁访问的属性，可以使用 `@property` 装饰器来缓存计算结果。

### 4. **错误处理和日志**
- **错误处理**：
  - 当前代码中对错误的处理较为简单，建议增加更详细的错误信息和异常处理逻辑。例如，在 `dump_member` 和 `build_member` 中，当遇到未知属性时，可以记录详细的错误信息。
  - 对于可能引发错误的操作（如文件读写、网络构建等），建议使用 `try-except` 块来捕获异常，并提供友好的错误提示。

- **日志优化**：
  - 当前的日志记录功能较为简单，建议增加日志级别（如 INFO、WARNING、ERROR）的控制，允许用户根据需要调整日志的详细程度。
  - 使用 Python 的 `logging` 模块来替代当前的日志实现，这样可以更灵活地配置日志输出。

### 5. **扩展性和兼容性**
- **支持更多 TensorRT 版本**：
  - 当前代码中有一些硬编码的逻辑（如 `self.use_patch_80`），这些逻辑可能依赖于特定版本的 TensorRT。建议将这些逻辑封装为可配置的选项，或者通过版本检测来动态调整行为。

- **支持插件层**：
  - 当前代码中对插件层的支持是缺失的（`self.log("ERROR", "Plugin Layer not supported")`）。建议增加对插件层的序列化和反序列化支持，或者提供一个扩展机制，允许用户自定义插件层的处理逻辑。

- **支持更多层类型**：
  - TensorRT 不断更新，可能会引入新的层类型。建议在代码中预留扩展接口，方便未来支持更多层类型。

### 6. **测试和验证**
- **单元测试**：
  - 当前代码缺乏单元测试，建议为每个主要功能（如序列化、反序列化、层处理等）编写单元测试，确保代码的正确性和稳定性。
  - 使用 `unittest` 或 `pytest` 等测试框架来组织测试用例。

- **验证序列化和反序列化结果**：
  - 在反序列化后，建议增加验证逻辑，确保反序列化后的网络与原始网络一致。可以通过比较网络的结构、权重和属性来实现。
