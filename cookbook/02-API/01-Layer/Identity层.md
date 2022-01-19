# Identity 层
+ 初始示例代码
+ 用于精度转换
+ 用于 iterator 层

---
### 初始示例代码
```python
import numpy as np
from cuda import cuda
import tensorrt as trt

nIn,cIn,hIn,wIn = 1,3,4,5                                                                           # 输入张量 NCHW
data    = np.arange(nIn*cIn*hIn*wIn,dtype=np.float32).reshape(nIn,cIn,hIn,wIn)                      # 输入数据

np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
cuda.cuInit(0)
cuda.cuDeviceGet(0)

logger  = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config  = builder.create_builder_config()
inputT0 = network.add_input('inputT0', trt.DataType.FLOAT, (nIn,cIn,hIn,wIn))
#---------------------------------------------------------------------------------------------------# 替换部分
identityLayer = network.add_identity(inputT0)
#---------------------------------------------------------------------------------------------------# 替换部分
network.mark_output(identityLayer.get_output(0))
engineString    = builder.build_serialized_network(network,config)
engine          = trt.Runtime(logger).deserialize_cuda_engine(engineString)
context         = engine.create_execution_context()
_, stream       = cuda.cuStreamCreate(0)

inputH0     = np.ascontiguousarray(data.reshape(-1))
outputH0    = np.empty(context.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
_,inputD0   = cuda.cuMemAllocAsync(inputH0.nbytes,stream)
_,outputD0  = cuda.cuMemAllocAsync(outputH0.nbytes,stream)

cuda.cuMemcpyHtoDAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, stream)
context.execute_async_v2([int(inputD0), int(outputD0)], stream)
cuda.cuMemcpyDtoHAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, stream)
cuda.cuStreamSynchronize(stream)

print("inputH0 :", data.shape)
print(data)
print("outputH0:", outputH0.shape)
print(outputH0)

cuda.cuStreamDestroy(stream)
cuda.cuMemFree(inputD0)
cuda.cuMemFree(outputD0)
```

+ 输入张量形状 (1,3,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             0. &  1. &  2. &  3. &  4. \\
             5. &  6. &  7. &  8. &  9. \\
            10. & 11. & 12. & 13. & 14. \\
            15. & 16. & 17. & 18. & 19.
        \end{matrix}\right]
        \left[\begin{matrix}
            20. & 21. & 22. & 23. & 24. \\
            25. & 26. & 27. & 28. & 29. \\
            30. & 31. & 32. & 33. & 34. \\
            35. & 36. & 37. & 38. & 39.
        \end{matrix}\right]
        \left[\begin{matrix}
            40. & 41. & 42. & 43. & 44. \\
            45. & 46. & 47. & 48. & 49. \\
            50. & 51. & 52. & 53. & 54. \\
            55. & 56. & 57. & 58. & 59.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量形状 (1,3,4,5)，与输入张量一模一样
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             0. &  1. &  2. &  3. &  4. \\
             5. &  6. &  7. &  8. &  9. \\
            10. & 11. & 12. & 13. & 14. \\
            15. & 16. & 17. & 18. & 19.
        \end{matrix}\right]
        \left[\begin{matrix}
            20. & 21. & 22. & 23. & 24. \\
            25. & 26. & 27. & 28. & 29. \\
            30. & 31. & 32. & 33. & 34. \\
            35. & 36. & 37. & 38. & 39.
        \end{matrix}\right]
        \left[\begin{matrix}
            40. & 41. & 42. & 43. & 44. \\
            45. & 46. & 47. & 48. & 49. \\
            50. & 51. & 52. & 53. & 54. \\
            55. & 56. & 57. & 58. & 59.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### 用于精度转换
```python
import numpy as np
from cuda import cuda
import tensorrt as trt

nIn,cIn,hIn,wIn = 1,3,4,5
data    = np.arange(nIn*cIn*hIn*wIn,dtype=np.float32).reshape(nIn,cIn,hIn,wIn)

np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
cuda.cuInit(0)
cuda.cuDeviceGet(0)

logger  = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config  = builder.create_builder_config()
config.flags  = 1 << int(trt.BuilderFlag.FP16) | 1 << int(trt.BuilderFlag.INT8)                     # 需要打开相应的 FP16 模式或者 INT8 模式
inputT0 = network.add_input('inputT0', trt.DataType.FLOAT, (nIn,cIn,hIn,wIn))
#---------------------------------------------------------------------------------------------------# 替换部分
convertToFloat16Layer = network.add_identity(inputT0)
convertToFloat16Layer.get_output(0).dtype = trt.DataType.HALF

convertToInt32Layer = network.add_identity(inputT0)
convertToInt32Layer.get_output(0).dtype = trt.DataType.INT32

convertToInt8Layer = network.add_identity(inputT0)
convertToInt8Layer.get_output(0).dtype = trt.DataType.INT8
convertToInt8Layer.get_output(0).set_dynamic_range(0,127)                                           # 需要设置 dynamic range 或者给定 calibration
#---------------------------------------------------------------------------------------------------# 替换部分
network.mark_output(convertToFloat16Layer.get_output(0))
network.mark_output(convertToInt32Layer.get_output(0))
network.mark_output(convertToInt8Layer.get_output(0))

engineString    = builder.build_serialized_network(network,config)
engine          = trt.Runtime(logger).deserialize_cuda_engine(engineString)
context         = engine.create_execution_context()
_, stream       = cuda.cuStreamCreate(0)

inputH0     = np.ascontiguousarray(data.reshape(-1))
outputH0    = np.empty(context.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
outputH1    = np.empty(context.get_binding_shape(2),dtype = trt.nptype(engine.get_binding_dtype(2)))
outputH2    = np.empty(context.get_binding_shape(3),dtype = trt.nptype(engine.get_binding_dtype(3)))
_,inputD0   = cuda.cuMemAllocAsync(inputH0.nbytes,stream)
_,outputD0  = cuda.cuMemAllocAsync(outputH0.nbytes,stream)
_,outputD1  = cuda.cuMemAllocAsync(outputH1.nbytes,stream)
_,outputD2  = cuda.cuMemAllocAsync(outputH2.nbytes,stream)

cuda.cuMemcpyHtoDAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, stream)
context.execute_async_v2([int(inputD0), int(outputD0), int(outputD1), int(outputD2)], stream)
cuda.cuMemcpyDtoHAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, stream)
cuda.cuMemcpyDtoHAsync(outputH1.ctypes.data, outputD1, outputH1.nbytes, stream)
cuda.cuMemcpyDtoHAsync(outputH2.ctypes.data, outputD2, outputH2.nbytes, stream)
cuda.cuStreamSynchronize(stream)

print("inputH0 :", data.shape)
print(data)
print("outputH0:", outputH0.shape, outputH0.dtype)
print(outputH0)
print("outputH1:", outputH1.shape, outputH1.dtype)
print(outputH1)
print("outputH2:", outputH2.shape, outputH2.dtype)
print(outputH2)

cuda.cuStreamDestroy(stream)
cuda.cuMemFree(inputD0)
cuda.cuMemFree(outputD0)
cuda.cuMemFree(outputD1)
cuda.cuMemFree(outputD2)
```

+ 输出张形状均为 (1,3,4,5)，结果与初始示例代码相同，数据类型分别为 float16（需要开启 fp16 模式）、int32、int8（需要开启 int8 模式并设置 dynamic range）

---
### 用于 iterator 层
+ 见“Loop 结构实现 RNN.md”中的“单层单向 LSTM”部分

