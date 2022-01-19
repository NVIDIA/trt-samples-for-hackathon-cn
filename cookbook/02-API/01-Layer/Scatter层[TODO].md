# Scatter 层[TODO]
+ 初始示例代码
+ axis
+ mode

---
### 初始示例代码
```python
import numpy as np
from cuda import cuda
import tensorrt as trt

np.random.seed(97)
nIn,cIn,hIn,wIn = 1,3,4,5                                                                           # 输入张量 NCHW
data0   = np.arange(nIn*cIn*hIn*wIn,dtype=np.float32).reshape(nIn,cIn,hIn,wIn)                      # 输入数据
#data1   = np.random.randint(0,4,[nIn,cIn,hIn,wIn])
data1   = np.tile(np.arange(wIn),[nIn,cIn,hIn,1]).astype(np.float32).reshape(nIn,cIn,hIn,wIn)
data2   = - data0

np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
cuda.cuInit(0)
cuda.cuDeviceGet(0)

logger  = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config  = builder.create_builder_config()
config.max_workspace_size = 1 << 30
inputT0 = network.add_input('inputT0', trt.DataType.FLOAT, (nIn,cIn,hIn,wIn))
inputT1 = network.add_input('inputT1', trt.DataType.INT32, (nIn,cIn,hIn,wIn))
inputT2 = network.add_input('inputT2', trt.DataType.FLOAT, (nIn,cIn,hIn,wIn))
#---------------------------------------------------------------------------------------------------# 替换部分
scatterLayer = network.add_scatter(inputT0,inputT1,inputT2,trt.ScatterMode.ELEMENT)
scatterLayer.axis = 3
#---------------------------------------------------------------------------------------------------# 替换部分
network.mark_output(scatterLayer.get_output(0))
engineString    = builder.build_serialized_network(network,config)
engine          = trt.Runtime(logger).deserialize_cuda_engine(engineString)
context         = engine.create_execution_context()
_, stream       = cuda.cuStreamCreate(0)

inputH0     = np.ascontiguousarray(data0.reshape(-1))
inputH1     = np.ascontiguousarray(data1.reshape(-1))
inputH2     = np.ascontiguousarray(data2.reshape(-1))
outputH0    = np.empty(context.get_binding_shape(3),dtype = trt.nptype(engine.get_binding_dtype(3)))
_,inputD0   = cuda.cuMemAllocAsync(inputH0.nbytes,stream)
_,inputD1   = cuda.cuMemAllocAsync(inputH1.nbytes,stream)
_,inputD2   = cuda.cuMemAllocAsync(inputH2.nbytes,stream)
_,outputD0  = cuda.cuMemAllocAsync(outputH0.nbytes,stream)

cuda.cuMemcpyHtoDAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, stream)
cuda.cuMemcpyHtoDAsync(inputD1, inputH1.ctypes.data, inputH1.nbytes, stream)
cuda.cuMemcpyHtoDAsync(inputD2, inputH2.ctypes.data, inputH2.nbytes, stream)
context.execute_async_v2([int(inputD0), int(inputD1), int(inputD2), int(outputD0)], stream)
cuda.cuMemcpyDtoHAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, stream)
cuda.cuStreamSynchronize(stream)

print("inputH0 :", data0.shape)
print(data0)
print("inputH1 :", data1.shape)
print(data1)
print("inputH2 :", data2.shape)
print(data2)
print("outputH0:", outputH0.shape)
print(outputH0)

cuda.cuStreamDestroy(stream)
cuda.cuMemFree(inputD0)
cuda.cuMemFree(outputD0)
```

+ 
