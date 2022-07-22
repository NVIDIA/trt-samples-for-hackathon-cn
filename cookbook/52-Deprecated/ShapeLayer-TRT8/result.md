### TensorRT 7
```
nInput = 0, nOutput = 1
Output 0: (4,) 
 [1 3 4 5]
```

### TensorRT 8
```
[TRT] [E] 1: [executionContext.cpp::executeInternal::667] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
nInput = 1, nOutput = 1
main.py:37: DeprecationWarning: Use build_serialized_network instead.
  engine = builder.build_engine(network, config)  # 使用旧版的 engine 生成 API
main.py:51: DeprecationWarning: Use execute_v2 instead.
  context.execute(nB, bufferD)
Traceback (most recent call last):
  File "main.py", line 53, in <module>
    cudart.cudaMemcpy(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
IndexError: list index out of range
[TRT] [E] 1: [cudaResources.cpp::~ScopedCudaStream::47] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
[TRT] [E] 1: [cudaResources.cpp::~ScopedCudaEvent::24] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
[TRT] [E] 1: [cudaResources.cpp::~ScopedCudaEvent::24] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
```

### 说明
+ TensorRT 7 中没有用到的输入张量会被网络优化掉，（打印 engine 信息显示输入张量数为 0），在运行期不需要绑定输入张量
+ TensorRT 8 中没有用到的输入张量仍然会保留，（打印 engine 信息显示输入张量数为 1），在运行期仍需要绑定输入张量
