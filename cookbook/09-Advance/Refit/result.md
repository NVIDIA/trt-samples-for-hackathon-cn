### 正常运行
```shell
Succeeded building engine!
Do not refit!
data: (1, 6, 9)
[[[1. 2. 3. 1. 2. 3. 1. 2. 3.]
  [4. 5. 6. 4. 5. 6. 4. 5. 6.]
  [7. 8. 9. 7. 8. 9. 7. 8. 9.]
  [1. 2. 3. 1. 2. 3. 1. 2. 3.]
  [4. 5. 6. 4. 5. 6. 4. 5. 6.]
  [7. 8. 9. 7. 8. 9. 7. 8. 9.]]]
outputH0: (1, 1, 4, 7)
[[[[0. 0. 0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0. 0. 0.]]]]
Succeeded loading engine!
Refit!
data: (1, 6, 9)
[[[1. 2. 3. 1. 2. 3. 1. 2. 3.]
  [4. 5. 6. 4. 5. 6. 4. 5. 6.]
  [7. 8. 9. 7. 8. 9. 7. 8. 9.]
  [1. 2. 3. 1. 2. 3. 1. 2. 3.]
  [4. 5. 6. 4. 5. 6. 4. 5. 6.]
  [7. 8. 9. 7. 8. 9. 7. 8. 9.]]]
outputH0: (1, 1, 4, 7)
[[[[12345.679 23156.49  31264.598 12345.679 23156.49  31264.598 12345.679]
   [45678.914 56489.723 64597.832 45678.914 56489.723 64597.832 45678.914]
   [78912.34  89723.16  97831.27  78912.34  89723.16  97831.27  78912.34 ]
   [12345.679 23156.49  31264.598 12345.679 23156.49  31264.598 12345.679]]]]
```

### 注释掉两个 refitter.set_weights 之一
```shell
Succeeded building engine!
Do not refit!
data: (1, 6, 9)
[[[1. 2. 3. 1. 2. 3. 1. 2. 3.]
  [4. 5. 6. 4. 5. 6. 4. 5. 6.]
  [7. 8. 9. 7. 8. 9. 7. 8. 9.]
  [1. 2. 3. 1. 2. 3. 1. 2. 3.]
  [4. 5. 6. 4. 5. 6. 4. 5. 6.]
  [7. 8. 9. 7. 8. 9. 7. 8. 9.]]]
outputH0: (1, 1, 4, 7)
[[[[0. 0. 0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0. 0. 0.]
   [0. 0. 0. 0. 0. 0. 0.]]]]
Succeeded loading engine!
Refit!
[ conv - WeightsRole.BIAS ]
[TensorRT] ERROR: 4: [refit.cpp::refitCudaEngine::1769] Error Code 4: Internal Error (missing 1 needed Weights. Call IRefitter::getMissing to get their layer names and roles or IRefitter::getMissingWeights to get their weights names.)
Failed Refitting engine!
```

### TensoRT8.4.0.6 还不支持 Dynamic Shape 模式下进行 Refit，以后版本可能开始支持
```
[TRT] [E] 4: [network.cpp::validate::2924] Error Code 4: Internal Error (Refittable networks with dynamic shapes is not supported.)
[TRT] [E] 2: [builder.cpp::buildSerializedNetwork::609] Error Code 2: Internal Error (Assertion enginePtr != nullptr failed. )
Failed building engine!

```
