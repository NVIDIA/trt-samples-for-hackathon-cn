### TensorRT 7
```
Context binding all? Yes
0 Input  (-1, 2) (1, 2)
1 Output (-1, 2) (1, 2)
Context binding all? Yes
0 Input  (-1, 2) (2, 2)
1 Output (-1, 2) (2, 2)
Context binding all? Yes
0 Input  (-1, 2) (3, 2)
1 Output (-1, 2) (3, 2)
Context binding all? Yes
0 Input  (-1, 2) (4, 2)
1 Output (-1, 2) (4, 2)
check result: True
check result: True
check result: True
check result: True
```

### TensorRT 8
```
[TRT] [E] 3: [executionContext.cpp::setOptimizationProfileInternal::794] Error Code 3: Internal Error (Profile 0 has been chosen by another IExecutionContext. Use another profileIndex or destroy the IExecutionContext that use this profile.)
main.py:33: DeprecationWarning: Use set_memory_pool_limit instead.
  config.max_workspace_size = 1 << 30
main.py:42: DeprecationWarning: Use build_serialized_network instead.
  engine = builder.build_engine(network, config)
main.py:47: DeprecationWarning: Use set_optimization_profile_async instead.
  context.active_optimization_profile = 0
Context binding all? Yes
0 Input  (-1, 2) (1, 2)
1 Output (-1, 2) (1, 2)
Traceback (most recent call last):
  File "main.py", line 47, in <module>
    context.active_optimization_profile = 0
RuntimeError: Error in set optimization profile.
```

### 说明
+ TensorRT 7 中使用 Multicontext 时，可以在构建期仅使用一个 Optimization Profile，然后在运行期创建的多个 context 之间共享该 Optimization Profile
+ TensorRT 8 开始使用 Multicontext 时，必须先在构建期使用多个 Optimization Profile，然后在运行期创建的多个 context 分别使用不同的 Optimization Profile
