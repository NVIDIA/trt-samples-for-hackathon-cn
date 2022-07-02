# TensorRT Cookbook in Chinese

## 09-Advance —— TensorRT 高级用法

### AlgorithmSelector
### CudaGraph
### EmptyTensor[TODO]
### EngineInspector
### ErrorRecoder[TODO]
### GPUAllocator[TODO]
### Logger
### MultiContext
### MultiOptimizationProfile
### MultiStream
### Profiling
### ProfilingVerbosity
### Refit
### StreamAndAsync
### StrictType
### TacticSource
### TimingCache




### AlgorithmSelector
+ 手工筛选 TensorRT 自动优化阶段网络每一层可用的 tactic
+ 环境：nvcr.io/nvidia/tensorrt:21.12-py3（包含 python 3.8.10，CUDA 11.5.50，cuDNN 8.3.1，TensoRT 8.2.3）
+ 运行方法
```shell
cd ./AlgorithmSelector
python AlgorithmSelector.py
```
+ 参考输出结果，见 ./AlgorithmSelector/result.txt














+ 保存和复用 auto tuning 缓存，用于多次构建相同 tactic 的引擎
+ 环境：nvcr.io/nvidia/tensorrt:21.12-py3（包含 python 3.8.10，CUDA 11.5.50，cuDNN 8.3.1，TensoRT 8.2.3）
+ 运行方法
```shell
cd ./TimingCache
python main.py
```
+ 参考输出结果，见 ./TimingCache/result-*.txt，注意读取 TimingCache 并再次构建时的日志中所有 Tactic 过程都没了

