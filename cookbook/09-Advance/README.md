# TensorRT Cookbook in Chinese

## 09-Advance
+ 一些 TensorRT 高级用法
+ TimingCache，保存和复用 auto tuning 缓存，用于多次构建相同 tactic 的引擎

### TimingCache
+ 环境：
    - nvcr.io/nvidia/tensorrt:21.12-py3（包含 python 3.8.10，CUDA 11.5.50，cuDNN 8.3.1，**TensoRT 8.2.3**）
+ 运行方法
```shell
cd ./TimingCache
python TimingCache.py 0 10 30   # 构建引擎，不使用 timing cache，热身 10 次测试 30 次，计时
python TimingCache.py 1 10 30   # 构建引擎，并且保存 timing cache，热身 10 次测试 30 次，计时
python TimingCache.py 1 10 30   # 构建引擎，使用上一步得到的 timing cache，热身 10 次测试 30 次，计时

# 修改 TimingCache.py 的 logger = trt.Logger(trt.Logger.VERBOSE)
rm model.cache
python TimingCache.py 0 0 1 > output0.log   # 构建引擎，不使用 timing cache，保存日志
python TimingCache.py 1 0 1 > output1.log   # 构建引擎，并且保存 timing cache，保存日志
python TimingCache.py 1 0 1 > output2.log   # 构建引擎，使用上一步得到的 timing cache，保存日志
```
+ 参考输出结果，见 ./TimingCache/result-*.txt，注意读取 TimingCache 的日志中所有 Tactic 过程都没了

