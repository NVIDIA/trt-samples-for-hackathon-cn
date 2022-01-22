# CumsumPlugin
+ Use to calculate the inclusive prefix sum of input tensor by the determined axis.
+ The axis should be known in building time, not in runtime.
+ The length of the last dimension of the input tensor should not be greater than 1024
+ Till now, the sum by the last axis is calculated by CUB, but by the other axis, the sum is calculated by naive iteration.
+ Not compatible for TensorRT8, need several edition before using in ensorRT8.
+ input tensor:
    - [0]: (w)/(...,w)  float32/float16/int32,
+ input parameter:
    - [0]: axis         int32,
+ output tensor:
    - [0]: (w)/(...,w)  float32/float16/int32,

# Version log
+ V1.0: support input tensor with 1 ~ 4 dimension ((w,)/(h,w)/(c,h,w)/(n,c,h,w)).
+ V2.0: support input tensor with any dimension.
+ V2.1: base on V2.0, improve for if-branch.

# Envionment：
+ nvcr.io/nvidia/tensorrt:21.06-py3 (including CUDA 11.3.1, cudnn 8.2.1, TensorRT 7.2.3.4)

# Quick start：
```shell
make

make test
```

# Result:
```
python ./testCumSumPlugin.py
test [16] <class 'numpy.float32'> axis=0
Check result: True
test [16] <class 'numpy.float16'> axis=0
Check result: True
test [16] <class 'numpy.int32'> axis=0
Check result: True
test [2, 16] <class 'numpy.float32'> axis=1
Check result: True
test [2, 16] <class 'numpy.float16'> axis=1
Check result: True
test [2, 16] <class 'numpy.int32'> axis=1
Check result: True
test [2, 3, 16] <class 'numpy.float32'> axis=2
Check result: True
test [2, 3, 16] <class 'numpy.float16'> axis=2
Check result: True
test [2, 3, 16] <class 'numpy.int32'> axis=2
Check result: True
test [2, 3, 4, 16] <class 'numpy.float32'> axis=3
Check result: True
test [2, 3, 4, 16] <class 'numpy.float16'> axis=3
Check result: True
test [2, 3, 4, 16] <class 'numpy.int32'> axis=3
Check result: True
test [256] <class 'numpy.float32'> axis=0
Check result: True
test [256] <class 'numpy.float16'> axis=0
Check result: True
test [256] <class 'numpy.int32'> axis=0
Check result: True
test [2, 256] <class 'numpy.float32'> axis=1
Check result: True
test [2, 256] <class 'numpy.float16'> axis=1
Check result: True
test [2, 256] <class 'numpy.int32'> axis=1
Check result: True
test [2, 3, 256] <class 'numpy.float32'> axis=2
Check result: True
test [2, 3, 256] <class 'numpy.float16'> axis=2
Check result: True
test [2, 3, 256] <class 'numpy.int32'> axis=2
Check result: True
test [2, 3, 4, 256] <class 'numpy.float32'> axis=3
Check result: True
test [2, 3, 4, 256] <class 'numpy.float16'> axis=3
Check result: True
test [2, 3, 4, 256] <class 'numpy.int32'> axis=3
Check result: True
test [2, 16] <class 'numpy.float32'> axis=0
Check result: True
test [2, 16] <class 'numpy.float16'> axis=0
Check result: True
test [2, 16] <class 'numpy.int32'> axis=0
Check result: True
test [2, 3, 16] <class 'numpy.float32'> axis=1
Check result: True
test [2, 3, 16] <class 'numpy.float16'> axis=1
Check result: True
test [2, 3, 16] <class 'numpy.int32'> axis=1
Check result: True
test [2, 3, 4, 16] <class 'numpy.float32'> axis=2
Check result: True
test [2, 3, 4, 16] <class 'numpy.float16'> axis=2
Check result: True
test [2, 3, 4, 16] <class 'numpy.int32'> axis=2
Check result: True
test [2, 256] <class 'numpy.float32'> axis=0
Check result: True
test [2, 256] <class 'numpy.float16'> axis=0
Check result: True
test [2, 256] <class 'numpy.int32'> axis=0
Check result: True
test [2, 3, 256] <class 'numpy.float32'> axis=1
Check result: True
test [2, 3, 256] <class 'numpy.float16'> axis=1
Check result: True
test [2, 3, 256] <class 'numpy.int32'> axis=1
Check result: True
test [2, 3, 4, 256] <class 'numpy.float32'> axis=2
Check result: True
test [2, 3, 4, 256] <class 'numpy.float16'> axis=2
Check result: True
test [2, 3, 4, 256] <class 'numpy.int32'> axis=2
Check result: True
test [2, 3, 16] <class 'numpy.float32'> axis=0
Check result: True
test [2, 3, 16] <class 'numpy.float16'> axis=0
Check result: True
test [2, 3, 16] <class 'numpy.int32'> axis=0
Check result: True
test [2, 3, 4, 16] <class 'numpy.float32'> axis=1
Check result: True
test [2, 3, 4, 16] <class 'numpy.float16'> axis=1
Check result: True
test [2, 3, 4, 16] <class 'numpy.int32'> axis=1
Check result: True
test [2, 3, 256] <class 'numpy.float32'> axis=0
Check result: True
test [2, 3, 256] <class 'numpy.float16'> axis=0
Check result: True
test [2, 3, 256] <class 'numpy.int32'> axis=0
Check result: True
test [2, 3, 4, 256] <class 'numpy.float32'> axis=1
Check result: True
test [2, 3, 4, 256] <class 'numpy.float16'> axis=1
Check result: True
test [2, 3, 4, 256] <class 'numpy.int32'> axis=1
Check result: True
test [2, 3, 4, 16] <class 'numpy.float32'> axis=0
Check result: True
test [2, 3, 4, 16] <class 'numpy.float16'> axis=0
Check result: True
test [2, 3, 4, 16] <class 'numpy.int32'> axis=0
Check result: True
test [2, 3, 4, 256] <class 'numpy.float32'> axis=0
Check result: True
test [2, 3, 4, 256] <class 'numpy.float16'> axis=0
Check result: True
test [2, 3, 4, 256] <class 'numpy.int32'> axis=0
Check result: True
test finish!
```
