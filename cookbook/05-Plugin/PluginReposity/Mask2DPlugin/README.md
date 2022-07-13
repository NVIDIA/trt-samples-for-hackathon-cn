# MAsk2DPlugin in MMDNN
+ Use to create a 2D Mask from input tensor
+ Not compatible for TensorRT8, need several edition before using in ensorRT8.
+ input tensor:
    - [0]: (nBatchSize,nSequenceLength,nHeight,nWidth)  float16/float32/int32,  input tensor, providing nHeight and nWidth
    - [1]: (nRealHeight)                                int32,                  real height, judgement of element in row [nRealHeight:] is False
    - [2]: (nRealWidth)                                 int32,                  real width, judgement of element in column [nRealWidth:] is False
+ input parameter:
    - [0]: datatype                                     int32,                  output mask datatype
    - [1]: mask2DTrueValue                              float32                 mask value when the judgement is True
    - [2]: mask2DFalseValue                             float32                 mask value when the judgement is False
+ output:
    - [0]: (nBatchSize,1,nHeight,nWidth)                float16/float32

# Envionment：
+ nvcr.io/nvidia/tensorrt:21.06-py3 (including CUDA 11.3.1, cudnn 8.2.1, TensorRT 7.2.3.4)

# Quick start：
```shell
make

make test
```

# Result:
```
python3 ./testMask2DPlugin.py
test [4, 3, 30, 40] <class 'numpy.float32'>
Succeeded building engine!
All bind: True
Check result: True
test [4, 3, 30, 40] <class 'numpy.float16'>
Succeeded building engine!
All bind: True
Check result: True
test finish!
```
