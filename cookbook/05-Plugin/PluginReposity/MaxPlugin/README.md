# MaxPlugin in MMDNN
+ Use to do max-reduce operation on the 1st dimension of the input tensor and discard this axis.
+ Not compatible for TensorRT8, need several edition before using in ensorRT8.
+ input tensor:
    - [0]: (nGroup,nWidth,nEmbed)   float/float16,  each element is a vector with length nEmbed
    - [1]: (nGroup,)                int32,          notes the number of valid elements of each row
+ output tenos:
    - [0]: (nGroup,nEmbed)          float/float16,

# Envionment：
+ nvcr.io/nvidia/tensorrt:21.06-py3 (including CUDA 11.3.1, cudnn 8.2.1, TensorRT 7.2.3.4)

# Quick start：
```shell
make

make test
```

# Result:
```
python3 ./testMaxPlugin.py
test [4, 3, 5] <class 'numpy.float32'>
Succeeded building engine!
All bind: True
Check result: True
test [9, 12, 6] <class 'numpy.float32'>
Succeeded building engine!
All bind: True
Check result: True
test [4, 3, 5] <class 'numpy.float16'>
Succeeded building engine!
All bind: True
Check result: True
test [9, 12, 6] <class 'numpy.float16'>
Succeeded building engine!
All bind: True
Check result: True
test finish!

```
