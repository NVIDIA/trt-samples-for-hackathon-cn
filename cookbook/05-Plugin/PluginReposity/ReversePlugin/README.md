# RevesePlugin in MMDNN
+ Use to arrange the valid elements of each row of input tensor in reverse order.
+ Not compatible for TensorRT8, need several edition before using in ensorRT8.
+ input tensor:
    - [0]: (nGroup,nWidth,nEmbed)   int8/float16/int/float32,   each element is a vector with length nEmbed
    - [1]: (nGroup,)                int32,                      notes the number of valid elements of each row
+ output:
    - [0]: (nGroup,nWidth,nEmbed)   int8/float16/int/float32

# Envionment：
+ nvcr.io/nvidia/tensorrt:21.06-py3 (including CUDA 11.3.1, cudnn 8.2.1, TensorRT 7.2.3.4)

# Quick start：
```shell
make

make test
```

# Result:
```
python3 ./testReversePlugin.py
test [2, 4, 3] <class 'numpy.int32'>
succeeded building engine!
Check result: True
test [4, 9, 12] <class 'numpy.int32'>
succeeded building engine!
Check result: True
test [2, 4, 3] <class 'numpy.float32'>
succeeded building engine!
Check result: True
test [4, 9, 3] <class 'numpy.float32'>
succeeded building engine!
Check result: True
test [2, 4, 3] <class 'numpy.float16'>
succeeded building engine!
Check result: True
test [4, 9, 12] <class 'numpy.float16'>
succeeded building engine!
Check result: True
test finish!
```
