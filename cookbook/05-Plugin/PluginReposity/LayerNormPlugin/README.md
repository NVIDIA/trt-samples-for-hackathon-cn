# LayerNorm in MMDNN
+ Use to do layer normoalization on the input tensor.
+ input tensor:
    - [0]: (nBatchSize,nSequenceLength,nEmbed)   float16/float32,   nEmbed only support 320 or 560 now, but you can change it to any compile-time constant 
+ output:
    - [0]: (nBatchSize,nSequenceLength,nEmbed)   float16/float32,

# Envionment：
+ nvcr.io/nvidia/tensorrt:21.09-py3 (including CUDA 11.4.2, cuDNN 8.2.4.15, python 3.8, TensorRT 8.0.3)

# Quick start：
```shell
make

make test
```

# Result:
```
python ./testReversePlugin.py
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
