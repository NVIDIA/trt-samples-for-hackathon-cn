# GruPlugin
+ Use to implement a GRU Layer with early stop on the input tensor.
+ Not compatible for TensorRT8, need several edition before using in ensorRT8.
+ input tensor:
    - [0]: (nBatchSzie,maxSL,nDimHidden)    float32/float16, input tensor
    - [1]: (nBatchSzie)                     int32,           real sequence length of each input
+ input parameter:
    - [0]: nDimInput                        int32
    - [1]: nDimHidden                       int32
    - [2]: WeightX                          float32 [nDimInput*nDimHidden*3]
    - [3]: WeightH                          float32 [nDimHidden*nDimHidden*3]
    - [4]: Bias                             float32 [nDimHidden]
+ output tensor:
    - [0]: (nBatchSize,maxSL,nDimHidden)    float32,        all output hidden state
    - [1]: (nBatchSzie,nDimHidden)          float32,        final output hidden state of each input

# Envionment：
+ nvcr.io/nvidia/tensorrt:21.06-py3 (including CUDA 11.3.1, cudnn 8.2.1, TensorRT 7.2.3.4)
+ scipy

# Quick start：
```shell
make

make test
```

# Result:
```
test <class 'numpy.float32'> 0 time
Succeeded building engine!
Bind0-> (2, 40, 128) (2, 40, 128)
Bind1-> (2,) (2,)
Bind2-> (2, 40, 128) (2, 40, 128)
Bind3-> (2, 128) (2, 128)
InputH0-> (2, 40, 128) DataType.FLOAT
InputH1-> (2,) DataType.INT32
OutputH0-> (2, 40, 128) DataType.FLOAT
OutputH1-> (2, 128) DataType.FLOAT
True
True
test <class 'numpy.float32'> 0 time finish
test <class 'numpy.float32'> 1 time
Succeeded loading engine!
Bind0-> (2, 40, 128) (2, 40, 128)
Bind1-> (2,) (2,)
Bind2-> (2, 40, 128) (2, 40, 128)
Bind3-> (2, 128) (2, 128)
InputH0-> (2, 40, 128) DataType.FLOAT
InputH1-> (2,) DataType.INT32
OutputH0-> (2, 40, 128) DataType.FLOAT
OutputH1-> (2, 128) DataType.FLOAT
True
True
test <class 'numpy.float32'> 1 time finish
test <class 'numpy.float16'> 0 time
Succeeded building engine!
Bind0-> (2, 40, 128) (2, 40, 128)
Bind1-> (2,) (2,)
Bind2-> (2, 40, 128) (2, 40, 128)
Bind3-> (2, 128) (2, 128)
InputH0-> (2, 40, 128) DataType.HALF
InputH1-> (2,) DataType.INT32
OutputH0-> (2, 40, 128) DataType.HALF
OutputH1-> (2, 128) DataType.HALF
True
True
test <class 'numpy.float16'> 0 time finish
test <class 'numpy.float16'> 1 time
Succeeded loading engine!
Bind0-> (2, 40, 128) (2, 40, 128)
Bind1-> (2,) (2,)
Bind2-> (2, 40, 128) (2, 40, 128)
Bind3-> (2, 128) (2, 128)
InputH0-> (2, 40, 128) DataType.HALF
InputH1-> (2,) DataType.INT32
OutputH0-> (2, 40, 128) DataType.HALF
OutputH1-> (2, 128) DataType.HALF
True
True
test <class 'numpy.float16'> 1 time finish
test finish!
```

