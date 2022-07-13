# TopKAveragePlugin
+ Use to calculate topk operation and following average operation base on the given nTopK.
+ Not compatible for TensorRT8, need several edition before using in ensorRT8.
+ input tensor:
    - [0]: (nBatchSize,nGroup,nChannel,nHeight,nWidth)  float/float16,
    - [1]: (nBatchSize,nGroup)                          int,            number of valid element in row
    - [2]: (nBatchSize,nGroup)                          int,            number of valid element in column
+ input parameter:
    - [0]: nTopK                                        int,
    - [1]: maxTopK                                      int,
+ output tensor:
    - [0]: (nBatchSize, nChannel, nGroup * nTopK)       float32/float16,

# Envionment：
+ nvcr.io/nvidia/tensorrt:21.06-py3 (including CUDA 11.3.1, cudnn 8.2.1, TensorRT 7.2.3.4)

# Quick start：
```shell
make

make test1

#make test2 # need input file testTopKAveragePlugin.npz
```

# Result:
+ make test1
```
python3 ./testTopKAveragePlugin.py
test (36, 10, 5, 30) <class 'numpy.float32'> [2, 3, 4]
Succeeded building engine!
All bind: True
Check result: True
test (36, 8, 5, 65) <class 'numpy.float32'> [1, 2, 5, 12]
Succeeded building engine!
All bind: True
Check result: True
test (36, 18, 5, 70) <class 'numpy.float32'> [1, 2, 5, 12]
Succeeded building engine!
All bind: True
Check result: True
test finish!
```

+ make test2
```
python3 ./testTopKAveragePlugin-useDataFromModel.py
test (36, 10, 5, 30) <class 'numpy.float32'> [1, 2, 4]
Succeeded building engine!
All bind: True
Check result: 9.072246e-07
test finish!

```

