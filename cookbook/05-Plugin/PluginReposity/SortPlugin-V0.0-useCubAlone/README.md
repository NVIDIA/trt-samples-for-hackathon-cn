# SortPlugin
+ Use to sort data using cub
+ ./sortByCub contains a example of sort using cub.
+ ./float contains a example of sort with datatype float.
+ ./float4 contains a example of sort with datatype float4.
+ Not compatible for TensorRT8 or dynamic shape mode, need several edition before using in ensorRT8.
+ input tensor:
    - [0]: (nElement,1)         float32, data of key
    - [1]: (nElement,nWidth)    float32, data of value
+ input parameter:
    - [0]: descending           int,
+ output tensor:
    - [0]: (nElement,1)         float32, data of key sorted
    - [1]: (nElement,nWidth)    float32, data of value sorted

# Envionment：
+ nvcr.io/nvidia/tensorrt:21.06-py3 (including CUDA 11.3.1, cudnn 8.2.1, TensorRT 7.2.3.4)

# Quick start：
```shell
cd ./sortByCub (or cd ./float or cd ./float4)

make

make test
```

# Result:
+ cubSort.exe
```
before sort, dKey.selector = 0, dValue.selector = 0
after sort, dKey.selector = 1, dValue.selector = 1
Test succeed!
  0: input(1.8043E+09,  0), outputCPU(3.5005E+07, 20), outputGPU(3.5005E+07, 20)
  1: input(8.4693E+08,  1), outputCPU(4.2999E+07, 57), outputGPU(4.2999E+07, 57)
  2: input(1.6817E+09,  2), outputCPU(8.4354E+07, 71), outputGPU(8.4354E+07, 71)

...

147: input(1.5734E+09,147), outputCPU(2.0890E+09, 37), outputGPU(2.0890E+09, 37)
148: input(1.4100E+09,148), outputCPU(2.1147E+09,124), outputGPU(2.1147E+09,124)
149: input(2.0775E+09,149), outputCPU(2.1452E+09, 28), outputGPU(2.1452E+09, 28)

```

+ float
```
python3 ./testSortPlugin.py
+--------nElement = 1024, width = 1, workSpaceSize = 1024
Succeeded building engine!
(1024, 1) (1024, 1)
Check result Key: True
Check result Value: True
test finish!
```

+ float4
```
python3 ./testSortPlugin.py
+--------nElement = 128, width = 4, workSpaceSize = 1024
Succeeded building engine!
(128, 1) (128, 4)
Check result Key: True
Check result Value: True
test finish!
```
