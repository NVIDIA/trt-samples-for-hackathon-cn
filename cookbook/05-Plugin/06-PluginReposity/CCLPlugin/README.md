# CCLPlugin
+ Use to do Connected component label operation.
+ Not compatible for TensorRT8, need several edition before using in ensorRT8.
+ input tensor:
    - [0]: (nBatchSize,nHeight,nWidth)      float32, 
    - [1]: (nBatchSize,8,nHeight,nWidth)    float32,
+ input parameter:
    - [0]: minPixelScore                    float32,
    - [1]: minLinkScore                     float32,
    - [2]: minArea                          int32,
    - [3]: maxcomponentCount                int32,
+ output tensor:
    - [0]: (nBatchSize,nHeight,nWidth)      int32,
    - [1]: (nBatchSize,)                    int32,

# Envionment：
+ nvcr.io/nvidia/tensorrt:21.06-py3 (including CUDA 11.3.1, cudnn 8.2.1, TensorRT 7.2.3.4)
+ nvcr.io/nvidia/tensorrt:21.09-py3 (including CUDA 11.4.2, cudnn 8.2.4.15, TensorRT 8.0.3)

# Quick start：
```shell
make

make test
```

# Result:
```
python ./testCCLPlugin.py
test [1, 1, 1]
[TensorRT] INFO: Detected 2 inputs and 2 output network tensors.
Succeeded building engine!
(1, 1, 1) (1,)
[[[2147483647]]]
[0]
2147483647 2147483647.0 0
test [2, 384, 640]
[TensorRT] INFO: Detected 2 inputs and 2 output network tensors.
Succeeded building engine!
(2, 384, 640) (2,)
[[[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]]
[0 0]
0 0.0 0
test [4, 768, 1280]
[TensorRT] INFO: Detected 2 inputs and 2 output network tensors.
Succeeded building engine!
(4, 768, 1280) (4,)
[[[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]]
[2147483647 2147483647 2147483647 2147483647]
0 0.0 0
test finish!
```
