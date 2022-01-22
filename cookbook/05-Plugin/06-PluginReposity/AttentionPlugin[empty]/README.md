# AttentionPlugin
+ Use to 
+ Not compatible for TensorRT8, need several edition before using in ensorRT8.
+ input tensor:
    - [0]: (nBatchSize,nBox,nClass,nBoxParameter)   float32,    Box
    - [1]: (nBatchSize,nBox,nClass)                 float32,    Score
+ input parameter:
    - [0]: shareLocation                            int32,
    - [1]: backgroundLabelId                        int32,
    - [2]: numClasses                               int32,
    - [3]: topK                                     int32,
    - [4]: keepTopK                                 int32,
    - [5]: scoreThreshold                           float32,
    - [6]: iouThreshold                             float32,
    - [7]: isNormalized                             int32,
+ output tensor:
    - [0]: (nBatchSize,1)                           int32,      number of output
    - [1]: (nRetainSize,4)                          float32,    number of non-max suppressed box
    - [2]: (nRetainSize,nKeepTopK)                  float32,    score for the box
    - [3]: (nRetainSize,nKeepTopK)                  float32,    class for the box

# Envionment：
+ nvcr.io/nvidia/tensorrt:21.06-py3 (including CUDA 11.3.1, cudnn 8.2.1, TensorRT 7.2.3.4)

# Quick start：
```shell

```

# Result:
```

```
