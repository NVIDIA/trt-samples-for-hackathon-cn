# batchedNMSPlugin

+ Batched Non-Maximum Suppression operation
+ Refer to TensorRT Plugin [link](https://github.com/NVIDIA/TensorRT/tree/main/plugin/batchedNMSPlugin)
+ Input tensor:
  + [0]: (nBatchSize, nBox, nClass, nBoxParameter)    float32,    Box
  + [1]: (nBatchSize, nBox, nClass)                   float32,    Score
+ Input parameter:
  + [0]: shareLocation                                int32,
  + [1]: backgroundLabelId                            int32,
  + [2]: numClasses                                   int32,
  + [3]: topK                                         int32,
  + [4]: keepTopK                                     int32,
  + [5]: scoreThreshold                               float32,
  + [6]: iouThreshold                                 float32,
  + [7]: isNormalized                                 int32,
+ Output tensor:
  + [0]: (nBatchSize, 1)                              int32,      count of output box
  + [1]: (nRetainSize, 4)                             float32,    coordinates of output box
  + [2]: (nRetainSize, nKeepTopK)                     float32,    score of output box
  + [3]: (nRetainSize, nKeepTopK)                     float32,    kind index of output box

+ Steps to run：`make test`
+ Output for reference: ./result.log
