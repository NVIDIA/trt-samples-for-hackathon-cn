# batchedNMSPlugin
+ 非最大值抑制（Batched Non-Maximum Suppression operation）
+ 参考 TensorRT 自带 Plugin 文档 [link](https://github.com/NVIDIA/TensorRT/tree/main/plugin/batchedNMSPlugin)
+ 输入张量:
    - [0]: (nBatchSize, nBox, nClass, nBoxParameter)    float32,    Box
    - [1]: (nBatchSize, nBox, nClass)                   float32,    Score
+ 输入参数:
    - [0]: shareLocation                                int32,
    - [1]: backgroundLabelId                            int32,
    - [2]: numClasses                                   int32,
    - [3]: topK                                         int32,
    - [4]: keepTopK                                     int32,
    - [5]: scoreThreshold                               float32,
    - [6]: iouThreshold                                 float32,
    - [7]: isNormalized                                 int32,
+ 输出张量:
    - [0]: (nBatchSize, 1)                              int32,      输出框数量
    - [1]: (nRetainSize, 4)                             float32,    输出框坐标
    - [2]: (nRetainSize, nKeepTopK)                     float32,    输出狂框得分
    - [3]: (nRetainSize, nKeepTopK)                     float32,    输出框类别号

+ Steps to run：`make test`
+ Output for reference: ./result.log
