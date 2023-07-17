# CumSumPlugin
+ Do inclusive accumulative sum on the input tensor
+ Input tensor:
    - [0]: (nBatchSize, n1, n2, ...,nK) float32/float16/int32, K>=0
+ Input parameter:
    - [0]: axis                         int32, axis >=0 && axis <= K, the axis the sum executes
+ Output tensor:
    - [0]: (nBatchSize, n1, n2, ...,nK) float32/float16/int32
+ Steps to run：`make test`
