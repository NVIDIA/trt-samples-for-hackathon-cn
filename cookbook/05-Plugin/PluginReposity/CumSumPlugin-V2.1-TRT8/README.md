# CumSumPlugin
+ 在Input tensor的指定维度上做闭前缀和
+ Input tensor:
    - [0]: (nBatchSize, n1, n2, ...,nK) float32/float16/int32, K>=0
+ Input parameter:
    - [0]: axis                         int32, axis >=0 && axis <= K, 需要做前缀和的维度
+ Output tensor:
    - [0]: (nBatchSize, n1, n2, ...,nK) float32/float16/int32
+ Steps to run：`make test`
+ Output for reference: ./result.log
