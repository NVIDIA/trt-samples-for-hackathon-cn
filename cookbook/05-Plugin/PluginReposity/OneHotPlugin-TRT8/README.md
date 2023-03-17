# OneHotPlugin
+ 索引张量转 One Hot 编码
+ Input tensor:
    - [0]: (nBatchSize, n1, n2, ...,nK)             int32, K>=0
+ Input parameter:
    - [0]: nEmbedding                               int32, One Hot 输出编码宽度
+ Output tensor:
    - [0]: (nBatchSize, n1, n2, ...,nK, nEmbedding) float32/float16
+ Steps to run：`make test`
+ Output for reference: ./result.log

