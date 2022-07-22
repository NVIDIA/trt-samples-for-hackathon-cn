# OneHotPlugin
+ 索引张量转 One Hot 编码
+ 输入张量:
    - [0]: (nBatchSize, n1, n2, ...,nK)             int32, K>=0
+ 输入参数:
    - [0]: nEmbedding                               int32, One Hot 输出编码宽度
+ 输出张量:
    - [0]: (nBatchSize, n1, n2, ...,nK, nEmbedding) float32/float16
+ 运行方法：`make test`
+ 参考输出结果，见 ./result.txt

