# CumSumPlugin
+ 在输入张量的指定维度上做闭前缀和
+ 输入张量:
    - [0]: (nBatchSize, n1, n2, ...,nK) float32/float16/int32, K>=0
+ 输入参数:
    - [0]: axis                         int32, axis >=0 && axis <= K, 需要做前缀和的维度
+ 输出张量:
    - [0]: (nBatchSize, n1, n2, ...,nK) float32/float16/int32
+ 运行方法：`make test`
+ 参考输出结果，见 ./result.txt
