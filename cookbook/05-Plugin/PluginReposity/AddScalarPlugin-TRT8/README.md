# AddScalarPlugin

+ Add a scalar value to input tensor
+ Input tensor:
  + [0]: (nBatchSize, n1, n2, ...,nK) int32, K>=0
+ Input parameter:
  + [0]: scalar                       int32,
+ Output tensor:
  + [0]: (nBatchSize, n1, n2, ...,nK) float32/float16
+ Steps to run: `make test`
+ Output for reference: ./result.log
