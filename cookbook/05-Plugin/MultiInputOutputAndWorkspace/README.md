# Plugin with multi input / output and workspace

+ We create a plugin with 2 input and 2 output tensors, using getWorkspaceSize method meanwhile.

+ Input tensor:
  + [0]: (nBatchSize, nLengthA)                     float32
  + [1]: (nBatchSize, nLengthB)                     float16
+ Input parameter:
  + None
+ Output tensor:
  + [0]: (nBatchSize, nLengthA * nLengthB)          float32, outer product A and B per batch
  + [1]: (nBatchSize, 1, min(nLengthA, nLengthB))   float32, elementwise sum of A and B, just keep the length of the shorter one

## Steps to run

```shell
make test
```