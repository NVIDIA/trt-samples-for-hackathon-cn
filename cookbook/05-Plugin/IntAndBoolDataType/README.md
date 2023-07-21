# Plugin with multi input / output and workspace

+ We create a plugin with 2 input and 1 output tensors, using int32 and bool data type

+ Input tensor:
  + [0]: (nBatchSize, nLength)                    int32
  + [1]: (nBatchSize, nLength)                    bool
+ Input parameter:
  + None
+ Output tensor:
  + [0]: (nBatchSize, nLength)        float32, elementwise multiplication of the inputs

## Steps to run

```shell
make test
```
