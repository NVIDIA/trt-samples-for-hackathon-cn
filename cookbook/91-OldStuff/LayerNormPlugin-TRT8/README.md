# LayerNormPlugin

+ Layer Normalization plugin, with several implementations.

+ Note: this example targets an older version of TensorRT (TensorRT 8) and uses the deprecated `IPluginV2DynamicExt` plugin API. It is kept for reference only and is not expected to build or run on recent TensorRT.

+ Input tensor:
  + [0]: (n1, n2, ..., nK, nHiddenDimension)   float32 / float16 / int8, K >= 0, the tensor to be normalized
  + [1]: (nK)                                  float32 / float16, Gamma, scale factor of the affine transform after normalization
  + [2]: (nK)                                  float32 / float16, Beta, bias of the affine transform after normalization

+ Input parameter:
  + [0]: nHiddenDimension                      int32, size of the hidden dimension
  + [1]: epsilon                               float32, small value added to the variance for numerical stability

+ Output tensor:
  + [0]: (n1, n2, ..., nK, nHiddenDimension)   float32 / float16 / int8

+ Steps to run.

```bash
make test
```

+ Comparison of the implementations

| Version | Toolkit  |   Supported input dtype  | Affine transform | Supported hidden width | epsilon passed at |
| :-----: | :------: | :----------------------: | :--------------: | :--------------------: | :---------------: |
|   V1    | CUDA C++ |    float32 / float16     |        No        |          256           |    build time     |
|   V2    |   CUB    |    float32 / float16     |        No        |    256 (extendable)    |    build time     |
|   V3    |   CUB    |    float32 / float16     |       Yes        |    256 (extendable)    |    build time     |
|   V4    |   CUB    | float32 / float16 / int8 |       Yes        |        $\ge 1$         |    build time     |
|   V5    | OneFlow  |    float32 / float16     |        No        |        $\ge 1$         |    build time     |

+ Source code of the OneFlow LayerNorm implementation:
<https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/layer_norm.cuh>
