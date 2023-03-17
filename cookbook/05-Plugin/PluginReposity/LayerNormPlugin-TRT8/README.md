# LayerNormPlugin

+ Layer Normalization
+ Input tensor:
  + [0]: (n1, n2, ...,nK, nHiddenDimension)   float32/float16/int8, K>=0，被归一化的张量
  + [1]: (nK)                                 float32/float16，Gamma，归一化后的线性变换的乘数因子
  + [2]: (nK)                                 float32/float16，Beta，归一化后的线性变换的偏置因子
+ Input parameter:
  + [0]: nHiddenDimension                     int32，隐藏维的尺寸
  + [1]: epsilon                              float32，归一化标准差增量
+ Output tensor:
  + [0]: (n1, n2, ...,nK, nHiddenDimension)   float32/float16/int8
+ Steps to run：`make test`
+ Output for reference: ./result.log
+ 几个版本的对比
| 版本号 | 使用工具 |     支持输入数据类型     | 后续线性变换 | 支持的隐藏层宽度  | epsilon 传入方式 |
| :----: | :------: | :----------------------: | :----------: | :---------------: | :--------------: |
|   V1   | CUDA C++ |    float32 / float16     |      无      |        256        |      构建期      |
|   V2   |   CUB    |    float32 / float16     |      无      | 256（可按需添加） |      构建期      |
|   V3   |   CUB    |    float32 / float16     |      有      | 256（可按需添加） |      构建期      |
|   V4   |   CUB    | float32 / float16 / int8 |      有      |      $\ge 1$      |      构建期      |
|   V5   | OneFlow  |    float32 / float16     |      无      |      $\ge 1$      |      构建期      |

+ OneFlow 版本 LayerNorm 的源代码
<https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/layer_norm.cuh>
