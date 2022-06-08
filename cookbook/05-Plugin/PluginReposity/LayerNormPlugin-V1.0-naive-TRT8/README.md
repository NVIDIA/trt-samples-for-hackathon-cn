# LayerNormPlugin
+ 给输入张量做 Layer Normalization 操作
+ 输入张量:
    - [0]: (n1, n2, ...,nK, nHiddenDimension)   float32/float16/int8, K>=0，被归一化的张量
    - [1]: (nK)                                 float32/float16，Gamma，归一化后的线性变换的乘数因子
    - [2]: (nK)                                 float32/float16，Beta，归一化后的线性变换的偏置因子
+ 输入参数:
    - [0]: nHiddenDimension                     int32，隐藏维的尺寸
    - [1]: epsilon                              float32，归一化标准差增量
+ 输出张量:
    - [0]: (n1, n2, ...,nK, nHiddenDimension)   float32/float16/int8
+ 环境：nvcr.io/nvidia/tensorrt:21.12-py3（包含 python 3.8.10，CUDA 11.5.50，cuDNN 8.3.1，TensoRT 8.2.3）
+ 运行方法：`make test`
+ 参考输出结果，见 ./result.txt
+ 几个版本的对比
| 版本号 | 使用工具 |     支持输入数据类型     | 后续线性变换 | 支持的隐藏层宽度  | epsilon 传入方式 |
| :----: | :------: | :----------------------: | :----------: | :---------------: | :--------------: |
|  V1.0  | CUDA C++ |    float32 / float16     |      无      |        256        |      构建期      |
|  V2.0  |   CUB    |    float32 / float16     |      无      | 256（可按需添加） |      构建期      |
|  V2.1  |   CUB    |    float32 / float16     |      有      | 256（可按需添加） |      构建期      |
|  V2.2  |   CUB    | float32 / float16 / int8 |      有      |      $\ge 1$      | 用宏写在 .cu 中  |
|  V3.0  | OneFlow  |    float32 / float16     |      无      |      $\ge 1$      |      构建期      |

