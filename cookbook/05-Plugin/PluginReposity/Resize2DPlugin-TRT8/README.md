# Resize2DPlugin
+ 给Input tensor最末尾两维度做插值操作
+ Input tensor:
    - [0]: (N, C, H, W)   float32/float16
+ Input parameter:
    - [0]: Mode           int32，插值方式，0（最近邻插值），1（双线性插值）
    - [1]: Scale          int32，放大比例（目前仅支持 int32 型输入），与 OutputHeight/OutputWidth 不可同时使用
    - [2]: OutputHeight   int32，输出张量 H 维度尺寸，与 Scale 不可同时使用
    - [3]: OutputWidth    int32，输出张量 W 维度尺寸，与 Scale 不可同时使用
+ Output tensor:
    - [0]: (N, C, H * Scale, W * Scale) 或 (N, C, OutputHeight, OutputWidth)   float32/float16
+ Steps to run：`make test`
+ Output for reference: ./result.log
+ 几个版本的对比
| 版本号 | 支持数据排布 |   支持数据类型    |  支持插值方法   |
| :----: | :----------: | :---------------: | :-------------: |
|   V1   | Linear(NCHW) | float32 / float16 | 最近邻 / 双线性 |
|   V2   |  NHWC(HWC8)  |      float16      | 最近邻 / 双线性 |

+ 默认插值算法的角落的对齐方法同 TensorRT Resize 层的 HALF_PIXEL 方法，若要使用其他对齐方法，则需要修改插值 kernel 中的 alpha 和 beta，可用修改方法可参考 02-API/Layer/ResizeLayer/README.md
