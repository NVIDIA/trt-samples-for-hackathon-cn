# Resize2DPlugin
+ 给输入张量做插值操作
+ 输入张量:
    - [0]: (N, C, H, W)   float32/float16
+ 输入参数:
    - [0]: Mode           int32，插值方式，0（最近邻插值），1（双线性插值）
    - [1]: Scale          int32，放大比例（目前仅支持 int32 型输入），与 OutputHeight/OutputWidth 不可同时使用
    - [2]: OutputHeight   int32，输出张量 H 维度尺寸，与 Scale 不可同时使用
    - [3]: OutputWidth    int32，输出张量 W 维度尺寸，与 Scale 不可同时使用
+ 输出张量:
    - [0]: (N, C, H * Scale, W * Scale) 或 (N, C, OutputHeight, OutputWidth)   float32/float16
+ 运行方法：`make test`
+ 参考输出结果，见 ./result.log
+ 几个版本的对比
| 版本号 | 支持数据排布 |   支持数据类型    |
| :----: | :----------: | :---------------: |
|   V1   | Linear(NCHW) | float32 / float16 |
|   V2   |  NHWC(HWC8)  | float32 / float16 |
