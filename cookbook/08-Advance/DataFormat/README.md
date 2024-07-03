# Use PluginV2DynamicExt

+ Steps to run

```bash
make test
```

## TensorFormat

+ [Link](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html#ac3e115b1a2b1e578e8221ef99d27cd45)

|  Format   |  LINEAR  | CHW2 | HWC8 | CHW4 | CHW16 | CHW32 | DHWC8 | CDHW32 | HWC  | DLA_LINEAR | DLA_HWC4 | HWC16 | DHWC |
| :-------: | :------: | :--: | :--: | :--: | :---: | :---: | :---: | :----: | :--: | :--------: | :------: | :---: | :--: |
|    VTF    |    0     |  1   |  2   |  3   |   4   |   5   |   6   |   7    |  8   |     9      |    10    |  11   |  12  |
|    mD     |    0     |  3   |  3   |  3   |   3   |   3   |   4   |   4    |  3   |     0      |    0     |   3   |  4   |
|   FP32    |    1     |      |      |      |       |   1   |       |        |  1   |            |          |       |  1   |
|   FP16    |    1     |  1   |  1   |  1   |   1   |   1   |   1   |   1    |      |            |          |   1   |      |
|   INT8    |    1     |      |      |  1   |       |   1   |       |   1    |      |            |          |       |      |
|   INT32   |    1     |      |      |      |       |       |       |        |      |            |          |       |      |
|   BOOL    |          |      |      |      |       |       |       |        |      |            |          |       |      |
|   UINT8   |    1     |      |      |      |       |       |       |        |  1   |            |          |       |      |
|    FP8    |          |      |      |      |       |       |       |        |      |            |          |       |      |
|   BF16    |          |      |      |      |       |       |       |        |      |            |          |       |      |
|   INT64   |          |      |      |      |       |       |       |        |      |            |          |       |      |
|   INT4    |          |      |      |      |       |       |       |        |      |            |          |       |      |
|    FP4    |          |      |      |      |       |       |       |        |      |            |          |       |      |
| DLA_FP32  |          |      |      |      |       |       |       |        |      |     1      |    1     |       |      |
| DLA_FP16  |          |      |      |      |       |       |       |        |      |     1      |    1     |       |      |
| DLA_INT32 |          |      |      |      |       |       |       |        |      |     1      |    1     |       |      |
| DLA_INT8  |          |      |      |      |       |       |       |        |      |     1      |    1     |       |      |

|   Format   | Description                                                  |
| :--------: | :----------------------------------------------------------- |
|   LINEAR   | layer Row major linear format.                               |
|    CHW2    | 2 wide channel vectorized row major format. Memory layout [N][(C+1)/2][H][W][2], coordinate (n, c, h, w) maps to [n][c/2][h][w][c%2]. |
|    HWC8    | 8 channel format where C is padded to a multiple of 8. Memory layout [N][H][W][(C+7)/8*8], (n, c, h, w) maps to [n][h][w][c]. |
|    CHW4    | 4 wide channel vectorized row major format. Memory layout [N][(C+3)/4][H][W][4], coordinate (n, c, h, w) maps to [n][c/4][h][w][c%4]. |
|   CHW16    | 16 wide channel vectorized row major format. Memory layout [N][(C+15)/16][H][W][16], coordinate (n, c, h, w) maps to [n][c/16][h][w][c%16]. |
|   CHW32    | 32 wide channel vectorized row major format. Memory layout [N][(C+31)/32][H][W][32], coordinate (n, c, h, w) maps to [n][c/32][h][w][c%32]. |
|   DHWC8    | 8 channel format where C is padded to a multiple of 8. Memory layout [N][D][H][W][(C+7)/8*8], coordinate (n, c, d, h, w) maps to [n][d][h][w][c]. |
|   CDHW32   | 32 wide channel vectorized row major format. Memory layout [N][(C+31)/32][D][H][W][32], coordinate (n, c, d, h, w) maps to [n][c/32][d][h][w][c%32]. |
|    HWC     | Non-vectorized channel-last format.                          |
| DLA_LINEAR | DLA format. Memory layout [N][C][H][roundUp(W, 64/elementSize)], coordinate (n, c, h, w) maps to [n][c][h][w]. |
|  DLA_HWC4  | DLA format. Memory layout [N][H][roundUp(W, 32/C'/elementSize)][C'] on Xavier and [N][H][roundUp(W, 64/C'/elementSize)][C'] on Orin, coordinate (n, c, h, w) maps to [n][h][w][c]. |
|   HWC16    | 16 channel format where C padded to a multiple of 16. Memory layout [N][H][W][(C+15)/16*16], coordinate (n, c, h, w) maps to [n][h][w][c]. |
|    DHWC    | Non-vectorized channel-last format.                          |

+ VTF: Value of int(trt.TensorFormat.XXX)
+ mD: minimum dimension of the tensor to use this format
