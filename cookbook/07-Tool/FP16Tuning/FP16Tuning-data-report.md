# FP16 Tuning Report

+ Generated at 2026-10-19 08:05:51

+ Layers specified [  0]: []
+ Layers skipped [  0] : []
+ Layers forced in FP32 [  0]: []
+ Layers could be tuned [  7]: ['Pure FP32 🟩', 'Pure FP16 🟦', 'FP16 + ForceFP32 🟪', 'node_conv2d', 'node_conv2d_1', 'node_linear', 'node_linear_1']
+ Layers actually tune in this session: 4

+ Focus tensor for BestAcc ranking: y
|   No. | LayerName           | TensorName   |   GPUTime (ms) |   MaxAbsError |   MeanAbsError | BestPerf   | BestAcc   |
|-------|---------------------|--------------|----------------|---------------|----------------|------------|-----------|
|     1 | Pure FP32 🟩        | y            |          0.036 |             0 |              0 |            |           |
|     1 | Pure FP32 🟩        | z            |          0.036 |             0 |              0 |            |           |
|     2 | Pure FP16 🟦        | y            |          0.028 |      0.019978 |      0.0070322 |            |           |
|     2 | Pure FP16 🟦        | z            |          0.028 |             0 |              0 |            |           |
|     3 | FP16 + ForceFP32 🟪 | y            |          0.028 |      0.019978 |      0.0070322 |            |           |
|     3 | FP16 + ForceFP32 🟪 | z            |          0.028 |             0 |              0 |            |           |
|     4 | node_conv2d         | y            |          0.028 |      0.029428 |       0.010647 | 1 🔴       | 4 🔴      |
|     4 | node_conv2d         | z            |          0.028 |             0 |              0 | 2 🔴       |           |
|     5 | node_conv2d_1       | y            |          0.034 |      0.013803 |      0.0046896 | 7 🟠       | 1 🔴      |
|     5 | node_conv2d_1       | z            |          0.034 |             0 |              0 | 8 🟠       |           |
|     6 | node_linear         | y            |          0.032 |      0.019978 |      0.0078623 | 5 🔴       | 3 🔴      |
|     6 | node_linear         | z            |          0.032 |             0 |              0 | 6 🟠       |           |
|     7 | node_linear_1       | y            |          0.028 |      0.015139 |      0.0047793 | 3 🔴       | 2 🔴      |
|     7 | node_linear_1       | z            |          0.028 |             0 |              0 | 4 🔴       |           |

+ Layers performs best in improving accuracy (sorted by `MaxAbsError`):

"node_conv2d_1", "node_linear_1", "node_linear", "node_conv2d",
