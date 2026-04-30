# FP16 Tuning Report

+ Generated at 2026-07-02 03:39:21

+ Layers specified [  0]: []
+ Layers skipped [  0] : []
+ Layers forced in FP32 [  0]: []
+ Layers could be tuned [  7]: ['Pure FP32 🟩', 'Pure FP16 🟦', 'FP16 + ForceFP32 🟪', 'node_conv2d', 'node_conv2d_1', 'node_linear', 'node_linear_1_2']
+ Layers actually tune in this session: 4

+ Focus tensor for BestAcc ranking: y
|   No. | LayerName           | TensorName   |   GPUTime (ms) |   MaxAbsError |   MeanAbsError | BestPerf   | BestAcc   |
|-------|---------------------|--------------|----------------|---------------|----------------|------------|-----------|
|     1 | Pure FP32 🟩        | y            |          0.034 |             0 |              0 |            |           |
|     1 | Pure FP32 🟩        | z            |          0.034 |             0 |              0 |            |           |
|     2 | Pure FP16 🟦        | y            |           0.03 |      0.022247 |      0.0095368 |            |           |
|     2 | Pure FP16 🟦        | z            |           0.03 |             0 |              0 |            |           |
|     3 | FP16 + ForceFP32 🟪 | y            |           0.03 |      0.022247 |      0.0095368 |            |           |
|     3 | FP16 + ForceFP32 🟪 | z            |           0.03 |             0 |              0 |            |           |
|     4 | node_conv2d         | y            |           0.03 |      0.022247 |       0.010212 | 1 🔴       | 2 🔴      |
|     4 | node_conv2d         | z            |           0.03 |             0 |              0 | 2 🔴       |           |
|     5 | node_conv2d_1       | y            |          0.034 |      0.022247 |       0.003959 | 7 🟠       | 3 🔴      |
|     5 | node_conv2d_1       | z            |          0.034 |             0 |              0 | 8 🟠       |           |
|     6 | node_linear         | y            |           0.03 |      0.022247 |      0.0095368 | 3 🔴       | 4 🔴      |
|     6 | node_linear         | z            |           0.03 |             0 |              0 | 4 🔴       |           |
|     7 | node_linear_1_2     | y            |           0.03 |      0.015971 |      0.0069489 | 5 🔴       | 1 🔴      |
|     7 | node_linear_1_2     | z            |           0.03 |             0 |              0 | 6 🟠       |           |

+ Layers performs best in improving accuracy (sorted by `MaxAbsError`):

"node_linear_1_2", "node_conv2d", "node_conv2d_1", "node_linear",
