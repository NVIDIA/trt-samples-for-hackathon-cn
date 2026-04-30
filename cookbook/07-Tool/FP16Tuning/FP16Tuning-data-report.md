# FP16 Tuning Report

+ Generated at 2026-07-07 01:58:09

+ Layers specified [  0]: []
+ Layers skipped [  0] : []
+ Layers forced in FP32 [  0]: []
+ Layers could be tuned [  7]: ['Pure FP32 🟩', 'Pure FP16 🟦', 'FP16 + ForceFP32 🟪', 'node_conv2d', 'node_conv2d_1', 'node_linear', 'node_linear_1_2']
+ Layers actually tune in this session: 4

+ Focus tensor for BestAcc ranking: y
|   No. | LayerName           | TensorName   |   GPUTime (ms) |   MaxAbsError |   MeanAbsError | BestPerf   | BestAcc   |
|-------|---------------------|--------------|----------------|---------------|----------------|------------|-----------|
|     1 | Pure FP32 🟩        | y            |           0.04 |             0 |              0 |            |           |
|     1 | Pure FP32 🟩        | z            |           0.04 |             0 |              0 |            |           |
|     2 | Pure FP16 🟦        | y            |          0.039 |     0.0090027 |      0.0027737 |            |           |
|     2 | Pure FP16 🟦        | z            |          0.039 |             0 |              0 |            |           |
|     3 | FP16 + ForceFP32 🟪 | y            |          0.039 |     0.0090027 |      0.0027737 |            |           |
|     3 | FP16 + ForceFP32 🟪 | z            |          0.039 |             0 |              0 |            |           |
|     4 | node_conv2d         | y            |          0.039 |     0.0090027 |      0.0027737 | 3 🔴       | 1 🔴      |
|     4 | node_conv2d         | z            |          0.039 |             0 |              0 | 4 🔴       |           |
|     5 | node_conv2d_1       | y            |           0.04 |     0.0090027 |      0.0027371 | 7 🟠       | 2 🔴      |
|     5 | node_conv2d_1       | z            |           0.04 |             0 |              0 | 8 🟠       |           |
|     6 | node_linear         | y            |          0.039 |     0.0090027 |      0.0021493 | 5 🔴       | 3 🔴      |
|     6 | node_linear         | z            |          0.039 |             0 |              0 | 6 🟠       |           |
|     7 | node_linear_1_2     | y            |          0.038 |     0.0090027 |      0.0027737 | 1 🔴       | 4 🔴      |
|     7 | node_linear_1_2     | z            |          0.038 |             0 |              0 | 2 🔴       |           |

+ Layers performs best in improving accuracy (sorted by `MaxAbsError`):

"node_conv2d", "node_conv2d_1", "node_linear", "node_linear_1_2",
