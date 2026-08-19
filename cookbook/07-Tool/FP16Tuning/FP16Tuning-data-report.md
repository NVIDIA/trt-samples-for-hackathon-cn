# FP16 Tuning Report

+ Generated at 2026-09-08 09:32:46

+ Layers specified [  0]: []
+ Layers skipped [  0] : []
+ Layers forced in FP32 [  0]: []
+ Layers could be tuned [  7]: ['Pure FP32 🟩', 'Pure FP16 🟦', 'FP16 + ForceFP32 🟪', 'node_conv2d', 'node_conv2d_1', 'node_linear', 'node_linear_1']
+ Layers actually tune in this session: 4

+ Focus tensor for BestAcc ranking: y
|   No. | LayerName           | TensorName   |   GPUTime (ms) |   MaxAbsError |   MeanAbsError | BestPerf   | BestAcc   |
|-------|---------------------|--------------|----------------|---------------|----------------|------------|-----------|
|     1 | Pure FP32 🟩        | y            |          0.038 |             0 |              0 |            |           |
|     1 | Pure FP32 🟩        | z            |          0.038 |             0 |              0 |            |           |
|     2 | Pure FP16 🟦        | y            |          0.036 |      0.024628 |      0.0075182 |            |           |
|     2 | Pure FP16 🟦        | z            |          0.036 |             0 |              0 |            |           |
|     3 | FP16 + ForceFP32 🟪 | y            |          0.036 |      0.024628 |      0.0075182 |            |           |
|     3 | FP16 + ForceFP32 🟪 | z            |          0.036 |             0 |              0 |            |           |
|     4 | node_conv2d         | y            |          0.038 |      0.024628 |      0.0067464 | 1 🔴       | 2 🔴      |
|     4 | node_conv2d         | z            |          0.038 |             0 |              0 | 2 🔴       |           |
|     5 | node_conv2d_1       | y            |           0.04 |      0.024628 |      0.0075182 | 5 🔴       | 3 🔴      |
|     5 | node_conv2d_1       | z            |           0.04 |             0 |              0 | 6 🟠       |           |
|     6 | node_linear         | y            |           0.04 |      0.024628 |      0.0075182 | 7 🟠       | 4 🔴      |
|     6 | node_linear         | z            |           0.04 |             0 |              0 | 8 🟠       |           |
|     7 | node_linear_1       | y            |          0.038 |     0.0040989 |       0.002098 | 3 🔴       | 1 🔴      |
|     7 | node_linear_1       | z            |          0.038 |             0 |              0 | 4 🔴       |           |

+ Layers performs best in improving accuracy (sorted by `MaxAbsError`):

"node_linear_1", "node_conv2d", "node_conv2d_1", "node_linear",
