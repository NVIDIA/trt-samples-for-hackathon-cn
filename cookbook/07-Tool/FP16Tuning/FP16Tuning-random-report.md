# FP16 Tuning Report

+ Generated at 2026-10-19 08:06:24

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
|     2 | Pure FP16 🟦        | y            |          0.028 |    3.5115e-05 |     1.1772e-05 |            |           |
|     2 | Pure FP16 🟦        | z            |          0.028 |             0 |              0 |            |           |
|     3 | FP16 + ForceFP32 🟪 | y            |          0.028 |    3.5115e-05 |     1.1772e-05 |            |           |
|     3 | FP16 + ForceFP32 🟪 | z            |          0.028 |             0 |              0 |            |           |
|     4 | node_conv2d         | y            |          0.028 |    3.5115e-05 |     1.1772e-05 | 1 🔴       | 2 🔴      |
|     4 | node_conv2d         | z            |          0.028 |             0 |              0 | 2 🔴       |           |
|     5 | node_conv2d_1       | y            |          0.034 |    3.5115e-05 |     1.0549e-05 | 7 🟠       | 3 🔴      |
|     5 | node_conv2d_1       | z            |          0.034 |             0 |              0 | 8 🟠       |           |
|     6 | node_linear         | y            |          0.032 |    3.5115e-05 |     1.1581e-05 | 5 🔴       | 4 🔴      |
|     6 | node_linear         | z            |          0.032 |             0 |              0 | 6 🟠       |           |
|     7 | node_linear_1       | y            |          0.028 |    9.9754e-06 |     2.4294e-06 | 3 🔴       | 1 🔴      |
|     7 | node_linear_1       | z            |          0.028 |             0 |              0 | 4 🔴       |           |

+ Layers performs best in improving accuracy (sorted by `MaxAbsError`):

"node_linear_1", "node_conv2d", "node_conv2d_1", "node_linear",
