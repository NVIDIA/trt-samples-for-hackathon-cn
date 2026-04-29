# FP16 Tuning Report

+ Generated at 2026-04-29 03:08:27

+ Layers specified [  0]: []
+ Layers skipped [  0] : []
+ Layers forced in FP32 [  0]: []
+ Layers could be tuned [  7]: ['Pure FP32 🟩', 'Pure FP16 🟦', 'FP16 + ForceFP32 🟪', 'node_conv2d', 'node_conv2d_1', 'node_linear', 'node_linear_1_2']
+ Focus tensor for BestAcc ranking: y
+ Layers actually tune in this session: 4

|   No. | LayerName           | TensorName   |   GPUTime (ms) |   MaxAbsError |   MeanAbsError | BestPerf   | BestAcc   |
|-------|---------------------|--------------|----------------|---------------|----------------|------------|-----------|
|     1 | Pure FP32 🟩        | y            |          0.033 |             0 |              0 |            |           |
|     1 | Pure FP32 🟩        | z            |          0.033 |             0 |              0 |            |           |
|     2 | Pure FP16 🟦        | y            |           0.03 |      0.042665 |       0.012763 |            |           |
|     2 | Pure FP16 🟦        | z            |           0.03 |             0 |              0 |            |           |
|     3 | FP16 + ForceFP32 🟪 | y            |           0.03 |      0.042665 |       0.012763 |            |           |
|     3 | FP16 + ForceFP32 🟪 | z            |           0.03 |             0 |              0 |            |           |
|     4 | node_conv2d         | y            |           0.03 |       0.02704 |       0.010407 | 3 🔴       | 2 🔴      |
|     4 | node_conv2d         | z            |           0.03 |             0 |              0 | 4 🔴       |           |
|     5 | node_conv2d_1       | y            |          0.034 |     0.0042095 |      0.0018847 | 7 🟠       | 1 🔴      |
|     5 | node_conv2d_1       | z            |          0.034 |             0 |              0 | 8 🟠       |           |
|     6 | node_linear         | y            |           0.03 |      0.042665 |       0.013567 | 1 🔴       | 3 🔴      |
|     6 | node_linear         | z            |           0.03 |             0 |              0 | 2 🔴       |           |
|     7 | node_linear_1_2     | y            |           0.03 |      0.042665 |       0.012763 | 5 🔴       | 4 🔴      |
|     7 | node_linear_1_2     | z            |           0.03 |             0 |              0 | 6 🟠       |           |

+ Layers performs best in improving accuracy (sorted by `Max Abs Error`):

"node_conv2d_1", "node_conv2d", "node_linear", "node_linear_1_2",
