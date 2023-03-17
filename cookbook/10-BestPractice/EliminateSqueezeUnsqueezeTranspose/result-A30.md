# Before optimization

+ Structure of the network

```shell
[V] Engine Layer Information:
Layer(CudnnConvolution): Conv0, Tactic: 0, tensor-0[Float(1,1,16,16)] -> tensor-1[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-0, Tactic: 7274495, tensor-1[Float(1,32,16,16)] -> tensor-0-1[Float(1,32,16,16)]
Layer(NoOp): Unsqueeze-%d0, Tactic: 0, tensor-0-1[Float(1,32,16,16)] -> tensor-0-2[Float(1,1,32,16,16)]
Layer(Scale): constant32 + (Unnamed Layer* 9) [Shuffle] + Add-0, Tactic: 0, tensor-0-2[Float(1,1,32,16,16)] -> tensor-0-3[Float(1,1,32,16,16)]
Layer(NoOp): Squeeze-%d0, Tactic: 0, tensor-0-3[Float(1,1,32,16,16)] -> tensor-0-4[Float(1,32,16,16)]
Layer(PointWiseV2): PWN(ReLU-0), Tactic: 2, tensor-0-4[Float(1,32,16,16)] -> tensor-0-5[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-1, Tactic: 7274495, tensor-0-5[Float(1,32,16,16)] -> tensor-1-1[Float(1,32,16,16)]
Layer(NoOp): Unsqueeze-%d1, Tactic: 0, tensor-1-1[Float(1,32,16,16)] -> tensor-1-2[Float(1,1,32,16,16)]
Layer(Scale): constant32_0 + (Unnamed Layer* 24) [Shuffle] + Add-1, Tactic: 0, tensor-1-2[Float(1,1,32,16,16)] -> tensor-1-3[Float(1,1,32,16,16)]
Layer(NoOp): Squeeze-%d1, Tactic: 0, tensor-1-3[Float(1,1,32,16,16)] -> tensor-1-4[Float(1,32,16,16)]
Layer(PointWiseV2): PWN(ReLU-1), Tactic: 2, tensor-1-4[Float(1,32,16,16)] -> tensor-1-5[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-2, Tactic: 7274495, tensor-1-5[Float(1,32,16,16)] -> tensor-2-1[Float(1,32,16,16)]
Layer(NoOp): Unsqueeze-%d2, Tactic: 0, tensor-2-1[Float(1,32,16,16)] -> tensor-2-2[Float(1,1,32,16,16)]
Layer(Scale): constant32_1 + (Unnamed Layer* 39) [Shuffle] + Add-2, Tactic: 0, tensor-2-2[Float(1,1,32,16,16)] -> tensor-2-3[Float(1,1,32,16,16)]
Layer(NoOp): Squeeze-%d2, Tactic: 0, tensor-2-3[Float(1,1,32,16,16)] -> tensor-2-4[Float(1,32,16,16)]
Layer(PointWiseV2): PWN(ReLU-2), Tactic: 2, tensor-2-4[Float(1,32,16,16)] -> tensor-2-5[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-3, Tactic: 7274495, tensor-2-5[Float(1,32,16,16)] -> tensor-3-1[Float(1,32,16,16)]
Layer(NoOp): Unsqueeze-%d3, Tactic: 0, tensor-3-1[Float(1,32,16,16)] -> tensor-3-2[Float(1,1,32,16,16)]
Layer(Scale): constant32_2 + (Unnamed Layer* 54) [Shuffle] + Add-3, Tactic: 0, tensor-3-2[Float(1,1,32,16,16)] -> tensor-3-3[Float(1,1,32,16,16)]
Layer(NoOp): Squeeze-%d3, Tactic: 0, tensor-3-3[Float(1,1,32,16,16)] -> tensor-3-4[Float(1,32,16,16)]
Layer(PointWiseV2): PWN(ReLU-3), Tactic: 2, tensor-3-4[Float(1,32,16,16)] -> tensor-3-5[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-4, Tactic: 7274495, tensor-3-5[Float(1,32,16,16)] -> tensor-4-1[Float(1,32,16,16)]
Layer(NoOp): Unsqueeze-%d4, Tactic: 0, tensor-4-1[Float(1,32,16,16)] -> tensor-4-2[Float(1,1,32,16,16)]
Layer(Scale): constant32_3 + (Unnamed Layer* 69) [Shuffle] + Add-4, Tactic: 0, tensor-4-2[Float(1,1,32,16,16)] -> tensor-4-3[Float(1,1,32,16,16)]
Layer(NoOp): Squeeze-%d4, Tactic: 0, tensor-4-3[Float(1,1,32,16,16)] -> tensor-4-4[Float(1,32,16,16)]
Layer(PointWiseV2): PWN(ReLU-4), Tactic: 2, tensor-4-4[Float(1,32,16,16)] -> tensor-4-5[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-5, Tactic: 7274495, tensor-4-5[Float(1,32,16,16)] -> tensor-5-1[Float(1,32,16,16)]
Layer(Shuffle): Transpose-%d5, Tactic: 0, tensor-5-1[Float(1,32,16,16)] -> tensor-5-2[Float(1,16,16,32)]
Layer(Constant): constant32t, Tactic: 0,  -> (Unnamed Layer* 78) [Constant]_output[Float(1,1,1,32)]
Layer(ElementWise): Add-5, Tactic: 1, tensor-5-2[Float(1,16,16,32)], (Unnamed Layer* 78) [Constant]_output[Float(1,1,1,32)] -> tensor-5-3[Float(1,16,16,32)]
Layer(Shuffle): Transpose-%d5_4, Tactic: 0, tensor-5-3[Float(1,16,16,32)] -> tensor-5-4[Float(1,32,16,16)]
Layer(PointWiseV2): PWN(ReLU-5), Tactic: 2, tensor-5-4[Float(1,32,16,16)] -> tensor-5-5[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-6, Tactic: 7274495, tensor-5-5[Float(1,32,16,16)] -> tensor-6-1[Float(1,32,16,16)]
Layer(Shuffle): Transpose-%d6, Tactic: 0, tensor-6-1[Float(1,32,16,16)] -> tensor-6-2[Float(1,16,16,32)]
Layer(Constant): constant32t_5, Tactic: 0,  -> (Unnamed Layer* 84) [Constant]_output[Float(1,1,1,32)]
Layer(ElementWise): Add-6, Tactic: 1, tensor-6-2[Float(1,16,16,32)], (Unnamed Layer* 84) [Constant]_output[Float(1,1,1,32)] -> tensor-6-3[Float(1,16,16,32)]
Layer(Shuffle): Transpose-%d6_6, Tactic: 0, tensor-6-3[Float(1,16,16,32)] -> tensor-6-4[Float(1,32,16,16)]
Layer(PointWiseV2): PWN(ReLU-6), Tactic: 2, tensor-6-4[Float(1,32,16,16)] -> tensor-6-5[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-7, Tactic: 7274495, tensor-6-5[Float(1,32,16,16)] -> tensor-7-1[Float(1,32,16,16)]
Layer(Shuffle): Transpose-%d7, Tactic: 0, tensor-7-1[Float(1,32,16,16)] -> tensor-7-2[Float(1,16,16,32)]
Layer(Constant): constant32t_7, Tactic: 0,  -> (Unnamed Layer* 90) [Constant]_output[Float(1,1,1,32)]
Layer(ElementWise): Add-7, Tactic: 1, tensor-7-2[Float(1,16,16,32)], (Unnamed Layer* 90) [Constant]_output[Float(1,1,1,32)] -> tensor-7-3[Float(1,16,16,32)]
Layer(Shuffle): Transpose-%d7_8, Tactic: 0, tensor-7-3[Float(1,16,16,32)] -> tensor-7-4[Float(1,32,16,16)]
Layer(PointWiseV2): PWN(ReLU-7), Tactic: 2, tensor-7-4[Float(1,32,16,16)] -> tensor-7-5[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-8, Tactic: 7274495, tensor-7-5[Float(1,32,16,16)] -> tensor-8-1[Float(1,32,16,16)]
Layer(Shuffle): Transpose-%d8, Tactic: 0, tensor-8-1[Float(1,32,16,16)] -> tensor-8-2[Float(1,16,16,32)]
Layer(Constant): constant32t_9, Tactic: 0,  -> (Unnamed Layer* 96) [Constant]_output[Float(1,1,1,32)]
Layer(ElementWise): Add-8, Tactic: 1, tensor-8-2[Float(1,16,16,32)], (Unnamed Layer* 96) [Constant]_output[Float(1,1,1,32)] -> tensor-8-3[Float(1,16,16,32)]
Layer(Shuffle): Transpose-%d8_10, Tactic: 0, tensor-8-3[Float(1,16,16,32)] -> tensor-8-4[Float(1,32,16,16)]
Layer(PointWiseV2): PWN(ReLU-8), Tactic: 2, tensor-8-4[Float(1,32,16,16)] -> tensor-8-5[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-9, Tactic: 7274495, tensor-8-5[Float(1,32,16,16)] -> tensor-9-1[Float(1,32,16,16)]
Layer(Shuffle): Transpose-%d9, Tactic: 0, tensor-9-1[Float(1,32,16,16)] -> tensor-9-2[Float(1,16,16,32)]
Layer(Constant): constant32t_11, Tactic: 0,  -> (Unnamed Layer* 102) [Constant]_output[Float(1,1,1,32)]
Layer(ElementWise): Add-9, Tactic: 1, tensor-9-2[Float(1,16,16,32)], (Unnamed Layer* 102) [Constant]_output[Float(1,1,1,32)] -> tensor-9-3[Float(1,16,16,32)]
Layer(Shuffle): Transpose-%d9_12, Tactic: 0, tensor-9-3[Float(1,16,16,32)] -> tensor-9-4[Float(1,32,16,16)]
Layer(PointWiseV2): PWN(ReLU-9), Tactic: 2, tensor-9-4[Float(1,32,16,16)] -> tensor-9-5[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-1, Tactic: 2621439, tensor-9-5[Float(1,32,16,16)] -> tensor-6[Float(1,1,14,14)]
```

+ Result of performance test

```shell
[I] === Performance summary ===
[I] Throughput: 4023.27 qps
[I] Latency: min = 0.233032 ms, max = 10.22 ms, mean = 0.244098 ms, median = 0.234619 ms, percentile(99%) = 0.361572 ms
[I] End-to-End Host Latency: min = 0.233032 ms, max = 10.22 ms, mean = 0.244098 ms, median = 0.234619 ms, percentile(99%) = 0.361572 ms
[I] Enqueue Time: min = 0.00830078 ms, max = 10.3574 ms, mean = 0.0231489 ms, median = 0.0217285 ms, percentile(99%) = 0.0292969 ms
[I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] GPU Compute Time: min = 0.233032 ms, max = 10.22 ms, mean = 0.244098 ms, median = 0.234619 ms, percentile(99%) = 0.361572 ms
[I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] Total Host Walltime: 2.0886 s
[I] Total GPU Compute Time: 2.05115 s
```

# After optimization

+ Structure of the network

```shell
[V] Engine Layer Information:
Layer(CudnnConvolution): Conv0, Tactic: 0, tensor-0[Float(1,1,16,16)] -> tensor-1[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-0 + ReLU-0, Tactic: 7274495, tensor-1[Float(1,32,16,16)] -> tensor-0-5[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-1 + ReLU-1, Tactic: 7274495, tensor-0-5[Float(1,32,16,16)] -> tensor-1-5[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-2 + ReLU-2, Tactic: 7274495, tensor-1-5[Float(1,32,16,16)] -> tensor-2-5[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-3 + ReLU-3, Tactic: 7274495, tensor-2-5[Float(1,32,16,16)] -> tensor-3-5[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-4 + ReLU-4, Tactic: 7274495, tensor-3-5[Float(1,32,16,16)] -> tensor-4-5[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-5 + ReLU-5, Tactic: 7274495, tensor-4-5[Float(1,32,16,16)] -> tensor-5-5[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-6 + ReLU-6, Tactic: 7274495, tensor-5-5[Float(1,32,16,16)] -> tensor-6-5[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-7 + ReLU-7, Tactic: 7274495, tensor-6-5[Float(1,32,16,16)] -> tensor-7-5[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-8 + ReLU-8, Tactic: 7274495, tensor-7-5[Float(1,32,16,16)] -> tensor-8-5[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-9 + ReLU-9, Tactic: 7274495, tensor-8-5[Float(1,32,16,16)] -> tensor-9-5[Float(1,32,16,16)]
Layer(FusedConvActConvolution): Conv-1, Tactic: 2621439, tensor-9-5[Float(1,32,16,16)] -> tensor-6[Float(1,1,14,14)]
```

+ Result of performance test

```shell
[I] === Performance summary ===
[I] Throughput: 8827.49 qps
[I] Latency: min = 0.105469 ms, max = 0.185181 ms, mean = 0.109195 ms, median = 0.106201 ms, percentile(99%) = 0.162842 ms
[I] End-to-End Host Latency: min = 0.105469 ms, max = 0.185181 ms, mean = 0.109195 ms, median = 0.106201 ms, percentile(99%) = 0.162842 ms
[I] Enqueue Time: min = 0.00292969 ms, max = 0.0427246 ms, mean = 0.00793225 ms, median = 0.0078125 ms, percentile(99%) = 0.00976562 ms
[I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] GPU Compute Time: min = 0.105469 ms, max = 0.185181 ms, mean = 0.109195 ms, median = 0.106201 ms, percentile(99%) = 0.162842 ms
[I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] Total Host Walltime: 2.07182 s
[I] Total GPU Compute Time: 1.99707 s
```
