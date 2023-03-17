# Transpose + Matirx Mulltiplication
+ Structure of the network
```
[05/09/2022-06:10:32] [TRT] [V] Engine Layer Information:
Layer(Shuffle): Transpose_51 + Reshape_60, Tactic: 0, inputTensor[Float(-8,256,-10,19)] -> 582[Float(-8,-10,4864)]
Layer(Constant): 3198 + (Unnamed Layer* 13) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 13) [Shuffle]_output[Float(1,4864,256)]
Layer(MatrixMultiply): MatMul_61, Tactic: 0, 582[Float(-8,-10,4864)], (Unnamed Layer* 13) [Shuffle]_output[Float(1,4864,256)] -> 584[Float(-8,-10,256)]
Layer(Constant): encoder.embed.out.0.bias + (Unnamed Layer* 16) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 16) [Shuffle]_output[Float(1,1,256)]
Layer(ElementWise): Add_62, Tactic: 1, (Unnamed Layer* 16) [Shuffle]_output[Float(1,1,256)], 584[Float(-8,-10,256)] -> 585[Float(-8,-10,256)]
```

+ trtexec Result of performance test
```
[05/09/2022-06:10:36] [I] === Performance summary ===
[05/09/2022-06:10:36] [I] Throughput: 797.639 qps
[05/09/2022-06:10:36] [I] Latency: min = 0.848877 ms, max = 12.2706 ms, mean = 0.942851 ms, median = 0.899048 ms, percentile(99%) = 1.80432 ms
[05/09/2022-06:10:36] [I] End-to-End Host Latency: min = 0.848877 ms, max = 12.2706 ms, mean = 0.942851 ms, median = 0.899048 ms, percentile(99%) = 1.80432 ms
[05/09/2022-06:10:36] [I] Enqueue Time: min = 0.0012207 ms, max = 0.104736 ms, mean = 0.0169319 ms, median = 0.017334 ms, percentile(99%) = 0.0490723 ms
[05/09/2022-06:10:36] [I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[05/09/2022-06:10:36] [I] GPU Compute Time: min = 0.848877 ms, max = 12.2706 ms, mean = 0.942851 ms, median = 0.899048 ms, percentile(99%) = 1.80432 ms
[05/09/2022-06:10:36] [I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[05/09/2022-06:10:36] [I] Total Host Walltime: 3.00387 s
[05/09/2022-06:10:36] [I] Total GPU Compute Time: 2.25907 s
```

# Convlolution + Shuffle
+ Structure of the network
```
[05/09/2022-06:10:37] [TRT] [V] Engine Layer Information:
Layer(CaskConvolution): ConvN, Tactic: -37215280111360163, inputTensor[Float(-6,256,-8,19)] -> ConvV[Float(-6,256,-8,1)]
Layer(Shuffle): SqueezeN + TransposeN, Tactic: 0, ConvV[Float(-6,256,-8,1)] -> TransposeV[Float(-6,-8,256)]
```

+ trtexec Result of performance test
``` 
[05/09/2022-06:10:41] [I] === Performance summary ===
[05/09/2022-06:10:41] [I] Throughput: 1253.1 qps
[05/09/2022-06:10:41] [I] Latency: min = 0.532471 ms, max = 5.46106 ms, mean = 0.594622 ms, median = 0.572998 ms, percentile(99%) = 1.25146 ms
[05/09/2022-06:10:41] [I] End-to-End Host Latency: min = 0.532471 ms, max = 5.46106 ms, mean = 0.594622 ms, median = 0.572998 ms, percentile(99%) = 1.25146 ms
[05/09/2022-06:10:41] [I] Enqueue Time: min = 0.000976562 ms, max = 0.117676 ms, mean = 0.0129857 ms, median = 0.0136719 ms, percentile(99%) = 0.0279541 ms
[05/09/2022-06:10:41] [I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[05/09/2022-06:10:41] [I] GPU Compute Time: min = 0.532471 ms, max = 5.46106 ms, mean = 0.594622 ms, median = 0.572998 ms, percentile(99%) = 1.25146 ms
[05/09/2022-06:10:41] [I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[05/09/2022-06:10:41] [I] Total Host Walltime: 3.00134 s
[05/09/2022-06:10:41] [I] Total GPU Compute Time: 2.23637 s
```
