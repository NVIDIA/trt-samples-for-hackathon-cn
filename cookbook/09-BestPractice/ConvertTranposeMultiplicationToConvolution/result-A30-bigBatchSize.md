# Transpose + Matirx Mulltiplication
+ Structure of the network
```
[05/09/2022-06:27:19] [TRT] [V] Engine Layer Information:
Layer(Shuffle): Transpose_51 + Reshape_60, Tactic: 0, inputTensor[Float(-8,256,-10,19)] -> 582[Float(-8,-10,4864)]
Layer(Constant): 3198 + (Unnamed Layer* 13) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 13) [Shuffle]_output[Float(1,4864,256)]
Layer(MatrixMultiply): MatMul_61, Tactic: 1, 582[Float(-8,-10,4864)], (Unnamed Layer* 13) [Shuffle]_output[Float(1,4864,256)] -> 584[Float(-8,-10,256)]
Layer(Constant): encoder.embed.out.0.bias + (Unnamed Layer* 16) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 16) [Shuffle]_output[Float(1,1,256)]
Layer(ElementWise): Add_62, Tactic: 1, (Unnamed Layer* 16) [Shuffle]_output[Float(1,1,256)], 584[Float(-8,-10,256)] -> 585[Float(-8,-10,256)]
```

+ trtexec Result of performance test
```
[05/09/2022-06:27:25] [I] === Performance summary ===
[05/09/2022-06:27:25] [I] Throughput: 476.66 qps
[05/09/2022-06:27:25] [I] Latency: min = 2.05615 ms, max = 2.59482 ms, mean = 2.09467 ms, median = 2.08276 ms, percentile(99%) = 2.58971 ms
[05/09/2022-06:27:25] [I] End-to-End Host Latency: min = 2.05615 ms, max = 2.59482 ms, mean = 2.09467 ms, median = 2.08276 ms, percentile(99%) = 2.58971 ms
[05/09/2022-06:27:25] [I] Enqueue Time: min = 0.00244141 ms, max = 0.0256348 ms, mean = 0.00720864 ms, median = 0.00634766 ms, percentile(99%) = 0.0192261 ms
[05/09/2022-06:27:25] [I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[05/09/2022-06:27:25] [I] GPU Compute Time: min = 2.05615 ms, max = 2.59482 ms, mean = 2.09467 ms, median = 2.08276 ms, percentile(99%) = 2.58971 ms
[05/09/2022-06:27:25] [I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[05/09/2022-06:27:25] [I] Total Host Walltime: 3.00424 s
[05/09/2022-06:27:25] [I] Total GPU Compute Time: 2.99956 s
```

# Convlolution + Shuffle
+ Structure of the network
```
[05/09/2022-06:27:28] [TRT] [V] Engine Layer Information:
Layer(Reformat): Reformatting CopyNode for Input Tensor 0 to ConvN, Tactic: 1002, inputTensor[Float(-8,256,-10,19)] -> Reformatted Input Tensor 0 to ConvN[Float(-8,256,-10,19)]
Layer(CaskConvolution): ConvN, Tactic: 5200329514761435342, Reformatted Input Tensor 0 to ConvN[Float(-8,256,-10,19)] -> ConvV[Float(-8,256,-10,1)]
Layer(NoOp): Reformatting CopyNode for Input Tensor 0 to SqueezeN + TransposeN, Tactic: 0, ConvV[Float(-8,256,-10,1)] -> Reformatted Input Tensor 0 to SqueezeN + TransposeN[Float(-8,256,-10,1)]
Layer(Shuffle): SqueezeN + TransposeN, Tactic: 1, Reformatted Input Tensor 0 to SqueezeN + TransposeN[Float(-8,256,-10,1)] -> Reformatted Output Tensor 0 to SqueezeN + TransposeN[Float(-8,-10,256)]
Layer(Reformat): Reformatting CopyNode for Output Tensor 0 to SqueezeN + TransposeN, Tactic: 1002, Reformatted Output Tensor 0 to SqueezeN + TransposeN[Float(-8,-10,256)] -> TransposeV[Float(-8,-10,256)]
```

+ trtexec Result of performance test
```
[05/09/2022-06:27:34] [I] === Performance summary ===
[05/09/2022-06:27:34] [I] Throughput: 544.953 qps
[05/09/2022-06:27:34] [I] Latency: min = 1.80737 ms, max = 1.86877 ms, mean = 1.83174 ms, median = 1.83105 ms, percentile(99%) = 1.86572 ms
[05/09/2022-06:27:34] [I] End-to-End Host Latency: min = 1.80737 ms, max = 1.86877 ms, mean = 1.83174 ms, median = 1.83105 ms, percentile(99%) = 1.86572 ms
[05/09/2022-06:27:34] [I] Enqueue Time: min = 0.00585938 ms, max = 0.0303955 ms, mean = 0.00792179 ms, median = 0.00732422 ms, percentile(99%) = 0.0194092 ms
[05/09/2022-06:27:34] [I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[05/09/2022-06:27:34] [I] GPU Compute Time: min = 1.80737 ms, max = 1.86877 ms, mean = 1.83174 ms, median = 1.83105 ms, percentile(99%) = 1.86572 ms
[05/09/2022-06:27:34] [I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[05/09/2022-06:27:34] [I] Total Host Walltime: 3.00393 s
[05/09/2022-06:27:34] [I] Total GPU Compute Time: 2.99856 s
```
