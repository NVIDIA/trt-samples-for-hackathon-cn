# Transpose + Matirx Mulltiplication
+ Structure of the network
```
[V] Engine Layer Information:
Layer(Shuffle): Transpose_51 + Reshape_60, Tactic: 0, inputTensor[Float(-8,256,-10,19)] -> 582[Float(-8,-10,4864)]
Layer(Constant): 3198 + (Unnamed Layer* 13) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 13) [Shuffle]_output[Float(1,4864,256)]
Layer(MatrixMultiply): MatMul_61, Tactic: 1, 582[Float(-8,-10,4864)], (Unnamed Layer* 13) [Shuffle]_output[Float(1,4864,256)] -> 584[Float(-8,-10,256)]
Layer(Constant): encoder.embed.out.0.bias + (Unnamed Layer* 16) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 16) [Shuffle]_output[Float(1,1,256)]
Layer(ElementWise): Add_62, Tactic: 1, (Unnamed Layer* 16) [Shuffle]_output[Float(1,1,256)], 584[Float(-8,-10,256)] -> 585[Float(-8,-10,256)]
```

+ trtexec Result of performance test
```
[I] === Performance summary ===
[I] Throughput: 6555.57 qps
[I] Latency: min = 0.140259 ms, max = 8.92926 ms, mean = 0.147262 ms, median = 0.146484 ms, percentile(99%) = 0.15155 ms
[I] End-to-End Host Latency: min = 0.140259 ms, max = 8.92926 ms, mean = 0.147262 ms, median = 0.146484 ms, percentile(99%) = 0.15155 ms
[I] Enqueue Time: min = 0.00427246 ms, max = 0.11853 ms, mean = 0.0111693 ms, median = 0.0100098 ms, percentile(99%) = 0.0263062 ms
[I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] GPU Compute Time: min = 0.140259 ms, max = 8.92926 ms, mean = 0.147262 ms, median = 0.146484 ms, percentile(99%) = 0.15155 ms
[I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] Total Host Walltime: 3.0002 s
[I] Total GPU Compute Time: 2.89635 s
```

# Convlolution + Shuffle
+ Structure of the network
```
[V] Engine Layer Information:
Layer(Reformat): Reformatting CopyNode for Input Tensor 0 to ConvN, Tactic: 1002, inputTensor[Float(-8,256,-10,19)] -> Reformatted Input Tensor 0 to ConvN[Float(-8,256,-10,19)]
Layer(CaskConvolution): ConvN, Tactic: 7347365539922924600, Reformatted Input Tensor 0 to ConvN[Float(-8,256,-10,19)] -> ConvV[Float(-8,256,-10,1)]
Layer(Reformat): Reformatting CopyNode for Input Tensor 0 to SqueezeN + TransposeN, Tactic: 0, ConvV[Float(-8,256,-10,1)] -> Reformatted Input Tensor 0 to SqueezeN + TransposeN[Float(-8,256,-10,1)]
Layer(Shuffle): SqueezeN + TransposeN, Tactic: 0, Reformatted Input Tensor 0 to SqueezeN + TransposeN[Float(-8,256,-10,1)] -> TransposeV[Float(-8,-10,256)]
```

+ trtexec Result of performance test
```
[I] === Performance summary ===
[I] Throughput: 6117.88 qps
[I] Latency: min = 0.154541 ms, max = 0.236542 ms, mean = 0.158555 ms, median = 0.157715 ms, percentile(99%) = 0.164917 ms
[I] End-to-End Host Latency: min = 0.154541 ms, max = 0.236542 ms, mean = 0.158555 ms, median = 0.157715 ms, percentile(99%) = 0.164917 ms
[I] Enqueue Time: min = 0.00244141 ms, max = 0.0817871 ms, mean = 0.00727572 ms, median = 0.00646973 ms, percentile(99%) = 0.0192261 ms
[I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] GPU Compute Time: min = 0.154541 ms, max = 0.236542 ms, mean = 0.158555 ms, median = 0.157715 ms, percentile(99%) = 0.164917 ms
[I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] Total Host Walltime: 3.00022 s
[I] Total GPU Compute Time: 2.91027 s
```
