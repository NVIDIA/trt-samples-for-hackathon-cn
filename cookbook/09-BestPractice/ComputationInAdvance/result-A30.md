# Before optimization

+ Structure of the network

```shell
[V] Engine Layer Information:
Layer(Constant): constantData, Tactic: 0,  -> (Unnamed Layer* 0) [Constant]_output[Float(1,512,256)]
Layer(Constant): 3215 + (Unnamed Layer* 19) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 19) [Shuffle]_output[Float(1,256,256)]
Layer(Constant): 3238 + (Unnamed Layer* 32) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 32) [Shuffle]_output[Float(1,256,256)]
Layer(Constant): 3261 + (Unnamed Layer* 45) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 45) [Shuffle]_output[Float(1,256,256)]
Layer(Constant): 3284 + (Unnamed Layer* 58) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 58) [Shuffle]_output[Float(1,256,256)]
Layer(Constant): 3307 + (Unnamed Layer* 71) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 71) [Shuffle]_output[Float(1,256,256)]
Layer(Constant): 3330 + (Unnamed Layer* 84) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 84) [Shuffle]_output[Float(1,256,256)]
Layer(Constant): 3353 + (Unnamed Layer* 97) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 97) [Shuffle]_output[Float(1,256,256)]
Layer(Constant): 3376 + (Unnamed Layer* 110) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 110) [Shuffle]_output[Float(1,256,256)]
Layer(Constant): 3399 + (Unnamed Layer* 123) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 123) [Shuffle]_output[Float(1,256,256)]
Layer(Constant): 3422 + (Unnamed Layer* 136) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 136) [Shuffle]_output[Float(1,256,256)]
Layer(Constant): 3445 + (Unnamed Layer* 149) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 149) [Shuffle]_output[Float(1,256,256)]
Layer(Constant): 3468 + (Unnamed Layer* 162) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 162) [Shuffle]_output[Float(1,256,256)]
Layer(Reformat): Slice_74, Tactic: 0, (Unnamed Layer* 0) [Constant]_output[Float(1,32,256)] -> 603[Float(1,32,256)]
Layer(MatrixMultiply): MatMul_141, Tactic: 1, 603[Float(1,32,256)], (Unnamed Layer* 19) [Shuffle]_output[Float(1,256,256)] -> 703[Float(1,32,256)]
Layer(MatrixMultiply): MatMul_298, Tactic: 1, 603[Float(1,32,256)], (Unnamed Layer* 32) [Shuffle]_output[Float(1,256,256)] -> 913[Float(1,32,256)]
Layer(MatrixMultiply): MatMul_455, Tactic: 1, 603[Float(1,32,256)], (Unnamed Layer* 45) [Shuffle]_output[Float(1,256,256)] -> 1123[Float(1,32,256)]
Layer(MatrixMultiply): MatMul_612, Tactic: 1, 603[Float(1,32,256)], (Unnamed Layer* 58) [Shuffle]_output[Float(1,256,256)] -> 1333[Float(1,32,256)]
Layer(MatrixMultiply): MatMul_769, Tactic: 1, 603[Float(1,32,256)], (Unnamed Layer* 71) [Shuffle]_output[Float(1,256,256)] -> 1543[Float(1,32,256)]
Layer(MatrixMultiply): MatMul_926, Tactic: 1, 603[Float(1,32,256)], (Unnamed Layer* 84) [Shuffle]_output[Float(1,256,256)] -> 1753[Float(1,32,256)]
Layer(MatrixMultiply): MatMul_1083, Tactic: 1, 603[Float(1,32,256)], (Unnamed Layer* 97) [Shuffle]_output[Float(1,256,256)] -> 1963[Float(1,32,256)]
Layer(MatrixMultiply): MatMul_1240, Tactic: 1, 603[Float(1,32,256)], (Unnamed Layer* 110) [Shuffle]_output[Float(1,256,256)] -> 2173[Float(1,32,256)]
Layer(MatrixMultiply): MatMul_1397, Tactic: 1, 603[Float(1,32,256)], (Unnamed Layer* 123) [Shuffle]_output[Float(1,256,256)] -> 2383[Float(1,32,256)]
Layer(MatrixMultiply): MatMul_1554, Tactic: 1, 603[Float(1,32,256)], (Unnamed Layer* 136) [Shuffle]_output[Float(1,256,256)] -> 2593[Float(1,32,256)]
Layer(MatrixMultiply): MatMul_1711, Tactic: 1, 603[Float(1,32,256)], (Unnamed Layer* 149) [Shuffle]_output[Float(1,256,256)] -> 2803[Float(1,32,256)]
Layer(MatrixMultiply): MatMul_1868, Tactic: 1, 603[Float(1,32,256)], (Unnamed Layer* 162) [Shuffle]_output[Float(1,256,256)] -> 3013[Float(1,32,256)]
Layer(Shuffle): Reshape_145 + Transpose_152, Tactic: 0, 703[Float(1,32,256)] -> 723[Float(1,4,64,32)]
Layer(Shuffle): Reshape_302 + Transpose_309, Tactic: 0, 913[Float(1,32,256)] -> 933[Float(1,4,64,32)]
Layer(Shuffle): Reshape_459 + Transpose_466, Tactic: 0, 1123[Float(1,32,256)] -> 1143[Float(1,4,64,32)]
Layer(Shuffle): Reshape_616 + Transpose_623, Tactic: 0, 1333[Float(1,32,256)] -> 1353[Float(1,4,64,32)]
Layer(Shuffle): Reshape_773 + Transpose_780, Tactic: 0, 1543[Float(1,32,256)] -> 1563[Float(1,4,64,32)]
Layer(Shuffle): Reshape_930 + Transpose_937, Tactic: 0, 1753[Float(1,32,256)] -> 1773[Float(1,4,64,32)]
Layer(Shuffle): Reshape_1087 + Transpose_1094, Tactic: 0, 1963[Float(1,32,256)] -> 1983[Float(1,4,64,32)]
Layer(Shuffle): Reshape_1244 + Transpose_1251, Tactic: 0, 2173[Float(1,32,256)] -> 2193[Float(1,4,64,32)]
Layer(Shuffle): Reshape_1401 + Transpose_1408, Tactic: 0, 2383[Float(1,32,256)] -> 2403[Float(1,4,64,32)]
Layer(Shuffle): Reshape_1558 + Transpose_1565, Tactic: 0, 2593[Float(1,32,256)] -> 2613[Float(1,4,64,32)]
Layer(Shuffle): Reshape_1715 + Transpose_1722, Tactic: 0, 2803[Float(1,32,256)] -> 2823[Float(1,4,64,32)]
Layer(Shuffle): Reshape_1872 + Transpose_1879, Tactic: 0, 3013[Float(1,32,256)] -> 3033[Float(1,4,64,32)]
```

+ Result of performance test

```shell
[I] === Performance summary ===
[I] Throughput: 8702.62 qps
[I] Latency: min = 0.106445 ms, max = 0.14566 ms, mean = 0.10741 ms, median = 0.106934 ms, percentile(99%) = 0.12204 ms
[I] End-to-End Host Latency: min = 0.106445 ms, max = 0.14566 ms, mean = 0.10741 ms, median = 0.106934 ms, percentile(99%) = 0.12204 ms
[I] Enqueue Time: min = 0.00830078 ms, max = 0.0664062 ms, mean = 0.0108104 ms, median = 0.0107422 ms, percentile(99%) = 0.012146 ms
[I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] GPU Compute Time: min = 0.106445 ms, max = 0.14566 ms, mean = 0.10741 ms, median = 0.106934 ms, percentile(99%) = 0.12204 ms
[I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] Total Host Walltime: 3.00025 s
[I] Total GPU Compute Time: 2.80447 s
```

# After optimization

+ Structure of the network

```shell
[V] Engine Layer Information:
Layer(Constant): wiliConstant-0, Tactic: 0,  -> (Unnamed Layer* 0) [Constant]_output[Float(1,4,64,512)]
Layer(Constant): wiliConstant-1, Tactic: 0,  -> (Unnamed Layer* 15) [Constant]_output[Float(1,4,64,512)]
Layer(Constant): wiliConstant-2, Tactic: 0,  -> (Unnamed Layer* 30) [Constant]_output[Float(1,4,64,512)]
Layer(Constant): wiliConstant-3, Tactic: 0,  -> (Unnamed Layer* 45) [Constant]_output[Float(1,4,64,512)]
Layer(Constant): wiliConstant-4, Tactic: 0,  -> (Unnamed Layer* 60) [Constant]_output[Float(1,4,64,512)]
Layer(Constant): wiliConstant-5, Tactic: 0,  -> (Unnamed Layer* 75) [Constant]_output[Float(1,4,64,512)]
Layer(Constant): wiliConstant-6, Tactic: 0,  -> (Unnamed Layer* 90) [Constant]_output[Float(1,4,64,512)]
Layer(Constant): wiliConstant-7, Tactic: 0,  -> (Unnamed Layer* 105) [Constant]_output[Float(1,4,64,512)]
Layer(Constant): wiliConstant-8, Tactic: 0,  -> (Unnamed Layer* 120) [Constant]_output[Float(1,4,64,512)]
Layer(Constant): wiliConstant-9, Tactic: 0,  -> (Unnamed Layer* 135) [Constant]_output[Float(1,4,64,512)]
Layer(Constant): wiliConstant-10, Tactic: 0,  -> (Unnamed Layer* 150) [Constant]_output[Float(1,4,64,512)]
Layer(Constant): wiliConstant-11, Tactic: 0,  -> (Unnamed Layer* 165) [Constant]_output[Float(1,4,64,512)]
Layer(Reformat): wiliSliceN-0, Tactic: 0, (Unnamed Layer* 0) [Constant]_output[Float(1,4,64,32)] -> 723[Float(1,4,64,32)]
Layer(Reformat): wiliSliceN-1, Tactic: 0, (Unnamed Layer* 15) [Constant]_output[Float(1,4,64,32)] -> 933[Float(1,4,64,32)]
Layer(Reformat): wiliSliceN-2, Tactic: 0, (Unnamed Layer* 30) [Constant]_output[Float(1,4,64,32)] -> 1143[Float(1,4,64,32)]
Layer(Reformat): wiliSliceN-3, Tactic: 0, (Unnamed Layer* 45) [Constant]_output[Float(1,4,64,32)] -> 1353[Float(1,4,64,32)]
Layer(Reformat): wiliSliceN-4, Tactic: 0, (Unnamed Layer* 60) [Constant]_output[Float(1,4,64,32)] -> 1563[Float(1,4,64,32)]
Layer(Reformat): wiliSliceN-5, Tactic: 0, (Unnamed Layer* 75) [Constant]_output[Float(1,4,64,32)] -> 1773[Float(1,4,64,32)]
Layer(Reformat): wiliSliceN-6, Tactic: 0, (Unnamed Layer* 90) [Constant]_output[Float(1,4,64,32)] -> 1983[Float(1,4,64,32)]
Layer(Reformat): wiliSliceN-7, Tactic: 0, (Unnamed Layer* 105) [Constant]_output[Float(1,4,64,32)] -> 2193[Float(1,4,64,32)]
Layer(Reformat): wiliSliceN-8, Tactic: 0, (Unnamed Layer* 120) [Constant]_output[Float(1,4,64,32)] -> 2403[Float(1,4,64,32)]
Layer(Reformat): wiliSliceN-9, Tactic: 0, (Unnamed Layer* 135) [Constant]_output[Float(1,4,64,32)] -> 2613[Float(1,4,64,32)]
Layer(Reformat): wiliSliceN-10, Tactic: 0, (Unnamed Layer* 150) [Constant]_output[Float(1,4,64,32)] -> 2823[Float(1,4,64,32)]
Layer(Reformat): wiliSliceN-11, Tactic: 0, (Unnamed Layer* 165) [Constant]_output[Float(1,4,64,32)] -> 3033[Float(1,4,64,32)]
```

+ Result of performance test

```shell
[I] === Performance summary ===
[I] Throughput: 28978.2 qps
[I] Latency: min = 0.0274658 ms, max = 0.0501099 ms, mean = 0.0280059 ms, median = 0.027832 ms, percentile(99%) = 0.0419464 ms
[I] End-to-End Host Latency: min = 0.0274658 ms, max = 0.0501099 ms, mean = 0.0280059 ms, median = 0.027832 ms, percentile(99%) = 0.0419464 ms
[I] Enqueue Time: min = 0.00219727 ms, max = 0.0253906 ms, mean = 0.00293333 ms, median = 0.00268555 ms, percentile(99%) = 0.00537109 ms
[I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] GPU Compute Time: min = 0.0274658 ms, max = 0.0501099 ms, mean = 0.0280059 ms, median = 0.027832 ms, percentile(99%) = 0.0419464 ms
[I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] Total Host Walltime: 3.00009 s
[I] Total GPU Compute Time: 2.43475 s
```
