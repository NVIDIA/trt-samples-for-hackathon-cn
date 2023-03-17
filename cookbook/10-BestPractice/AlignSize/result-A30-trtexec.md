# [M=32, K=256, N=2048]
+ Structure of the network
```
[V] Engine Layer Information:
Layer(NoOp): (Unnamed Layer* 1) [Shuffle], Tactic: 0, tensor0[Float(32,1)] -> (Unnamed Layer* 1) [Shuffle]_output[Float(32,1,1,1)]
Layer(Reformat): Reformatting CopyNode for Input Tensor 0 to MMU1, Tactic: 0, (Unnamed Layer* 1) [Shuffle]_output[Float(32,1,1,1)] -> Reformatted Input Tensor 0 to MMU1[Half(32,1,1,1)]
Layer(CublasConvolution): MMU1, Tactic: 0, Reformatted Input Tensor 0 to MMU1[Half(32,1,1,1)] -> (Unnamed Layer* 2) [Fully Connected]_output[Half(32,256,1,1)]
Layer(NoOp): Reformatting CopyNode for Input Tensor 0 to MMU-0 + constantN + (Unnamed Layer* 9) [Shuffle] + unsqueeze_node_after_constantN + (Unnamed Layer* 9) [Shuffle]_(Unnamed Layer* 9) [Shuffle]_output + AddU-0 + ReLUU-0, Tactic: 0, (Unnamed Layer* 2) [Fully Connected]_output[Half(32,256,1,1)] -> Reformatted Input Tensor 0 to MMU-0 + constantN + (Unnamed Layer* 9) [Shuffle] + unsqueeze_node_after_constantN + (Unnamed Layer* 9) [Shuffle]_(Unnamed Layer* 9) [Shuffle]_output + AddU-0 + ReLUU-0[Half(32,256,1,1)]
Layer(CaskConvolution): MMU-0 + constantN + (Unnamed Layer* 9) [Shuffle] + unsqueeze_node_after_constantN + (Unnamed Layer* 9) [Shuffle]_(Unnamed Layer* 9) [Shuffle]_output + AddU-0 + ReLUU-0, Tactic: 3899310434460415852, Reformatted Input Tensor 0 to MMU-0 + constantN + (Unnamed Layer* 9) [Shuffle] + unsqueeze_node_after_constantN + (Unnamed Layer* 9) [Shuffle]_(Unnamed Layer* 9) [Shuffle]_output + AddU-0 + ReLUU-0[Half(32,256,1,1)] -> ReLUU-0_out_tensor[Half(32,2048,1,1)]
Layer(CaskConvolution): MMD-0 + constantK + (Unnamed Layer* 17) [Shuffle] + unsqueeze_node_after_constantK + (Unnamed Layer* 17) [Shuffle]_(Unnamed Layer* 17) [Shuffle]_output + AddD-0 + ReLUD-0, Tactic: 3071479995783211391, ReLUU-0_out_tensor[Half(32,2048,1,1)] -> ReLUD-0_out_tensor[Half(32,256,1,1)]
Layer(CaskConvolution): MMU-1 + constantN_1 + (Unnamed Layer* 25) [Shuffle] + unsqueeze_node_after_constantN_1 + (Unnamed Layer* 25) [Shuffle]_(Unnamed Layer* 25) [Shuffle]_output + AddU-1 + ReLUU-1, Tactic: 3899310434460415852, ReLUD-0_out_tensor[Half(32,256,1,1)] -> ReLUU-1_out_tensor[Half(32,2048,1,1)]
Layer(CaskConvolution): MMD-1 + constantK_3 + (Unnamed Layer* 33) [Shuffle] + unsqueeze_node_after_constantK_3 + (Unnamed Layer* 33) [Shuffle]_(Unnamed Layer* 33) [Shuffle]_output + AddD-1 + ReLUD-1, Tactic: 3071479995783211391, ReLUU-1_out_tensor[Half(32,2048,1,1)] -> ReLUD-1_out_tensor[Half(32,256,1,1)]
Layer(CaskConvolution): MMU-2 + constantN_5 + (Unnamed Layer* 41) [Shuffle] + unsqueeze_node_after_constantN_5 + (Unnamed Layer* 41) [Shuffle]_(Unnamed Layer* 41) [Shuffle]_output + AddU-2 + ReLUU-2, Tactic: 3899310434460415852, ReLUD-1_out_tensor[Half(32,256,1,1)] -> ReLUU-2_out_tensor[Half(32,2048,1,1)]
Layer(CaskConvolution): MMD-2 + constantK_7 + (Unnamed Layer* 49) [Shuffle] + unsqueeze_node_after_constantK_7 + (Unnamed Layer* 49) [Shuffle]_(Unnamed Layer* 49) [Shuffle]_output + AddD-2 + ReLUD-2, Tactic: 3071479995783211391, ReLUU-2_out_tensor[Half(32,2048,1,1)] -> ReLUD-2_out_tensor[Half(32,256,1,1)]
Layer(CaskConvolution): MMU-3 + constantN_9 + (Unnamed Layer* 57) [Shuffle] + unsqueeze_node_after_constantN_9 + (Unnamed Layer* 57) [Shuffle]_(Unnamed Layer* 57) [Shuffle]_output + AddU-3 + ReLUU-3, Tactic: 3899310434460415852, ReLUD-2_out_tensor[Half(32,256,1,1)] -> ReLUU-3_out_tensor[Half(32,2048,1,1)]
Layer(CaskConvolution): MMD-3 + constantK_11 + (Unnamed Layer* 65) [Shuffle] + unsqueeze_node_after_constantK_11 + (Unnamed Layer* 65) [Shuffle]_(Unnamed Layer* 65) [Shuffle]_output + AddD-3 + ReLUD-3, Tactic: 3071479995783211391, ReLUU-3_out_tensor[Half(32,2048,1,1)] -> ReLUD-3_out_tensor[Half(32,256,1,1)]
Layer(CaskConvolution): MMU-4 + constantN_13 + (Unnamed Layer* 73) [Shuffle] + unsqueeze_node_after_constantN_13 + (Unnamed Layer* 73) [Shuffle]_(Unnamed Layer* 73) [Shuffle]_output + AddU-4 + ReLUU-4, Tactic: 3899310434460415852, ReLUD-3_out_tensor[Half(32,256,1,1)] -> ReLUU-4_out_tensor[Half(32,2048,1,1)]
Layer(CaskConvolution): MMD-4 + constantK_15 + (Unnamed Layer* 81) [Shuffle] + unsqueeze_node_after_constantK_15 + (Unnamed Layer* 81) [Shuffle]_(Unnamed Layer* 81) [Shuffle]_output + AddD-4 + ReLUD-4, Tactic: 3071479995783211391, ReLUU-4_out_tensor[Half(32,2048,1,1)] -> ReLUD-4_out_tensor[Half(32,256,1,1)]
Layer(CaskConvolution): MMU-5 + constantN_17 + (Unnamed Layer* 89) [Shuffle] + unsqueeze_node_after_constantN_17 + (Unnamed Layer* 89) [Shuffle]_(Unnamed Layer* 89) [Shuffle]_output + AddU-5 + ReLUU-5, Tactic: 3899310434460415852, ReLUD-4_out_tensor[Half(32,256,1,1)] -> ReLUU-5_out_tensor[Half(32,2048,1,1)]
Layer(CaskConvolution): MMD-5 + constantK_19 + (Unnamed Layer* 97) [Shuffle] + unsqueeze_node_after_constantK_19 + (Unnamed Layer* 97) [Shuffle]_(Unnamed Layer* 97) [Shuffle]_output + AddD-5 + ReLUD-5, Tactic: 3071479995783211391, ReLUU-5_out_tensor[Half(32,2048,1,1)] -> ReLUD-5_out_tensor[Half(32,256,1,1)]
Layer(CaskConvolution): MMU-6 + constantN_21 + (Unnamed Layer* 105) [Shuffle] + unsqueeze_node_after_constantN_21 + (Unnamed Layer* 105) [Shuffle]_(Unnamed Layer* 105) [Shuffle]_output + AddU-6 + ReLUU-6, Tactic: 3899310434460415852, ReLUD-5_out_tensor[Half(32,256,1,1)] -> ReLUU-6_out_tensor[Half(32,2048,1,1)]
Layer(CaskConvolution): MMD-6 + constantK_23 + (Unnamed Layer* 113) [Shuffle] + unsqueeze_node_after_constantK_23 + (Unnamed Layer* 113) [Shuffle]_(Unnamed Layer* 113) [Shuffle]_output + AddD-6 + ReLUD-6, Tactic: 3071479995783211391, ReLUU-6_out_tensor[Half(32,2048,1,1)] -> ReLUD-6_out_tensor[Half(32,256,1,1)]
Layer(CaskConvolution): MMU-7 + constantN_25 + (Unnamed Layer* 121) [Shuffle] + unsqueeze_node_after_constantN_25 + (Unnamed Layer* 121) [Shuffle]_(Unnamed Layer* 121) [Shuffle]_output + AddU-7 + ReLUU-7, Tactic: 3899310434460415852, ReLUD-6_out_tensor[Half(32,256,1,1)] -> ReLUU-7_out_tensor[Half(32,2048,1,1)]
Layer(CaskConvolution): MMD-7 + constantK_27 + (Unnamed Layer* 129) [Shuffle] + unsqueeze_node_after_constantK_27 + (Unnamed Layer* 129) [Shuffle]_(Unnamed Layer* 129) [Shuffle]_output + AddD-7 + ReLUD-7, Tactic: 3071479995783211391, ReLUU-7_out_tensor[Half(32,2048,1,1)] -> ReLUD-7_out_tensor[Half(32,256,1,1)]
Layer(CaskConvolution): MMU-8 + constantN_29 + (Unnamed Layer* 137) [Shuffle] + unsqueeze_node_after_constantN_29 + (Unnamed Layer* 137) [Shuffle]_(Unnamed Layer* 137) [Shuffle]_output + AddU-8 + ReLUU-8, Tactic: 3899310434460415852, ReLUD-7_out_tensor[Half(32,256,1,1)] -> ReLUU-8_out_tensor[Half(32,2048,1,1)]
Layer(CaskConvolution): MMD-8 + constantK_31 + (Unnamed Layer* 145) [Shuffle] + unsqueeze_node_after_constantK_31 + (Unnamed Layer* 145) [Shuffle]_(Unnamed Layer* 145) [Shuffle]_output + AddD-8 + ReLUD-8, Tactic: 3071479995783211391, ReLUU-8_out_tensor[Half(32,2048,1,1)] -> ReLUD-8_out_tensor[Half(32,256,1,1)]
Layer(CaskConvolution): MMU-9 + constantN_33 + (Unnamed Layer* 153) [Shuffle] + unsqueeze_node_after_constantN_33 + (Unnamed Layer* 153) [Shuffle]_(Unnamed Layer* 153) [Shuffle]_output + AddU-9 + ReLUU-9, Tactic: 3899310434460415852, ReLUD-8_out_tensor[Half(32,256,1,1)] -> ReLUU-9_out_tensor[Half(32,2048,1,1)]
Layer(CaskConvolution): MMD-9 + constantK_35 + (Unnamed Layer* 161) [Shuffle] + unsqueeze_node_after_constantK_35 + (Unnamed Layer* 161) [Shuffle]_(Unnamed Layer* 161) [Shuffle]_output + AddD-9 + ReLUD-9, Tactic: 3071479995783211391, ReLUU-9_out_tensor[Half(32,2048,1,1)] -> ReLUD-9_out_tensor[Half(32,256,1,1)]
Layer(Reformat): Reformatting CopyNode for Input Tensor 0 to squeeze_after_ReLUD-9, Tactic: 0, ReLUD-9_out_tensor[Half(32,256,1,1)] -> Reformatted Input Tensor 0 to squeeze_after_ReLUD-9[Float(32,256,1,1)]
Layer(NoOp): squeeze_after_ReLUD-9, Tactic: 0, Reformatted Input Tensor 0 to squeeze_after_ReLUD-9[Float(32,256,1,1)] -> squeeze_after_ReLUD-9_out_tensor[Float(32,256)]
Layer(Reduce): Reduce, Tactic: 1, squeeze_after_ReLUD-9_out_tensor[Float(32,256)] -> tensor8[Float(32)]
```

+ Result of performance test
```
[I] === Performance summary ===
[I] Throughput: 5947.16 qps
[I] Latency: min = 0.161621 ms, max = 0.233398 ms, mean = 0.164381 ms, median = 0.163818 ms, percentile(99%) = 0.202759 ms
[I] End-to-End Host Latency: min = 0.161621 ms, max = 0.233398 ms, mean = 0.164381 ms, median = 0.163818 ms, percentile(99%) = 0.202759 ms
[I] Enqueue Time: min = 0.00854492 ms, max = 0.104736 ms, mean = 0.0107556 ms, median = 0.0106201 ms, percentile(99%) = 0.012207 ms
[I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] GPU Compute Time: min = 0.161621 ms, max = 0.233398 ms, mean = 0.164381 ms, median = 0.163818 ms, percentile(99%) = 0.202759 ms
[I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] Total Host Walltime: 3.00025 s
[I] Total GPU Compute Time: 2.93304 s
```

# [M=31, K=256, N=2048]
+ Structure of the network
```
[V] Engine Layer Information:
Layer(NoOp): (Unnamed Layer* 1) [Shuffle], Tactic: 0, tensor0[Float(31,1)] -> (Unnamed Layer* 1) [Shuffle]_output[Float(31,1,1,1)]
Layer(Reformat): Reformatting CopyNode for Input Tensor 0 to MMU1, Tactic: 0, (Unnamed Layer* 1) [Shuffle]_output[Float(31,1,1,1)] -> Reformatted Input Tensor 0 to MMU1[Half(31,1,1,1)]
Layer(CublasConvolution): MMU1, Tactic: 0, Reformatted Input Tensor 0 to MMU1[Half(31,1,1,1)] -> (Unnamed Layer* 2) [Fully Connected]_output[Half(31,256,1,1)]
Layer(NoOp): Reformatting CopyNode for Input Tensor 0 to MMU-0 + constantN + (Unnamed Layer* 9) [Shuffle] + unsqueeze_node_after_constantN + (Unnamed Layer* 9) [Shuffle]_(Unnamed Layer* 9) [Shuffle]_output + AddU-0 + ReLUU-0, Tactic: 0, (Unnamed Layer* 2) [Fully Connected]_output[Half(31,256,1,1)] -> Reformatted Input Tensor 0 to MMU-0 + constantN + (Unnamed Layer* 9) [Shuffle] + unsqueeze_node_after_constantN + (Unnamed Layer* 9) [Shuffle]_(Unnamed Layer* 9) [Shuffle]_output + AddU-0 + ReLUU-0[Half(31,256,1,1)]
Layer(CaskConvolution): MMU-0 + constantN + (Unnamed Layer* 9) [Shuffle] + unsqueeze_node_after_constantN + (Unnamed Layer* 9) [Shuffle]_(Unnamed Layer* 9) [Shuffle]_output + AddU-0 + ReLUU-0, Tactic: 3899310434460415852, Reformatted Input Tensor 0 to MMU-0 + constantN + (Unnamed Layer* 9) [Shuffle] + unsqueeze_node_after_constantN + (Unnamed Layer* 9) [Shuffle]_(Unnamed Layer* 9) [Shuffle]_output + AddU-0 + ReLUU-0[Half(31,256,1,1)] -> ReLUU-0_out_tensor[Half(31,2048,1,1)]
Layer(CaskConvolution): MMD-0 + constantK + (Unnamed Layer* 17) [Shuffle] + unsqueeze_node_after_constantK + (Unnamed Layer* 17) [Shuffle]_(Unnamed Layer* 17) [Shuffle]_output + AddD-0 + ReLUD-0, Tactic: -1828674067637788567, ReLUU-0_out_tensor[Half(31,2048,1,1)] -> ReLUD-0_out_tensor[Half(31,256,1,1)]
Layer(CaskConvolution): MMU-1 + constantN_1 + (Unnamed Layer* 25) [Shuffle] + unsqueeze_node_after_constantN_1 + (Unnamed Layer* 25) [Shuffle]_(Unnamed Layer* 25) [Shuffle]_output + AddU-1 + ReLUU-1, Tactic: 3899310434460415852, ReLUD-0_out_tensor[Half(31,256,1,1)] -> ReLUU-1_out_tensor[Half(31,2048,1,1)]
Layer(CaskConvolution): MMD-1 + constantK_3 + (Unnamed Layer* 33) [Shuffle] + unsqueeze_node_after_constantK_3 + (Unnamed Layer* 33) [Shuffle]_(Unnamed Layer* 33) [Shuffle]_output + AddD-1 + ReLUD-1, Tactic: -1828674067637788567, ReLUU-1_out_tensor[Half(31,2048,1,1)] -> ReLUD-1_out_tensor[Half(31,256,1,1)]
Layer(CaskConvolution): MMU-2 + constantN_5 + (Unnamed Layer* 41) [Shuffle] + unsqueeze_node_after_constantN_5 + (Unnamed Layer* 41) [Shuffle]_(Unnamed Layer* 41) [Shuffle]_output + AddU-2 + ReLUU-2, Tactic: 3899310434460415852, ReLUD-1_out_tensor[Half(31,256,1,1)] -> ReLUU-2_out_tensor[Half(31,2048,1,1)]
Layer(CaskConvolution): MMD-2 + constantK_7 + (Unnamed Layer* 49) [Shuffle] + unsqueeze_node_after_constantK_7 + (Unnamed Layer* 49) [Shuffle]_(Unnamed Layer* 49) [Shuffle]_output + AddD-2 + ReLUD-2, Tactic: -1828674067637788567, ReLUU-2_out_tensor[Half(31,2048,1,1)] -> ReLUD-2_out_tensor[Half(31,256,1,1)]
Layer(CaskConvolution): MMU-3 + constantN_9 + (Unnamed Layer* 57) [Shuffle] + unsqueeze_node_after_constantN_9 + (Unnamed Layer* 57) [Shuffle]_(Unnamed Layer* 57) [Shuffle]_output + AddU-3 + ReLUU-3, Tactic: 3899310434460415852, ReLUD-2_out_tensor[Half(31,256,1,1)] -> ReLUU-3_out_tensor[Half(31,2048,1,1)]
Layer(CaskConvolution): MMD-3 + constantK_11 + (Unnamed Layer* 65) [Shuffle] + unsqueeze_node_after_constantK_11 + (Unnamed Layer* 65) [Shuffle]_(Unnamed Layer* 65) [Shuffle]_output + AddD-3 + ReLUD-3, Tactic: -1828674067637788567, ReLUU-3_out_tensor[Half(31,2048,1,1)] -> ReLUD-3_out_tensor[Half(31,256,1,1)]
Layer(CaskConvolution): MMU-4 + constantN_13 + (Unnamed Layer* 73) [Shuffle] + unsqueeze_node_after_constantN_13 + (Unnamed Layer* 73) [Shuffle]_(Unnamed Layer* 73) [Shuffle]_output + AddU-4 + ReLUU-4, Tactic: 3899310434460415852, ReLUD-3_out_tensor[Half(31,256,1,1)] -> ReLUU-4_out_tensor[Half(31,2048,1,1)]
Layer(CaskConvolution): MMD-4 + constantK_15 + (Unnamed Layer* 81) [Shuffle] + unsqueeze_node_after_constantK_15 + (Unnamed Layer* 81) [Shuffle]_(Unnamed Layer* 81) [Shuffle]_output + AddD-4 + ReLUD-4, Tactic: -1828674067637788567, ReLUU-4_out_tensor[Half(31,2048,1,1)] -> ReLUD-4_out_tensor[Half(31,256,1,1)]
Layer(CaskConvolution): MMU-5 + constantN_17 + (Unnamed Layer* 89) [Shuffle] + unsqueeze_node_after_constantN_17 + (Unnamed Layer* 89) [Shuffle]_(Unnamed Layer* 89) [Shuffle]_output + AddU-5 + ReLUU-5, Tactic: 3899310434460415852, ReLUD-4_out_tensor[Half(31,256,1,1)] -> ReLUU-5_out_tensor[Half(31,2048,1,1)]
Layer(CaskConvolution): MMD-5 + constantK_19 + (Unnamed Layer* 97) [Shuffle] + unsqueeze_node_after_constantK_19 + (Unnamed Layer* 97) [Shuffle]_(Unnamed Layer* 97) [Shuffle]_output + AddD-5 + ReLUD-5, Tactic: -1828674067637788567, ReLUU-5_out_tensor[Half(31,2048,1,1)] -> ReLUD-5_out_tensor[Half(31,256,1,1)]
Layer(CaskConvolution): MMU-6 + constantN_21 + (Unnamed Layer* 105) [Shuffle] + unsqueeze_node_after_constantN_21 + (Unnamed Layer* 105) [Shuffle]_(Unnamed Layer* 105) [Shuffle]_output + AddU-6 + ReLUU-6, Tactic: 3899310434460415852, ReLUD-5_out_tensor[Half(31,256,1,1)] -> ReLUU-6_out_tensor[Half(31,2048,1,1)]
Layer(CaskConvolution): MMD-6 + constantK_23 + (Unnamed Layer* 113) [Shuffle] + unsqueeze_node_after_constantK_23 + (Unnamed Layer* 113) [Shuffle]_(Unnamed Layer* 113) [Shuffle]_output + AddD-6 + ReLUD-6, Tactic: -1828674067637788567, ReLUU-6_out_tensor[Half(31,2048,1,1)] -> ReLUD-6_out_tensor[Half(31,256,1,1)]
Layer(CaskConvolution): MMU-7 + constantN_25 + (Unnamed Layer* 121) [Shuffle] + unsqueeze_node_after_constantN_25 + (Unnamed Layer* 121) [Shuffle]_(Unnamed Layer* 121) [Shuffle]_output + AddU-7 + ReLUU-7, Tactic: 3899310434460415852, ReLUD-6_out_tensor[Half(31,256,1,1)] -> ReLUU-7_out_tensor[Half(31,2048,1,1)]
Layer(CaskConvolution): MMD-7 + constantK_27 + (Unnamed Layer* 129) [Shuffle] + unsqueeze_node_after_constantK_27 + (Unnamed Layer* 129) [Shuffle]_(Unnamed Layer* 129) [Shuffle]_output + AddD-7 + ReLUD-7, Tactic: -1828674067637788567, ReLUU-7_out_tensor[Half(31,2048,1,1)] -> ReLUD-7_out_tensor[Half(31,256,1,1)]
Layer(CaskConvolution): MMU-8 + constantN_29 + (Unnamed Layer* 137) [Shuffle] + unsqueeze_node_after_constantN_29 + (Unnamed Layer* 137) [Shuffle]_(Unnamed Layer* 137) [Shuffle]_output + AddU-8 + ReLUU-8, Tactic: 3899310434460415852, ReLUD-7_out_tensor[Half(31,256,1,1)] -> ReLUU-8_out_tensor[Half(31,2048,1,1)]
Layer(CaskConvolution): MMD-8 + constantK_31 + (Unnamed Layer* 145) [Shuffle] + unsqueeze_node_after_constantK_31 + (Unnamed Layer* 145) [Shuffle]_(Unnamed Layer* 145) [Shuffle]_output + AddD-8 + ReLUD-8, Tactic: -1828674067637788567, ReLUU-8_out_tensor[Half(31,2048,1,1)] -> ReLUD-8_out_tensor[Half(31,256,1,1)]
Layer(CaskConvolution): MMU-9 + constantN_33 + (Unnamed Layer* 153) [Shuffle] + unsqueeze_node_after_constantN_33 + (Unnamed Layer* 153) [Shuffle]_(Unnamed Layer* 153) [Shuffle]_output + AddU-9 + ReLUU-9, Tactic: 3899310434460415852, ReLUD-8_out_tensor[Half(31,256,1,1)] -> ReLUU-9_out_tensor[Half(31,2048,1,1)]
Layer(CaskConvolution): MMD-9 + constantK_35 + (Unnamed Layer* 161) [Shuffle] + unsqueeze_node_after_constantK_35 + (Unnamed Layer* 161) [Shuffle]_(Unnamed Layer* 161) [Shuffle]_output + AddD-9 + ReLUD-9, Tactic: -1828674067637788567, ReLUU-9_out_tensor[Half(31,2048,1,1)] -> ReLUD-9_out_tensor[Half(31,256,1,1)]
Layer(NoOp): Reformatting CopyNode for Input Tensor 0 to squeeze_after_ReLUD-9, Tactic: 0, ReLUD-9_out_tensor[Half(31,256,1,1)] -> Reformatted Input Tensor 0 to squeeze_after_ReLUD-9[Half(31,256,1,1)]
Layer(NoOp): squeeze_after_ReLUD-9, Tactic: 0, Reformatted Input Tensor 0 to squeeze_after_ReLUD-9[Half(31,256,1,1)] -> squeeze_after_ReLUD-9_out_tensor[Half(31,256)]
Layer(Reduce): Reduce, Tactic: 1, squeeze_after_ReLUD-9_out_tensor[Half(31,256)] -> Reformatted Output Tensor 0 to Reduce[Half(31)]
Layer(Reformat): Reformatting CopyNode for Output Tensor 0 to Reduce, Tactic: 0, Reformatted Output Tensor 0 to Reduce[Half(31)] -> tensor8[Float(31)]
```

+ Result of performance test
```
[I] === Performance summary ===
[I] Throughput: 6047.47 qps
[I] Latency: min = 0.159668 ms, max = 0.227295 ms, mean = 0.161587 ms, median = 0.161743 ms, percentile(99%) = 0.162842 ms
[I] End-to-End Host Latency: min = 0.159668 ms, max = 0.227295 ms, mean = 0.161587 ms, median = 0.161743 ms, percentile(99%) = 0.162842 ms
[I] Enqueue Time: min = 0.00610352 ms, max = 0.0974121 ms, mean = 0.0111169 ms, median = 0.0107422 ms, percentile(99%) = 0.019043 ms
[I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] GPU Compute Time: min = 0.159668 ms, max = 0.227295 ms, mean = 0.161587 ms, median = 0.161743 ms, percentile(99%) = 0.162842 ms
[I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] Total Host Walltime: 3.00043 s
[I] Total GPU Compute Time: 2.93199 s
```

# [M=32, K=255, N=2048]
+ Structure of the network
```
[V] Engine Layer Information:
Layer(NoOp): (Unnamed Layer* 1) [Shuffle], Tactic: 0, tensor0[Float(32,1)] -> (Unnamed Layer* 1) [Shuffle]_output[Float(32,1,1,1)]
Layer(CublasConvolution): MMU1, Tactic: 0, (Unnamed Layer* 1) [Shuffle]_output[Float(32,1,1,1)] -> (Unnamed Layer* 2) [Fully Connected]_output[Float(32,255,1,1)]
Layer(Reformat): Reformatting CopyNode for Input Tensor 0 to MMU-0 + constantN + (Unnamed Layer* 9) [Shuffle] + unsqueeze_node_after_constantN + (Unnamed Layer* 9) [Shuffle]_(Unnamed Layer* 9) [Shuffle]_output + AddU-0 + ReLUU-0, Tactic: 0, (Unnamed Layer* 2) [Fully Connected]_output[Float(32,255,1,1)] -> Reformatted Input Tensor 0 to MMU-0 + constantN + (Unnamed Layer* 9) [Shuffle] + unsqueeze_node_after_constantN + (Unnamed Layer* 9) [Shuffle]_(Unnamed Layer* 9) [Shuffle]_output + AddU-0 + ReLUU-0[Half(32,255,1,1)]
Layer(CaskConvolution): MMU-0 + constantN + (Unnamed Layer* 9) [Shuffle] + unsqueeze_node_after_constantN + (Unnamed Layer* 9) [Shuffle]_(Unnamed Layer* 9) [Shuffle]_output + AddU-0 + ReLUU-0, Tactic: 3899310434460415852, Reformatted Input Tensor 0 to MMU-0 + constantN + (Unnamed Layer* 9) [Shuffle] + unsqueeze_node_after_constantN + (Unnamed Layer* 9) [Shuffle]_(Unnamed Layer* 9) [Shuffle]_output + AddU-0 + ReLUU-0[Half(32,255,1,1)] -> ReLUU-0_out_tensor[Half(32,2048,1,1)]
Layer(CaskConvolution): MMD-0 + constantK + (Unnamed Layer* 17) [Shuffle] + unsqueeze_node_after_constantK + (Unnamed Layer* 17) [Shuffle]_(Unnamed Layer* 17) [Shuffle]_output + AddD-0 + ReLUD-0, Tactic: -1828674067637788567, ReLUU-0_out_tensor[Half(32,2048,1,1)] -> ReLUD-0_out_tensor[Half(32,255,1,1)]
Layer(CaskConvolution): MMU-1 + constantN_1 + (Unnamed Layer* 25) [Shuffle] + unsqueeze_node_after_constantN_1 + (Unnamed Layer* 25) [Shuffle]_(Unnamed Layer* 25) [Shuffle]_output + AddU-1 + ReLUU-1, Tactic: 3899310434460415852, ReLUD-0_out_tensor[Half(32,255,1,1)] -> ReLUU-1_out_tensor[Half(32,2048,1,1)]
Layer(CaskConvolution): MMD-1 + constantK_3 + (Unnamed Layer* 33) [Shuffle] + unsqueeze_node_after_constantK_3 + (Unnamed Layer* 33) [Shuffle]_(Unnamed Layer* 33) [Shuffle]_output + AddD-1 + ReLUD-1, Tactic: -1828674067637788567, ReLUU-1_out_tensor[Half(32,2048,1,1)] -> ReLUD-1_out_tensor[Half(32,255,1,1)]
Layer(CaskConvolution): MMU-2 + constantN_5 + (Unnamed Layer* 41) [Shuffle] + unsqueeze_node_after_constantN_5 + (Unnamed Layer* 41) [Shuffle]_(Unnamed Layer* 41) [Shuffle]_output + AddU-2 + ReLUU-2, Tactic: 3899310434460415852, ReLUD-1_out_tensor[Half(32,255,1,1)] -> ReLUU-2_out_tensor[Half(32,2048,1,1)]
Layer(CaskConvolution): MMD-2 + constantK_7 + (Unnamed Layer* 49) [Shuffle] + unsqueeze_node_after_constantK_7 + (Unnamed Layer* 49) [Shuffle]_(Unnamed Layer* 49) [Shuffle]_output + AddD-2 + ReLUD-2, Tactic: -1828674067637788567, ReLUU-2_out_tensor[Half(32,2048,1,1)] -> ReLUD-2_out_tensor[Half(32,255,1,1)]
Layer(CaskConvolution): MMU-3 + constantN_9 + (Unnamed Layer* 57) [Shuffle] + unsqueeze_node_after_constantN_9 + (Unnamed Layer* 57) [Shuffle]_(Unnamed Layer* 57) [Shuffle]_output + AddU-3 + ReLUU-3, Tactic: 3899310434460415852, ReLUD-2_out_tensor[Half(32,255,1,1)] -> ReLUU-3_out_tensor[Half(32,2048,1,1)]
Layer(CaskConvolution): MMD-3 + constantK_11 + (Unnamed Layer* 65) [Shuffle] + unsqueeze_node_after_constantK_11 + (Unnamed Layer* 65) [Shuffle]_(Unnamed Layer* 65) [Shuffle]_output + AddD-3 + ReLUD-3, Tactic: -1828674067637788567, ReLUU-3_out_tensor[Half(32,2048,1,1)] -> ReLUD-3_out_tensor[Half(32,255,1,1)]
Layer(CaskConvolution): MMU-4 + constantN_13 + (Unnamed Layer* 73) [Shuffle] + unsqueeze_node_after_constantN_13 + (Unnamed Layer* 73) [Shuffle]_(Unnamed Layer* 73) [Shuffle]_output + AddU-4 + ReLUU-4, Tactic: 3899310434460415852, ReLUD-3_out_tensor[Half(32,255,1,1)] -> ReLUU-4_out_tensor[Half(32,2048,1,1)]
Layer(CaskConvolution): MMD-4 + constantK_15 + (Unnamed Layer* 81) [Shuffle] + unsqueeze_node_after_constantK_15 + (Unnamed Layer* 81) [Shuffle]_(Unnamed Layer* 81) [Shuffle]_output + AddD-4 + ReLUD-4, Tactic: -1828674067637788567, ReLUU-4_out_tensor[Half(32,2048,1,1)] -> ReLUD-4_out_tensor[Half(32,255,1,1)]
Layer(CaskConvolution): MMU-5 + constantN_17 + (Unnamed Layer* 89) [Shuffle] + unsqueeze_node_after_constantN_17 + (Unnamed Layer* 89) [Shuffle]_(Unnamed Layer* 89) [Shuffle]_output + AddU-5 + ReLUU-5, Tactic: 3899310434460415852, ReLUD-4_out_tensor[Half(32,255,1,1)] -> ReLUU-5_out_tensor[Half(32,2048,1,1)]
Layer(CaskConvolution): MMD-5 + constantK_19 + (Unnamed Layer* 97) [Shuffle] + unsqueeze_node_after_constantK_19 + (Unnamed Layer* 97) [Shuffle]_(Unnamed Layer* 97) [Shuffle]_output + AddD-5 + ReLUD-5, Tactic: -1828674067637788567, ReLUU-5_out_tensor[Half(32,2048,1,1)] -> ReLUD-5_out_tensor[Half(32,255,1,1)]
Layer(CaskConvolution): MMU-6 + constantN_21 + (Unnamed Layer* 105) [Shuffle] + unsqueeze_node_after_constantN_21 + (Unnamed Layer* 105) [Shuffle]_(Unnamed Layer* 105) [Shuffle]_output + AddU-6 + ReLUU-6, Tactic: 3899310434460415852, ReLUD-5_out_tensor[Half(32,255,1,1)] -> ReLUU-6_out_tensor[Half(32,2048,1,1)]
Layer(CaskConvolution): MMD-6 + constantK_23 + (Unnamed Layer* 113) [Shuffle] + unsqueeze_node_after_constantK_23 + (Unnamed Layer* 113) [Shuffle]_(Unnamed Layer* 113) [Shuffle]_output + AddD-6 + ReLUD-6, Tactic: -1828674067637788567, ReLUU-6_out_tensor[Half(32,2048,1,1)] -> ReLUD-6_out_tensor[Half(32,255,1,1)]
Layer(CaskConvolution): MMU-7 + constantN_25 + (Unnamed Layer* 121) [Shuffle] + unsqueeze_node_after_constantN_25 + (Unnamed Layer* 121) [Shuffle]_(Unnamed Layer* 121) [Shuffle]_output + AddU-7 + ReLUU-7, Tactic: 3899310434460415852, ReLUD-6_out_tensor[Half(32,255,1,1)] -> ReLUU-7_out_tensor[Half(32,2048,1,1)]
Layer(CaskConvolution): MMD-7 + constantK_27 + (Unnamed Layer* 129) [Shuffle] + unsqueeze_node_after_constantK_27 + (Unnamed Layer* 129) [Shuffle]_(Unnamed Layer* 129) [Shuffle]_output + AddD-7 + ReLUD-7, Tactic: -1828674067637788567, ReLUU-7_out_tensor[Half(32,2048,1,1)] -> ReLUD-7_out_tensor[Half(32,255,1,1)]
Layer(CaskConvolution): MMU-8 + constantN_29 + (Unnamed Layer* 137) [Shuffle] + unsqueeze_node_after_constantN_29 + (Unnamed Layer* 137) [Shuffle]_(Unnamed Layer* 137) [Shuffle]_output + AddU-8 + ReLUU-8, Tactic: 3899310434460415852, ReLUD-7_out_tensor[Half(32,255,1,1)] -> ReLUU-8_out_tensor[Half(32,2048,1,1)]
Layer(CaskConvolution): MMD-8 + constantK_31 + (Unnamed Layer* 145) [Shuffle] + unsqueeze_node_after_constantK_31 + (Unnamed Layer* 145) [Shuffle]_(Unnamed Layer* 145) [Shuffle]_output + AddD-8 + ReLUD-8, Tactic: -1828674067637788567, ReLUU-8_out_tensor[Half(32,2048,1,1)] -> ReLUD-8_out_tensor[Half(32,255,1,1)]
Layer(CaskConvolution): MMU-9 + constantN_33 + (Unnamed Layer* 153) [Shuffle] + unsqueeze_node_after_constantN_33 + (Unnamed Layer* 153) [Shuffle]_(Unnamed Layer* 153) [Shuffle]_output + AddU-9 + ReLUU-9, Tactic: 3899310434460415852, ReLUD-8_out_tensor[Half(32,255,1,1)] -> ReLUU-9_out_tensor[Half(32,2048,1,1)]
Layer(CaskConvolution): MMD-9 + constantK_35 + (Unnamed Layer* 161) [Shuffle] + unsqueeze_node_after_constantK_35 + (Unnamed Layer* 161) [Shuffle]_(Unnamed Layer* 161) [Shuffle]_output + AddD-9 + ReLUD-9, Tactic: -1828674067637788567, ReLUU-9_out_tensor[Half(32,2048,1,1)] -> ReLUD-9_out_tensor[Half(32,255,1,1)]
Layer(Reformat): Reformatting CopyNode for Input Tensor 0 to squeeze_after_ReLUD-9, Tactic: 0, ReLUD-9_out_tensor[Half(32,255,1,1)] -> Reformatted Input Tensor 0 to squeeze_after_ReLUD-9[Float(32,255,1,1)]
Layer(NoOp): squeeze_after_ReLUD-9, Tactic: 0, Reformatted Input Tensor 0 to squeeze_after_ReLUD-9[Float(32,255,1,1)] -> squeeze_after_ReLUD-9_out_tensor[Float(32,255)]
Layer(Reduce): Reduce, Tactic: 0, squeeze_after_ReLUD-9_out_tensor[Float(32,255)] -> tensor8[Float(32)]
```

+ Result of performance test
```
[I] === Performance summary ===
[I] Throughput: 6061.93 qps
[I] Latency: min = 0.159668 ms, max = 0.201721 ms, mean = 0.16192 ms, median = 0.161621 ms, percentile(99%) = 0.199677 ms
[I] End-to-End Host Latency: min = 0.159668 ms, max = 0.201721 ms, mean = 0.16192 ms, median = 0.161621 ms, percentile(99%) = 0.199677 ms
[I] Enqueue Time: min = 0.0100098 ms, max = 0.0261841 ms, mean = 0.0107642 ms, median = 0.0107422 ms, percentile(99%) = 0.0113525 ms
[I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] GPU Compute Time: min = 0.159668 ms, max = 0.201721 ms, mean = 0.16192 ms, median = 0.161621 ms, percentile(99%) = 0.199677 ms
[I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] Total Host Walltime: 3.00037 s
[I] Total GPU Compute Time: 2.94499 s
```

# [M=32, K=256, N=2047]
+ Structure of the network
```
[V] Engine Layer Information:
Layer(NoOp): (Unnamed Layer* 1) [Shuffle], Tactic: 0, tensor0[Float(32,1)] -> (Unnamed Layer* 1) [Shuffle]_output[Float(32,1,1,1)]
Layer(Reformat): Reformatting CopyNode for Input Tensor 0 to MMU1, Tactic: 0, (Unnamed Layer* 1) [Shuffle]_output[Float(32,1,1,1)] -> Reformatted Input Tensor 0 to MMU1[Half(32,1,1,1)]
Layer(CublasConvolution): MMU1, Tactic: 0, Reformatted Input Tensor 0 to MMU1[Half(32,1,1,1)] -> (Unnamed Layer* 2) [Fully Connected]_output[Half(32,256,1,1)]
Layer(NoOp): Reformatting CopyNode for Input Tensor 0 to MMU-0 + constantN + (Unnamed Layer* 9) [Shuffle] + unsqueeze_node_after_constantN + (Unnamed Layer* 9) [Shuffle]_(Unnamed Layer* 9) [Shuffle]_output + AddU-0 + ReLUU-0, Tactic: 0, (Unnamed Layer* 2) [Fully Connected]_output[Half(32,256,1,1)] -> Reformatted Input Tensor 0 to MMU-0 + constantN + (Unnamed Layer* 9) [Shuffle] + unsqueeze_node_after_constantN + (Unnamed Layer* 9) [Shuffle]_(Unnamed Layer* 9) [Shuffle]_output + AddU-0 + ReLUU-0[Half(32,256,1,1)]
Layer(CaskConvolution): MMU-0 + constantN + (Unnamed Layer* 9) [Shuffle] + unsqueeze_node_after_constantN + (Unnamed Layer* 9) [Shuffle]_(Unnamed Layer* 9) [Shuffle]_output + AddU-0 + ReLUU-0, Tactic: 3899310434460415852, Reformatted Input Tensor 0 to MMU-0 + constantN + (Unnamed Layer* 9) [Shuffle] + unsqueeze_node_after_constantN + (Unnamed Layer* 9) [Shuffle]_(Unnamed Layer* 9) [Shuffle]_output + AddU-0 + ReLUU-0[Half(32,256,1,1)] -> ReLUU-0_out_tensor[Half(32,2047,1,1)]
Layer(CaskConvolution): MMD-0 + constantK + (Unnamed Layer* 17) [Shuffle] + unsqueeze_node_after_constantK + (Unnamed Layer* 17) [Shuffle]_(Unnamed Layer* 17) [Shuffle]_output + AddD-0 + ReLUD-0, Tactic: -1828674067637788567, ReLUU-0_out_tensor[Half(32,2047,1,1)] -> ReLUD-0_out_tensor[Half(32,256,1,1)]
Layer(CaskConvolution): MMU-1 + constantN_1 + (Unnamed Layer* 25) [Shuffle] + unsqueeze_node_after_constantN_1 + (Unnamed Layer* 25) [Shuffle]_(Unnamed Layer* 25) [Shuffle]_output + AddU-1 + ReLUU-1, Tactic: 3899310434460415852, ReLUD-0_out_tensor[Half(32,256,1,1)] -> ReLUU-1_out_tensor[Half(32,2047,1,1)]
Layer(CaskConvolution): MMD-1 + constantK_3 + (Unnamed Layer* 33) [Shuffle] + unsqueeze_node_after_constantK_3 + (Unnamed Layer* 33) [Shuffle]_(Unnamed Layer* 33) [Shuffle]_output + AddD-1 + ReLUD-1, Tactic: -1828674067637788567, ReLUU-1_out_tensor[Half(32,2047,1,1)] -> ReLUD-1_out_tensor[Half(32,256,1,1)]
Layer(CaskConvolution): MMU-2 + constantN_5 + (Unnamed Layer* 41) [Shuffle] + unsqueeze_node_after_constantN_5 + (Unnamed Layer* 41) [Shuffle]_(Unnamed Layer* 41) [Shuffle]_output + AddU-2 + ReLUU-2, Tactic: 3899310434460415852, ReLUD-1_out_tensor[Half(32,256,1,1)] -> ReLUU-2_out_tensor[Half(32,2047,1,1)]
Layer(CaskConvolution): MMD-2 + constantK_7 + (Unnamed Layer* 49) [Shuffle] + unsqueeze_node_after_constantK_7 + (Unnamed Layer* 49) [Shuffle]_(Unnamed Layer* 49) [Shuffle]_output + AddD-2 + ReLUD-2, Tactic: -1828674067637788567, ReLUU-2_out_tensor[Half(32,2047,1,1)] -> ReLUD-2_out_tensor[Half(32,256,1,1)]
Layer(CaskConvolution): MMU-3 + constantN_9 + (Unnamed Layer* 57) [Shuffle] + unsqueeze_node_after_constantN_9 + (Unnamed Layer* 57) [Shuffle]_(Unnamed Layer* 57) [Shuffle]_output + AddU-3 + ReLUU-3, Tactic: 3899310434460415852, ReLUD-2_out_tensor[Half(32,256,1,1)] -> ReLUU-3_out_tensor[Half(32,2047,1,1)]
Layer(CaskConvolution): MMD-3 + constantK_11 + (Unnamed Layer* 65) [Shuffle] + unsqueeze_node_after_constantK_11 + (Unnamed Layer* 65) [Shuffle]_(Unnamed Layer* 65) [Shuffle]_output + AddD-3 + ReLUD-3, Tactic: -1828674067637788567, ReLUU-3_out_tensor[Half(32,2047,1,1)] -> ReLUD-3_out_tensor[Half(32,256,1,1)]
Layer(CaskConvolution): MMU-4 + constantN_13 + (Unnamed Layer* 73) [Shuffle] + unsqueeze_node_after_constantN_13 + (Unnamed Layer* 73) [Shuffle]_(Unnamed Layer* 73) [Shuffle]_output + AddU-4 + ReLUU-4, Tactic: 3899310434460415852, ReLUD-3_out_tensor[Half(32,256,1,1)] -> ReLUU-4_out_tensor[Half(32,2047,1,1)]
Layer(CaskConvolution): MMD-4 + constantK_15 + (Unnamed Layer* 81) [Shuffle] + unsqueeze_node_after_constantK_15 + (Unnamed Layer* 81) [Shuffle]_(Unnamed Layer* 81) [Shuffle]_output + AddD-4 + ReLUD-4, Tactic: -1828674067637788567, ReLUU-4_out_tensor[Half(32,2047,1,1)] -> ReLUD-4_out_tensor[Half(32,256,1,1)]
Layer(CaskConvolution): MMU-5 + constantN_17 + (Unnamed Layer* 89) [Shuffle] + unsqueeze_node_after_constantN_17 + (Unnamed Layer* 89) [Shuffle]_(Unnamed Layer* 89) [Shuffle]_output + AddU-5 + ReLUU-5, Tactic: 3899310434460415852, ReLUD-4_out_tensor[Half(32,256,1,1)] -> ReLUU-5_out_tensor[Half(32,2047,1,1)]
Layer(CaskConvolution): MMD-5 + constantK_19 + (Unnamed Layer* 97) [Shuffle] + unsqueeze_node_after_constantK_19 + (Unnamed Layer* 97) [Shuffle]_(Unnamed Layer* 97) [Shuffle]_output + AddD-5 + ReLUD-5, Tactic: -1828674067637788567, ReLUU-5_out_tensor[Half(32,2047,1,1)] -> ReLUD-5_out_tensor[Half(32,256,1,1)]
Layer(CaskConvolution): MMU-6 + constantN_21 + (Unnamed Layer* 105) [Shuffle] + unsqueeze_node_after_constantN_21 + (Unnamed Layer* 105) [Shuffle]_(Unnamed Layer* 105) [Shuffle]_output + AddU-6 + ReLUU-6, Tactic: 3899310434460415852, ReLUD-5_out_tensor[Half(32,256,1,1)] -> ReLUU-6_out_tensor[Half(32,2047,1,1)]
Layer(CaskConvolution): MMD-6 + constantK_23 + (Unnamed Layer* 113) [Shuffle] + unsqueeze_node_after_constantK_23 + (Unnamed Layer* 113) [Shuffle]_(Unnamed Layer* 113) [Shuffle]_output + AddD-6 + ReLUD-6, Tactic: -1828674067637788567, ReLUU-6_out_tensor[Half(32,2047,1,1)] -> ReLUD-6_out_tensor[Half(32,256,1,1)]
Layer(CaskConvolution): MMU-7 + constantN_25 + (Unnamed Layer* 121) [Shuffle] + unsqueeze_node_after_constantN_25 + (Unnamed Layer* 121) [Shuffle]_(Unnamed Layer* 121) [Shuffle]_output + AddU-7 + ReLUU-7, Tactic: 3899310434460415852, ReLUD-6_out_tensor[Half(32,256,1,1)] -> ReLUU-7_out_tensor[Half(32,2047,1,1)]
Layer(CaskConvolution): MMD-7 + constantK_27 + (Unnamed Layer* 129) [Shuffle] + unsqueeze_node_after_constantK_27 + (Unnamed Layer* 129) [Shuffle]_(Unnamed Layer* 129) [Shuffle]_output + AddD-7 + ReLUD-7, Tactic: -1828674067637788567, ReLUU-7_out_tensor[Half(32,2047,1,1)] -> ReLUD-7_out_tensor[Half(32,256,1,1)]
Layer(CaskConvolution): MMU-8 + constantN_29 + (Unnamed Layer* 137) [Shuffle] + unsqueeze_node_after_constantN_29 + (Unnamed Layer* 137) [Shuffle]_(Unnamed Layer* 137) [Shuffle]_output + AddU-8 + ReLUU-8, Tactic: 3899310434460415852, ReLUD-7_out_tensor[Half(32,256,1,1)] -> ReLUU-8_out_tensor[Half(32,2047,1,1)]
Layer(CaskConvolution): MMD-8 + constantK_31 + (Unnamed Layer* 145) [Shuffle] + unsqueeze_node_after_constantK_31 + (Unnamed Layer* 145) [Shuffle]_(Unnamed Layer* 145) [Shuffle]_output + AddD-8 + ReLUD-8, Tactic: -1828674067637788567, ReLUU-8_out_tensor[Half(32,2047,1,1)] -> ReLUD-8_out_tensor[Half(32,256,1,1)]
Layer(CaskConvolution): MMU-9 + constantN_33 + (Unnamed Layer* 153) [Shuffle] + unsqueeze_node_after_constantN_33 + (Unnamed Layer* 153) [Shuffle]_(Unnamed Layer* 153) [Shuffle]_output + AddU-9 + ReLUU-9, Tactic: 3899310434460415852, ReLUD-8_out_tensor[Half(32,256,1,1)] -> ReLUU-9_out_tensor[Half(32,2047,1,1)]
Layer(CaskConvolution): MMD-9 + constantK_35 + (Unnamed Layer* 161) [Shuffle] + unsqueeze_node_after_constantK_35 + (Unnamed Layer* 161) [Shuffle]_(Unnamed Layer* 161) [Shuffle]_output + AddD-9 + ReLUD-9, Tactic: -1828674067637788567, ReLUU-9_out_tensor[Half(32,2047,1,1)] -> ReLUD-9_out_tensor[Half(32,256,1,1)]
Layer(Reformat): Reformatting CopyNode for Input Tensor 0 to squeeze_after_ReLUD-9, Tactic: 0, ReLUD-9_out_tensor[Half(32,256,1,1)] -> Reformatted Input Tensor 0 to squeeze_after_ReLUD-9[Float(32,256,1,1)]
Layer(NoOp): squeeze_after_ReLUD-9, Tactic: 0, Reformatted Input Tensor 0 to squeeze_after_ReLUD-9[Float(32,256,1,1)] -> squeeze_after_ReLUD-9_out_tensor[Float(32,256)]
Layer(Reduce): Reduce, Tactic: 0, squeeze_after_ReLUD-9_out_tensor[Float(32,256)] -> tensor8[Float(32)]
```

+ Result of performance test
```
[I] === Performance summary ===
[I] Throughput: 6024.82 qps
[I] Latency: min = 0.159668 ms, max = 0.200714 ms, mean = 0.161828 ms, median = 0.161743 ms, percentile(99%) = 0.198654 ms
[I] End-to-End Host Latency: min = 0.159668 ms, max = 0.200714 ms, mean = 0.161828 ms, median = 0.161743 ms, percentile(99%) = 0.198654 ms
[I] Enqueue Time: min = 0.00952148 ms, max = 0.0385742 ms, mean = 0.0108875 ms, median = 0.0106201 ms, percentile(99%) = 0.0153809 ms
[I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] GPU Compute Time: min = 0.159668 ms, max = 0.200714 ms, mean = 0.161828 ms, median = 0.161743 ms, percentile(99%) = 0.198654 ms
[I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] Total Host Walltime: 3.00042 s
[I] Total GPU Compute Time: 2.92537 s
```
