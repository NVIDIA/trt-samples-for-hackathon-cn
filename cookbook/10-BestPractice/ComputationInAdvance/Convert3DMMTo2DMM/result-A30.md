# 3D Matrix Multiplication
+ Structure of the network
```
[V] Engine Layer Information:
Layer(Constant): constant1x256 + (Unnamed Layer* 1) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 1) [Shuffle]_output[Float(1,1,256)]
Layer(MatrixMultiply): MMU0, Tactic: 1, tensor0[Float(-7,-8,1)], (Unnamed Layer* 1) [Shuffle]_output[Float(1,1,256)] -> tensor1[Float(-7,-8,256)]
Layer(Constant): constant256x2048 + (Unnamed Layer* 4) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 4) [Shuffle]_output[Float(1,256,2048)]
Layer(MatrixMultiply): MMU-0, Tactic: 1, tensor1[Float(-7,-8,256)], (Unnamed Layer* 4) [Shuffle]_output[Float(1,256,2048)] -> tensor0-1[Float(-7,-8,2048)]
Layer(Constant): constant2048 + (Unnamed Layer* 7) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 7) [Shuffle]_output[Float(1,1,2048)]
Layer(ElementWise): AddU-0 + ReLUU-0, Tactic: 1, tensor0-1[Float(-7,-8,2048)], (Unnamed Layer* 7) [Shuffle]_output[Float(1,1,2048)] -> tensor0-3[Float(-7,-8,2048)]
Layer(Constant): constant2048x256 + (Unnamed Layer* 11) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 11) [Shuffle]_output[Float(1,2048,256)]
Layer(MatrixMultiply): MMD-0, Tactic: 1, tensor0-3[Float(-7,-8,2048)], (Unnamed Layer* 11) [Shuffle]_output[Float(1,2048,256)] -> tensor0-4[Float(-7,-8,256)]
Layer(Constant): constant256 + (Unnamed Layer* 14) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 14) [Shuffle]_output[Float(1,1,256)]
Layer(ElementWise): AddD-0 + ReLUD-0, Tactic: 1, tensor0-4[Float(-7,-8,256)], (Unnamed Layer* 14) [Shuffle]_output[Float(1,1,256)] -> tensor0-6[Float(-7,-8,256)]
Layer(Constant): constant256x2048_0 + (Unnamed Layer* 18) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 18) [Shuffle]_output[Float(1,256,2048)]
Layer(MatrixMultiply): MMU-1, Tactic: 1, tensor0-6[Float(-7,-8,256)], (Unnamed Layer* 18) [Shuffle]_output[Float(1,256,2048)] -> tensor1-1[Float(-7,-8,2048)]
Layer(Constant): constant2048_1 + (Unnamed Layer* 21) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 21) [Shuffle]_output[Float(1,1,2048)]
Layer(ElementWise): AddU-1 + ReLUU-1, Tactic: 1, tensor1-1[Float(-7,-8,2048)], (Unnamed Layer* 21) [Shuffle]_output[Float(1,1,2048)] -> tensor1-3[Float(-7,-8,2048)]
Layer(Constant): constant2048x256_2 + (Unnamed Layer* 25) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 25) [Shuffle]_output[Float(1,2048,256)]
Layer(MatrixMultiply): MMD-1, Tactic: 1, tensor1-3[Float(-7,-8,2048)], (Unnamed Layer* 25) [Shuffle]_output[Float(1,2048,256)] -> tensor1-4[Float(-7,-8,256)]
Layer(Constant): constant256_3 + (Unnamed Layer* 28) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 28) [Shuffle]_output[Float(1,1,256)]
Layer(ElementWise): AddD-1 + ReLUD-1, Tactic: 1, tensor1-4[Float(-7,-8,256)], (Unnamed Layer* 28) [Shuffle]_output[Float(1,1,256)] -> tensor1-6[Float(-7,-8,256)]
Layer(Constant): constant256x2048_4 + (Unnamed Layer* 32) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 32) [Shuffle]_output[Float(1,256,2048)]
Layer(MatrixMultiply): MMU-2, Tactic: 1, tensor1-6[Float(-7,-8,256)], (Unnamed Layer* 32) [Shuffle]_output[Float(1,256,2048)] -> tensor2-1[Float(-7,-8,2048)]
Layer(Constant): constant2048_5 + (Unnamed Layer* 35) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 35) [Shuffle]_output[Float(1,1,2048)]
Layer(ElementWise): AddU-2 + ReLUU-2, Tactic: 1, tensor2-1[Float(-7,-8,2048)], (Unnamed Layer* 35) [Shuffle]_output[Float(1,1,2048)] -> tensor2-3[Float(-7,-8,2048)]
Layer(Constant): constant2048x256_6 + (Unnamed Layer* 39) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 39) [Shuffle]_output[Float(1,2048,256)]
Layer(MatrixMultiply): MMD-2, Tactic: 1, tensor2-3[Float(-7,-8,2048)], (Unnamed Layer* 39) [Shuffle]_output[Float(1,2048,256)] -> tensor2-4[Float(-7,-8,256)]
Layer(Constant): constant256_7 + (Unnamed Layer* 42) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 42) [Shuffle]_output[Float(1,1,256)]
Layer(ElementWise): AddD-2 + ReLUD-2, Tactic: 1, tensor2-4[Float(-7,-8,256)], (Unnamed Layer* 42) [Shuffle]_output[Float(1,1,256)] -> tensor2-6[Float(-7,-8,256)]
Layer(Constant): constant256x2048_8 + (Unnamed Layer* 46) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 46) [Shuffle]_output[Float(1,256,2048)]
Layer(MatrixMultiply): MMU-3, Tactic: 1, tensor2-6[Float(-7,-8,256)], (Unnamed Layer* 46) [Shuffle]_output[Float(1,256,2048)] -> tensor3-1[Float(-7,-8,2048)]
Layer(Constant): constant2048_9 + (Unnamed Layer* 49) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 49) [Shuffle]_output[Float(1,1,2048)]
Layer(ElementWise): AddU-3 + ReLUU-3, Tactic: 1, tensor3-1[Float(-7,-8,2048)], (Unnamed Layer* 49) [Shuffle]_output[Float(1,1,2048)] -> tensor3-3[Float(-7,-8,2048)]
Layer(Constant): constant2048x256_10 + (Unnamed Layer* 53) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 53) [Shuffle]_output[Float(1,2048,256)]
Layer(MatrixMultiply): MMD-3, Tactic: 1, tensor3-3[Float(-7,-8,2048)], (Unnamed Layer* 53) [Shuffle]_output[Float(1,2048,256)] -> tensor3-4[Float(-7,-8,256)]
Layer(Constant): constant256_11 + (Unnamed Layer* 56) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 56) [Shuffle]_output[Float(1,1,256)]
Layer(ElementWise): AddD-3 + ReLUD-3, Tactic: 1, tensor3-4[Float(-7,-8,256)], (Unnamed Layer* 56) [Shuffle]_output[Float(1,1,256)] -> tensor3-6[Float(-7,-8,256)]
Layer(Constant): constant256x2048_12 + (Unnamed Layer* 60) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 60) [Shuffle]_output[Float(1,256,2048)]
Layer(MatrixMultiply): MMU-4, Tactic: 1, tensor3-6[Float(-7,-8,256)], (Unnamed Layer* 60) [Shuffle]_output[Float(1,256,2048)] -> tensor4-1[Float(-7,-8,2048)]
Layer(Constant): constant2048_13 + (Unnamed Layer* 63) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 63) [Shuffle]_output[Float(1,1,2048)]
Layer(ElementWise): AddU-4 + ReLUU-4, Tactic: 1, tensor4-1[Float(-7,-8,2048)], (Unnamed Layer* 63) [Shuffle]_output[Float(1,1,2048)] -> tensor4-3[Float(-7,-8,2048)]
Layer(Constant): constant2048x256_14 + (Unnamed Layer* 67) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 67) [Shuffle]_output[Float(1,2048,256)]
Layer(MatrixMultiply): MMD-4, Tactic: 1, tensor4-3[Float(-7,-8,2048)], (Unnamed Layer* 67) [Shuffle]_output[Float(1,2048,256)] -> tensor4-4[Float(-7,-8,256)]
Layer(Constant): constant256_15 + (Unnamed Layer* 70) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 70) [Shuffle]_output[Float(1,1,256)]
Layer(ElementWise): AddD-4 + ReLUD-4, Tactic: 1, tensor4-4[Float(-7,-8,256)], (Unnamed Layer* 70) [Shuffle]_output[Float(1,1,256)] -> tensor4-6[Float(-7,-8,256)]
Layer(Constant): constant256x2048_16 + (Unnamed Layer* 74) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 74) [Shuffle]_output[Float(1,256,2048)]
Layer(MatrixMultiply): MMU-5, Tactic: 1, tensor4-6[Float(-7,-8,256)], (Unnamed Layer* 74) [Shuffle]_output[Float(1,256,2048)] -> tensor5-1[Float(-7,-8,2048)]
Layer(Constant): constant2048_17 + (Unnamed Layer* 77) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 77) [Shuffle]_output[Float(1,1,2048)]
Layer(ElementWise): AddU-5 + ReLUU-5, Tactic: 1, tensor5-1[Float(-7,-8,2048)], (Unnamed Layer* 77) [Shuffle]_output[Float(1,1,2048)] -> tensor5-3[Float(-7,-8,2048)]
Layer(Constant): constant2048x256_18 + (Unnamed Layer* 81) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 81) [Shuffle]_output[Float(1,2048,256)]
Layer(MatrixMultiply): MMD-5, Tactic: 1, tensor5-3[Float(-7,-8,2048)], (Unnamed Layer* 81) [Shuffle]_output[Float(1,2048,256)] -> tensor5-4[Float(-7,-8,256)]
Layer(Constant): constant256_19 + (Unnamed Layer* 84) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 84) [Shuffle]_output[Float(1,1,256)]
Layer(ElementWise): AddD-5 + ReLUD-5, Tactic: 1, tensor5-4[Float(-7,-8,256)], (Unnamed Layer* 84) [Shuffle]_output[Float(1,1,256)] -> tensor5-6[Float(-7,-8,256)]
Layer(Constant): constant256x2048_20 + (Unnamed Layer* 88) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 88) [Shuffle]_output[Float(1,256,2048)]
Layer(MatrixMultiply): MMU-6, Tactic: 1, tensor5-6[Float(-7,-8,256)], (Unnamed Layer* 88) [Shuffle]_output[Float(1,256,2048)] -> tensor6-1[Float(-7,-8,2048)]
Layer(Constant): constant2048_21 + (Unnamed Layer* 91) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 91) [Shuffle]_output[Float(1,1,2048)]
Layer(ElementWise): AddU-6 + ReLUU-6, Tactic: 1, tensor6-1[Float(-7,-8,2048)], (Unnamed Layer* 91) [Shuffle]_output[Float(1,1,2048)] -> tensor6-3[Float(-7,-8,2048)]
Layer(Constant): constant2048x256_22 + (Unnamed Layer* 95) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 95) [Shuffle]_output[Float(1,2048,256)]
Layer(MatrixMultiply): MMD-6, Tactic: 1, tensor6-3[Float(-7,-8,2048)], (Unnamed Layer* 95) [Shuffle]_output[Float(1,2048,256)] -> tensor6-4[Float(-7,-8,256)]
Layer(Constant): constant256_23 + (Unnamed Layer* 98) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 98) [Shuffle]_output[Float(1,1,256)]
Layer(ElementWise): AddD-6 + ReLUD-6, Tactic: 1, tensor6-4[Float(-7,-8,256)], (Unnamed Layer* 98) [Shuffle]_output[Float(1,1,256)] -> tensor6-6[Float(-7,-8,256)]
Layer(Constant): constant256x2048_24 + (Unnamed Layer* 102) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 102) [Shuffle]_output[Float(1,256,2048)]
Layer(MatrixMultiply): MMU-7, Tactic: 1, tensor6-6[Float(-7,-8,256)], (Unnamed Layer* 102) [Shuffle]_output[Float(1,256,2048)] -> tensor7-1[Float(-7,-8,2048)]
Layer(Constant): constant2048_25 + (Unnamed Layer* 105) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 105) [Shuffle]_output[Float(1,1,2048)]
Layer(ElementWise): AddU-7 + ReLUU-7, Tactic: 1, tensor7-1[Float(-7,-8,2048)], (Unnamed Layer* 105) [Shuffle]_output[Float(1,1,2048)] -> tensor7-3[Float(-7,-8,2048)]
Layer(Constant): constant2048x256_26 + (Unnamed Layer* 109) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 109) [Shuffle]_output[Float(1,2048,256)]
Layer(MatrixMultiply): MMD-7, Tactic: 1, tensor7-3[Float(-7,-8,2048)], (Unnamed Layer* 109) [Shuffle]_output[Float(1,2048,256)] -> tensor7-4[Float(-7,-8,256)]
Layer(Constant): constant256_27 + (Unnamed Layer* 112) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 112) [Shuffle]_output[Float(1,1,256)]
Layer(ElementWise): AddD-7 + ReLUD-7, Tactic: 1, tensor7-4[Float(-7,-8,256)], (Unnamed Layer* 112) [Shuffle]_output[Float(1,1,256)] -> tensor7-6[Float(-7,-8,256)]
Layer(Constant): constant256x2048_28 + (Unnamed Layer* 116) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 116) [Shuffle]_output[Float(1,256,2048)]
Layer(MatrixMultiply): MMU-8, Tactic: 1, tensor7-6[Float(-7,-8,256)], (Unnamed Layer* 116) [Shuffle]_output[Float(1,256,2048)] -> tensor8-1[Float(-7,-8,2048)]
Layer(Constant): constant2048_29 + (Unnamed Layer* 119) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 119) [Shuffle]_output[Float(1,1,2048)]
Layer(ElementWise): AddU-8 + ReLUU-8, Tactic: 1, tensor8-1[Float(-7,-8,2048)], (Unnamed Layer* 119) [Shuffle]_output[Float(1,1,2048)] -> tensor8-3[Float(-7,-8,2048)]
Layer(Constant): constant2048x256_30 + (Unnamed Layer* 123) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 123) [Shuffle]_output[Float(1,2048,256)]
Layer(MatrixMultiply): MMD-8, Tactic: 1, tensor8-3[Float(-7,-8,2048)], (Unnamed Layer* 123) [Shuffle]_output[Float(1,2048,256)] -> tensor8-4[Float(-7,-8,256)]
Layer(Constant): constant256_31 + (Unnamed Layer* 126) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 126) [Shuffle]_output[Float(1,1,256)]
Layer(ElementWise): AddD-8 + ReLUD-8, Tactic: 1, tensor8-4[Float(-7,-8,256)], (Unnamed Layer* 126) [Shuffle]_output[Float(1,1,256)] -> tensor8-6[Float(-7,-8,256)]
Layer(Constant): constant256x2048_32 + (Unnamed Layer* 130) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 130) [Shuffle]_output[Float(1,256,2048)]
Layer(MatrixMultiply): MMU-9, Tactic: 1, tensor8-6[Float(-7,-8,256)], (Unnamed Layer* 130) [Shuffle]_output[Float(1,256,2048)] -> tensor9-1[Float(-7,-8,2048)]
Layer(Constant): constant2048_33 + (Unnamed Layer* 133) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 133) [Shuffle]_output[Float(1,1,2048)]
Layer(ElementWise): AddU-9 + ReLUU-9, Tactic: 1, tensor9-1[Float(-7,-8,2048)], (Unnamed Layer* 133) [Shuffle]_output[Float(1,1,2048)] -> tensor9-3[Float(-7,-8,2048)]
Layer(Constant): constant2048x256_34 + (Unnamed Layer* 137) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 137) [Shuffle]_output[Float(1,2048,256)]
Layer(MatrixMultiply): MMD-9, Tactic: 1, tensor9-3[Float(-7,-8,2048)], (Unnamed Layer* 137) [Shuffle]_output[Float(1,2048,256)] -> tensor9-4[Float(-7,-8,256)]
Layer(Constant): constant256_35 + (Unnamed Layer* 140) [Shuffle], Tactic: 0,  -> (Unnamed Layer* 140) [Shuffle]_output[Float(1,1,256)]
Layer(ElementWise): AddD-9 + ReLUD-9, Tactic: 1, tensor9-4[Float(-7,-8,256)], (Unnamed Layer* 140) [Shuffle]_output[Float(1,1,256)] -> tensor9-6[Float(-7,-8,256)]
Layer(Reduce): Reduce, Tactic: 2, tensor9-6[Float(-7,-8,256)] -> tensor8[Float(-7,-8)]
```

+ trtexec Result of performance test
```
[I] === Performance summary ===
[I] Throughput: 155.33 qps
[I] Latency: min = 6.34647 ms, max = 6.71741 ms, mean = 6.43479 ms, median = 6.44275 ms, percentile(99%) = 6.70544 ms
[I] End-to-End Host Latency: min = 6.34647 ms, max = 6.71741 ms, mean = 6.43479 ms, median = 6.44275 ms, percentile(99%) = 6.70544 ms
[I] Enqueue Time: min = 0.0146484 ms, max = 0.0446167 ms, mean = 0.0166305 ms, median = 0.0158386 ms, percentile(99%) = 0.0266113 ms
[I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] GPU Compute Time: min = 6.34647 ms, max = 6.71741 ms, mean = 6.43479 ms, median = 6.44275 ms, percentile(99%) = 6.70544 ms
[I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] Total Host Walltime: 3.01294 s
[I] Total GPU Compute Time: 3.01148 s
```

# 2D Matrix Multiplication
+ Structure of the network
```
[V] Engine Layer Information:
Layer(NoOp): myReshapeN-input + (Unnamed Layer* 17) [Shuffle], Tactic: 0, tensor0[Float(-3,-4,1)] -> (Unnamed Layer* 17) [Shuffle]_output[Float(-6,1,1,1)]
Layer(CublasConvolution): MMU0, Tactic: 0, (Unnamed Layer* 17) [Shuffle]_output[Float(-6,1,1,1)] -> (Unnamed Layer* 18) [Fully Connected]_output[Float(-6,256,1,1)]
Layer(NoOp): Reformatting CopyNode for Input Tensor 0 to MMU-0 + constant2048 + (Unnamed Layer* 36) [Shuffle] + unsqueeze_node_after_constant2048 + (Unnamed Layer* 36) [Shuffle]_(Unnamed Layer* 36) [Shuffle]_output + AddU-0 + ReLUU-0, Tactic: 0, (Unnamed Layer* 18) [Fully Connected]_output[Float(-6,256,1,1)] -> Reformatted Input Tensor 0 to MMU-0 + constant2048 + (Unnamed Layer* 36) [Shuffle] + unsqueeze_node_after_constant2048 + (Unnamed Layer* 36) [Shuffle]_(Unnamed Layer* 36) [Shuffle]_output + AddU-0 + ReLUU-0[Float(-6,256,1,1)]
Layer(CaskConvolution): MMU-0 + constant2048 + (Unnamed Layer* 36) [Shuffle] + unsqueeze_node_after_constant2048 + (Unnamed Layer* 36) [Shuffle]_(Unnamed Layer* 36) [Shuffle]_output + AddU-0 + ReLUU-0, Tactic: -6073218138311523634, Reformatted Input Tensor 0 to MMU-0 + constant2048 + (Unnamed Layer* 36) [Shuffle] + unsqueeze_node_after_constant2048 + (Unnamed Layer* 36) [Shuffle]_(Unnamed Layer* 36) [Shuffle]_output + AddU-0 + ReLUU-0[Float(-6,256,1,1)] -> ReLUU-0_out_tensor[Float(-6,2048,1,1)]
Layer(CaskConvolution): MMD-0 + constant256 + (Unnamed Layer* 52) [Shuffle] + unsqueeze_node_after_constant256 + (Unnamed Layer* 52) [Shuffle]_(Unnamed Layer* 52) [Shuffle]_output + AddD-0 + ReLUD-0, Tactic: -7067026478815706014, ReLUU-0_out_tensor[Float(-6,2048,1,1)] -> ReLUD-0_out_tensor[Float(-6,256,1,1)]
Layer(CaskConvolution): MMU-1 + constant2048_1 + (Unnamed Layer* 68) [Shuffle] + unsqueeze_node_after_constant2048_1 + (Unnamed Layer* 68) [Shuffle]_(Unnamed Layer* 68) [Shuffle]_output + AddU-1 + ReLUU-1, Tactic: -6073218138311523634, ReLUD-0_out_tensor[Float(-6,256,1,1)] -> ReLUU-1_out_tensor[Float(-6,2048,1,1)]
Layer(CaskConvolution): MMD-1 + constant256_3 + (Unnamed Layer* 84) [Shuffle] + unsqueeze_node_after_constant256_3 + (Unnamed Layer* 84) [Shuffle]_(Unnamed Layer* 84) [Shuffle]_output + AddD-1 + ReLUD-1, Tactic: -7067026478815706014, ReLUU-1_out_tensor[Float(-6,2048,1,1)] -> ReLUD-1_out_tensor[Float(-6,256,1,1)]
Layer(CaskConvolution): MMU-2 + constant2048_5 + (Unnamed Layer* 100) [Shuffle] + unsqueeze_node_after_constant2048_5 + (Unnamed Layer* 100) [Shuffle]_(Unnamed Layer* 100) [Shuffle]_output + AddU-2 + ReLUU-2, Tactic: -6073218138311523634, ReLUD-1_out_tensor[Float(-6,256,1,1)] -> ReLUU-2_out_tensor[Float(-6,2048,1,1)]
Layer(CaskConvolution): MMD-2 + constant256_7 + (Unnamed Layer* 116) [Shuffle] + unsqueeze_node_after_constant256_7 + (Unnamed Layer* 116) [Shuffle]_(Unnamed Layer* 116) [Shuffle]_output + AddD-2 + ReLUD-2, Tactic: -7067026478815706014, ReLUU-2_out_tensor[Float(-6,2048,1,1)] -> ReLUD-2_out_tensor[Float(-6,256,1,1)]
Layer(CaskConvolution): MMU-3 + constant2048_9 + (Unnamed Layer* 132) [Shuffle] + unsqueeze_node_after_constant2048_9 + (Unnamed Layer* 132) [Shuffle]_(Unnamed Layer* 132) [Shuffle]_output + AddU-3 + ReLUU-3, Tactic: -6073218138311523634, ReLUD-2_out_tensor[Float(-6,256,1,1)] -> ReLUU-3_out_tensor[Float(-6,2048,1,1)]
Layer(CaskConvolution): MMD-3 + constant256_11 + (Unnamed Layer* 148) [Shuffle] + unsqueeze_node_after_constant256_11 + (Unnamed Layer* 148) [Shuffle]_(Unnamed Layer* 148) [Shuffle]_output + AddD-3 + ReLUD-3, Tactic: -7067026478815706014, ReLUU-3_out_tensor[Float(-6,2048,1,1)] -> ReLUD-3_out_tensor[Float(-6,256,1,1)]
Layer(CaskConvolution): MMU-4 + constant2048_13 + (Unnamed Layer* 164) [Shuffle] + unsqueeze_node_after_constant2048_13 + (Unnamed Layer* 164) [Shuffle]_(Unnamed Layer* 164) [Shuffle]_output + AddU-4 + ReLUU-4, Tactic: -6073218138311523634, ReLUD-3_out_tensor[Float(-6,256,1,1)] -> ReLUU-4_out_tensor[Float(-6,2048,1,1)]
Layer(CaskConvolution): MMD-4 + constant256_15 + (Unnamed Layer* 180) [Shuffle] + unsqueeze_node_after_constant256_15 + (Unnamed Layer* 180) [Shuffle]_(Unnamed Layer* 180) [Shuffle]_output + AddD-4 + ReLUD-4, Tactic: -7067026478815706014, ReLUU-4_out_tensor[Float(-6,2048,1,1)] -> ReLUD-4_out_tensor[Float(-6,256,1,1)]
Layer(CaskConvolution): MMU-5 + constant2048_17 + (Unnamed Layer* 196) [Shuffle] + unsqueeze_node_after_constant2048_17 + (Unnamed Layer* 196) [Shuffle]_(Unnamed Layer* 196) [Shuffle]_output + AddU-5 + ReLUU-5, Tactic: -6073218138311523634, ReLUD-4_out_tensor[Float(-6,256,1,1)] -> ReLUU-5_out_tensor[Float(-6,2048,1,1)]
Layer(CaskConvolution): MMD-5 + constant256_19 + (Unnamed Layer* 212) [Shuffle] + unsqueeze_node_after_constant256_19 + (Unnamed Layer* 212) [Shuffle]_(Unnamed Layer* 212) [Shuffle]_output + AddD-5 + ReLUD-5, Tactic: -7067026478815706014, ReLUU-5_out_tensor[Float(-6,2048,1,1)] -> ReLUD-5_out_tensor[Float(-6,256,1,1)]
Layer(CaskConvolution): MMU-6 + constant2048_21 + (Unnamed Layer* 228) [Shuffle] + unsqueeze_node_after_constant2048_21 + (Unnamed Layer* 228) [Shuffle]_(Unnamed Layer* 228) [Shuffle]_output + AddU-6 + ReLUU-6, Tactic: -6073218138311523634, ReLUD-5_out_tensor[Float(-6,256,1,1)] -> ReLUU-6_out_tensor[Float(-6,2048,1,1)]
Layer(CaskConvolution): MMD-6 + constant256_23 + (Unnamed Layer* 244) [Shuffle] + unsqueeze_node_after_constant256_23 + (Unnamed Layer* 244) [Shuffle]_(Unnamed Layer* 244) [Shuffle]_output + AddD-6 + ReLUD-6, Tactic: -7067026478815706014, ReLUU-6_out_tensor[Float(-6,2048,1,1)] -> ReLUD-6_out_tensor[Float(-6,256,1,1)]
Layer(CaskConvolution): MMU-7 + constant2048_25 + (Unnamed Layer* 260) [Shuffle] + unsqueeze_node_after_constant2048_25 + (Unnamed Layer* 260) [Shuffle]_(Unnamed Layer* 260) [Shuffle]_output + AddU-7 + ReLUU-7, Tactic: -6073218138311523634, ReLUD-6_out_tensor[Float(-6,256,1,1)] -> ReLUU-7_out_tensor[Float(-6,2048,1,1)]
Layer(CaskConvolution): MMD-7 + constant256_27 + (Unnamed Layer* 276) [Shuffle] + unsqueeze_node_after_constant256_27 + (Unnamed Layer* 276) [Shuffle]_(Unnamed Layer* 276) [Shuffle]_output + AddD-7 + ReLUD-7, Tactic: -7067026478815706014, ReLUU-7_out_tensor[Float(-6,2048,1,1)] -> ReLUD-7_out_tensor[Float(-6,256,1,1)]
Layer(CaskConvolution): MMU-8 + constant2048_29 + (Unnamed Layer* 292) [Shuffle] + unsqueeze_node_after_constant2048_29 + (Unnamed Layer* 292) [Shuffle]_(Unnamed Layer* 292) [Shuffle]_output + AddU-8 + ReLUU-8, Tactic: -6073218138311523634, ReLUD-7_out_tensor[Float(-6,256,1,1)] -> ReLUU-8_out_tensor[Float(-6,2048,1,1)]
Layer(CaskConvolution): MMD-8 + constant256_31 + (Unnamed Layer* 308) [Shuffle] + unsqueeze_node_after_constant256_31 + (Unnamed Layer* 308) [Shuffle]_(Unnamed Layer* 308) [Shuffle]_output + AddD-8 + ReLUD-8, Tactic: -7067026478815706014, ReLUU-8_out_tensor[Float(-6,2048,1,1)] -> ReLUD-8_out_tensor[Float(-6,256,1,1)]
Layer(CaskConvolution): MMU-9 + constant2048_33 + (Unnamed Layer* 324) [Shuffle] + unsqueeze_node_after_constant2048_33 + (Unnamed Layer* 324) [Shuffle]_(Unnamed Layer* 324) [Shuffle]_output + AddU-9 + ReLUU-9, Tactic: -6073218138311523634, ReLUD-8_out_tensor[Float(-6,256,1,1)] -> ReLUU-9_out_tensor[Float(-6,2048,1,1)]
Layer(CaskConvolution): MMD-9 + constant256_35 + (Unnamed Layer* 340) [Shuffle] + unsqueeze_node_after_constant256_35 + (Unnamed Layer* 340) [Shuffle]_(Unnamed Layer* 340) [Shuffle]_output + AddD-9 + ReLUD-9, Tactic: -7067026478815706014, ReLUU-9_out_tensor[Float(-6,2048,1,1)] -> ReLUD-9_out_tensor[Float(-6,256,1,1)]
Layer(NoOp): Reformatting CopyNode for Input Tensor 0 to squeeze_after_ReLUD-9, Tactic: 0, ReLUD-9_out_tensor[Float(-6,256,1,1)] -> Reformatted Input Tensor 0 to squeeze_after_ReLUD-9[Float(-6,256,1,1)]
Layer(NoOp): squeeze_after_ReLUD-9, Tactic: 0, Reformatted Input Tensor 0 to squeeze_after_ReLUD-9[Float(-6,256,1,1)] -> squeeze_after_ReLUD-9_out_tensor[Float(-6,256)]
Layer(Reduce): Reduce, Tactic: 0, squeeze_after_ReLUD-9_out_tensor[Float(-6,256)] -> tensor8[Float(-6)]
Layer(NoOp): myReshapeN-output, Tactic: 0, tensor8[Float(-6)] -> reshapeV-output[Float(-3,-4,1)]
```

+ trtexec Result of performance test
```
[I] === Performance summary ===
[I] Throughput: 218.85 qps
[I] Latency: min = 4.38843 ms, max = 4.74341 ms, mean = 4.55748 ms, median = 4.53888 ms, percentile(99%) = 4.73462 ms
[I] End-to-End Host Latency: min = 4.38843 ms, max = 4.74341 ms, mean = 4.55748 ms, median = 4.53888 ms, percentile(99%) = 4.73462 ms
[I] Enqueue Time: min = 0.00830078 ms, max = 0.0300293 ms, mean = 0.0111543 ms, median = 0.010498 ms, percentile(99%) = 0.0217285 ms
[I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] GPU Compute Time: min = 4.38843 ms, max = 4.74341 ms, mean = 4.55748 ms, median = 4.53888 ms, percentile(99%) = 4.73462 ms
[I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(99%) = 0 ms
[I] Total Host Walltime: 3.00607 s
[I] Total GPU Compute Time: 2.98971 s
```
