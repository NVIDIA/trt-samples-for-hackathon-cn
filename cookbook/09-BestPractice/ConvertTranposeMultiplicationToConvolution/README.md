#

## Introduction

## Result

+ Example code tests the effect on GTX 1070 and A30

+ On GTX1070, small BatchSize has better acceleration effect

+ On the A30, the generated network structure is different from the GTX1070. There is no significant acceleration on the small BatchSize, but it shows a significant acceleration effect after increasing the BatchSize

+ The performance of two-dimensional matrix multiplication is better than that of three-dimensional matrix multiplication

+ Thanks to the "grand" students of TensorRT Hackathon 2022 for providing ideas
