#

## Introduction

+ We test the code in two scenarios.

  1. `[32,256,1] -> [32,256,256] -> [32,256,2048] -> [32,256,256] -> [32,256,2048] -> ... -> [32,256,256] -> [32,256,1]`
  2. Add two Reshape layers at the beginning and the end of scenario 1 to merge the first two dimensions of the input tensor at the beginning, and restore the first two dimensions of the output tensor at the end. `[32,256,1] -> [32*256,256] -> [32*256,2048] -> [32*256,256] -> [32*256,2048] -> ... -> [32*256,256] -> [32,256,1]`

## Result

+ The performance of two-dimensional matrix multiplication is better than that of three-dimensional matrix multiplication

