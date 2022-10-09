#

## Introduction

+ We test the code in two scenarios.

  1. Using the source code algorithm, the input tensor in the runtime is sliced and then involved in 12 matrix multiplication, and then transposed
  2. Complete matrix multiplication and transposition in advance in the calculation diagram, and input tensor during the operation period to slice the results

## Result

+ 4 times the original performance after optimization

+ Shape input is used in TensorRT (because the shape of the output tensor depends on the value of the input tensor), which is different from the dynamic shape of other examples
