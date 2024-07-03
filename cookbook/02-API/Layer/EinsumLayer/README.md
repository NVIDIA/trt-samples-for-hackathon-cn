# Einsum Layer

+ Steps to run.

```bash
python3 main.py
```

## Case Contraction

+ Single contraction with two tensor: $A_{1 \times 3 \times 4} \times B_{2 \times 3 \times 5} \rightarrow C_{1 \times 4 \times 2 \times 5}$

## Case Equation

+ Adjust the computation expression after adding the layer.

## Case Transpose

+ Transpose tensor: $A_{1 \times 3 \times 4} \rightarrow B_{3 \times 4 \times 1}$

## Case Sum-Reduce

+ Compute reduce sum: $A_{1 \times 3 \times 4} \rightarrow B_{1 \times 3}$
+ The disappeared character on the right side of "->", k, is the axis of sum-reduce operation.
+ More than one axes can be set to do sum-reduce in one layer.

## Case Dot Product

+ Compute tensor dot product: $A_{1 \times 1 \times 4} \times B_{1 \times 1 \times 4} \rightarrow c$
+ Computation process: the same character on the left side of "->", k, is the axis of contraction operation, and the disappeared character on the right side of "->", i, j, p and q, is the axis of sum-reduce operation.

$$
0 * 0 + 1 * 1 + 2 * 2 + 3 * 3 = 6
$$

+ Substitutive example 1: change the shape of input tensors as (1,2,4) / (1,3,4). So there are 2 * 3 =  6 groups join the sum-reduce operation.

$$
6 + 6 + 6 + 22 + 22 + 22 = 84
$$

+ Substitutive example 2: change the shape of input tensors as (1,2,4) / (1,3,4) and keep "j" in the equation. So the axis "j" was exculed from sum-reduce operation.

$$
6 + 6 + 6 = 18，22 + 22 + 22 = 66
$$

## Case Matrix Multiplication

+ Compute matrix multiplication: $A_{2 \times 2 \times 3} \times B_{2 \times 3 \times 4} \rightarrow C_{2 \times 2 \times 4}$
+ Similar to batched matric multiplication.

## Multi-Tensor contraction (not support)

+ Do contraction on three or more tensors. $A_{1 \times 2 \times 3} \times B_{4 \times 3 \times 2} \times C_{4 \times 5} \rightarrow D_{1 \times 5}$
+ Error information:

```txt
[TRT] [E] 3: [einsumLayer.cpp::EinsumLayer::23] Error Code 3: API Usage Error (Parameter check failed at: optimizer/api/layers/einsumLayer.cpp::EinsumLayer::23, condition: nbInputs > 0 && nbInputs <= kMAX_EINSUM_NB_INPUTS : einsum does not support more than two inputs.)
```

## Case Diagonal (not support)

+ Take diagonal elements from a tensor: $A_{1 \times 4 \times 4} \rightarrow B_{1 \times 4}$
+ Error informarion:

```txt
[TRT] [E] 3: [einsumLayer.cpp::validateEquation::85] Error Code 3: Internal Error ((Unnamed Layer* 0) [Einsum]: Diagonal operations are not permitted in Einsum equation)
```

## Case Ellipsis (not support)

+ Use ellipsis in computation equation: $A_{1 \times 3 \times 4} \rightarrow A_{1 \times 3 \times 4}$
+ Error information:

```txt
[TRT] [E] 3: [einsumLayer.cpp::validateEquation::63] Error Code 3: Internal Error ((Unnamed Layer* 0) [Einsum]: ellipsis is not permitted in Einsum equation)
[TRT] [E] 4: [einsumNode.cpp::computeOutputExtents::126] Error Code 4: Internal Error ((Unnamed Layer* 0) [Einsum]: number of subscripts for input operand 0 in the equation is not equal to the rank of corresponding input tensor)
[TRT] [E] 4: [graphShapeAnalyzer.cpp::needTypeAndDimensions::2276] Error Code 4: Internal Error ((Unnamed Layer* 0) [Einsum]: output shape can not be computed)
```
