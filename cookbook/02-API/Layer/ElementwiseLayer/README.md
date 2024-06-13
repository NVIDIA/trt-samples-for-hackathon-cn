# Elementwise Layer

## Case Simple

+ Compute elementewise addition.

## Case Op

+ Adjust the operator of the computation after adding the layer.

## Case Broadcast

+ Broadcast the elements while elementwise operation. It works when:
  + The ranks of the two input tensors are same: len(tensor0.shape) == len(tensor1.shape).
  + For each dimension of the two input tensors, either the lengths of this dimension are same, or at least one tensor has length of 1 at this dimension.
