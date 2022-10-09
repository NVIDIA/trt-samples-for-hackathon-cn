#

## Introduction

+ We test the code in two scenarios.

  1. Conv -> Conv -> Unsqueeze -> Add -> Squeeze -> ReLU -> ... -> Conv ->Transpose -> Add -> Transpose -> ReLU -> ... -> Conv.
  2. All Squeeze / Unsqueeze / Transpose layers in scenario 1 are removed.

## Result

+ all Conv+Add+ReLU can be merged into a kernel by TensorRT


+ Nearly doubled performance after optimization
