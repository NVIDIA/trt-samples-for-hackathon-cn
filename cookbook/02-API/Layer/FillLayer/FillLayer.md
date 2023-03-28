# Fill Layer

+ Simple example
+ Linear fill with setting shape in buildtime and range in runtime
+ Linear fill with setting shape and range in runtime
+ Random seed (not support)

---

## Simple example

+ Refer to SimpleExampleLinear.py and SimpleExampleNormalRandom.py and SimpleExampleUniformRandom.py and result-SimpleExampleLinear.py and SimpleExampleNormalRandom.py
+ Produce a constant tensor in buildtime with different type of filling.

+ Available fill operation
| trt.FillOperation | Input tensor 0 | Input tensor 1 / default value |     Input tensor 2 / default value     |
| :---------------: | :------------: | :----------------------------: | :------------------------------------: |
|     LINSPACE      |  Shape tensor  |  Scalar tensor (Start) / None  |      Scalar tensor (Delta) / None      |
|   RANDOM_NORMAL   |  Shape tensor  |    Scalar tensor (Mean) / 0    | Scalar tensor (standard deviation) / 1 |
|  RANDOM_UNIFORM   |  Shape tensor  |  Scalar tensor (Minimum) / 0   |       Scalar tensor (Maimum) / 1       |

+ Eror information when using linear fill without giving start and delta tensor.

```txt
[TensorRT] ERROR: 2: [fillRunner.cpp::executeLinSpace::46] Error Code 2: Internal Error (Assertion dims.nbDims == 1 failed.Alpha and beta tensor should be set when output an ND tensor)
[TensorRT] INTERNAL ERROR: Assertion failed: dims.nbDims == 1 && "Alpha and beta tensor should be set when output an ND tensor"
```

+ The random seed of random fill is static and solid, meaning the value among multiple times of engine building is the same.

---

## Linear fill with setting shape in buildtime and range in runtime

+ Refer to LinearFill+BuildtimeShape+RuntimeRange.py
+ Produce a constant tensor with linear filling, the shape of output tensor is determined in buildtime, but the value range is determined in runtime.

---

## Linear fill with setting shape and range in runtime

+ Refer to LinearFill+RuntimeShapeRange.py
+ Produce a constant tensor with linear filling, the shape and value range of output tensor is determined in runtime.

+ This example shows the usage of input shape tensor in TensorRT8.5, which is much different from that in the older versions.

---

## Random seed (not support)
