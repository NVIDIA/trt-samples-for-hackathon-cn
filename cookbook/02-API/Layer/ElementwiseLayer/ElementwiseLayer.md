# Elementwise Layer

+ Simple example
+ op
+ Broadcast

---

## Simple example

+ Refer to SimpleExample.py
+ Compute elementewise addition on the two input tensor.

+ Shape of output tensor 0: (1,3,4,5), the two tensor do addition element ny element.

---

## op

+ Refer to Op.py
+ Adjust the operator of the computation after constructor.

+ available elementwise operation
| trt.ElementWiseOperation |  $f\left(a,b\right)$   |           Comment            |
| :----------------------: | :--------------------: | :--------------------------: |
|           SUM            |         a + b          |                              |
|           PROD           |         a * b          |                              |
|           MAX            | $\max\left(a,b\right)$ |                              |
|           MIN            | $\min\left(a,b\right)$ |                              |
|           SUB            |         a - b          |                              |
|           DIV            |         a / b          |                              |
|           POW            |   a \*\* b ($a^{b}$)   | Input can not be INT32 type  |
|        FLOOR_DIV         |         a // b         |                              |
|           AND            |        a and b         | Input / Output are Bool type |
|            OR            |         a or b         | Input / Output are Bool type |
|           XOR            |    a ^ b (a xor b)     | Input / Output are Bool type |
|          EQUAL           |         a == b         |     Output is Bool type      |
|         GREATER          |         a > b          |     Output is Bool type      |
|           LESS           |         a < b          |     Output is Bool type      |

+ Error information of using other data type while BOOL type is needed as input:

```txt
[TensorRT] ERROR: 4: [layers.cpp::validate::2304] Error Code 4: Internal Error ((Unnamed Layer* 0) [ElementWise]: operation AND requires boolean inputs.)
```

+ Error information of using non-activation tensor type (datstype beside float32/float16) as input:

```txt
# Both base and exponent are INT32
[TensorRT] ERROR: 4: [layers.cpp::validate::2322] Error Code 4: Internal Error ((Unnamed Layer* 0) [ElementWise]: operation POW requires inputs with activation type.)
# Bse is INT32, exponent is float32
[TensorRT] ERROR: 4: [layers.cpp::validate::2291] Error Code 4: Internal Error ((Unnamed Layer* 0) [ElementWise]: operation POW has incompatible input types Float and Int32)
```

## Broadcast

+ Refer to Broadcast.py
+ Broadcast the element while elementwise operation.

+ Boardcast works when:
  + The dimension of the two input tensors are same: len(tensor0.shape) == len(tensor1.shape).
  + For each dimension of the two input tensors, either the length of this dimension are same, or there is at least one "1" of the length of this dimension.
