# Non Zero Layer

+ Common
+ Simple example

---

## Common

+ Input tensor
  + T0

+ Output tensor
  + T1

+ Data type
  + T0: float32, float16, int32 ,int8, bool
  + T1: int32

+ Shape
  + T1.shape: [n, m], where n = len(T0.shape), m = Count of 0 in T0

+ Attribution and default value
  + alpha = 0, parameter for some kinds of activation function
  + beta = 0, parameter for some kinds of activation function
  + type, avialable activation function, shown as table below

---

## Simple example

+ Refer to SimpleExample.py

+ Notice that before the first inference, **context.get_tensor_shape(lTensorName[1])** returns shape including -1 though the shape of input tensor is set.

+ Use data-dependent API network with Output Allocator, refer to 02-API/OutputAllocator