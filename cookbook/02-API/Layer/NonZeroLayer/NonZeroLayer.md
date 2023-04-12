# Non Zero Layer

+ Simple example

---

## Simple example

+ Refer to SimpleExample.py

+ Notice that before the first inference, **context.get_tensor_shape(lTensorName[1])** returns shape including -1 though the shape of input tensor is set.

+ Use data-dependent API network with Output Allocator, refer to 09-Advance/OutputAllocator