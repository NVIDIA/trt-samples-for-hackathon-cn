# NonZero Layer

+ Output Allocator MUST be used for Data-Dependent-Shape mode like this example.

+ In TensorRT-8.6 ~ TensorRT-9.X, `context.get_tensor_shape()` returns real output shape after the enqueue in DDS mode, so we can use it to knowreal output shape. But this API always return shape with -1 in DDS mode since TensorRT-10, so we must use Output Allocator to get the real output shape
