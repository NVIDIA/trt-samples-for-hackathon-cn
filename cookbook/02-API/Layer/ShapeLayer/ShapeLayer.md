# Shape 层
+ Simple example（Shape + Static Shape）
+ Shape + Dynamic Shape
+ Shape Tensor 用法 [TODO]

---
## Simple example
+ Refer to SimpleExample.py
+ Shape of input tensor 0: (1,3,4,5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1.
        \end{matrix}\right]
        \left[\begin{matrix}
            1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1.
        \end{matrix}\right]
        \left[\begin{matrix}
            1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output tensor 0: (4,)
$$
\left[\begin{matrix}
  1 & 3 & 4 & 5 \\
\end{matrix}\right]
$$

+ 必须使用 explicit batch 模式，否则报错
```
[TRT] [E] 3: [network.cpp::addShape::1163] Error Code 3: API Usage Error (Parameter check failed at: optimizer/api/network.cpp::addShape::1163, condition: !hasImplicitBatchDimension()
)
```

---

## Shape + Dynamic Shape
+ DynamicShape.py，在 Dynamic shape 模式下使用 Shape 层。此时输入张量的维度确定、形状不确定、值无关（求得 Shape 的维度确定、形状确定、值不确定）

+ Shape of output tensor 0: (4,)，结果与初始范例代码相同

---

## Shape Tensor 用法 [TODO]
+ 注意，TensorRT 7 起两种方法均可使用，TensorRT 6 中只能使用 Shape Tensor 专用方法，若在 TensorRT 6 中使用常规的 execution tensor 方法，则会得到如下报错：

> [TensorRT] ERROR: (Unnamed Layer* 1) [Shape]: ShapeLayer output tensor ((Unnamed Layer* 1) [Shape]_output) used as an execution tensor,  but must be used only as shape tensor.
build engine failed.
