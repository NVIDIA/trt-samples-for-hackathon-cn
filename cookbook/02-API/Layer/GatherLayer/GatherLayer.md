# Gather Layer

+ Simple example
+ axis
+ mode（since TensorRT 8.2）
  + Gather DEFAULT mode
  + Gather ELEMENT mode
  + Gather ND mode & num_elementwise_dims
+ Another example of GatherND mode
+ add_gather_v2 (since TensorRT 8.5)

---

## Simple example

+ Refer to SimpleExample.py
+ Gather the elements of data tensor (inptu tensor 0) in the second high dimension according to index tensor (input tensor 1).

---

## axis

+ Refer to Axis.py
+ Set the axis of the gather operation takes.

+ setting **axis=0**, gahter on the highest dimension.

+ setting **axis=1**, gahter on the second highest dimension.

+ setting **axis=2**, gahter on the thrid highest dimension.

+ setting **axis=3**, gahter on the fourth highest dimension.

---

## mode（since TensorRT 8.2）

### DEFAULT mode

+ Refer to ModeDefault.py
+ Use default mode to gather elements.

+ setting mode=trt.GatherMode.DEFAUT, axis=2.

+ Computation process:
  + Refer to [Onnx Gather Operation](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gather)
  + Data tensor $\bold{data}[d_{0},d_{1},...,d_{r-1}] (dim=r)$, index tensor $\bold{index}[a_{0},a_{1},...,a_{q-1}] (dim=q)$, setting $axis=p (0 \le p < r)$.
  + Output tensor $\bold{output}[d_{0},d_{1},...,d_{p-1},a_{0},a_{1},...,a_{q-1},d_{p+1},d_{p+2},...,d_{r-1}], (dim=r+q-1)$, exception: when $p=0$, the length of the highest dimension is $a_{0}$.
  + Notice that the dimension of $d_{p}$ in the $\bold{output}$ is replaced into the dimensions of $\bold{index}$ ($a_{0},a_{1},...,a_{q-1}$). For the each element $\bold{e}$ in the $\bold{index}$, the $\bold{e}$th element in the $\bold{input}$ is extracted.

  + Using grammar in numpy, define $i_{j}$, s.t. $0 \le i_{j} < a_{j}$, the process of computation can be translated as (the $i_{*}$ on the left side of the equals sign and the $\bold{index}[...]$ on the right side of the equal sign both locate at the dimension of $d_{p}$).
$$
\begin{aligned}
\bold{output}[:,:,...,:,i_{0},i_{1},...,i_{q-1},:,:,...,:] = \bold{data}[:,:,...,:,\bold{index}[i_{0},i_{1},...,i_{q-1}],:,:,...,:]
\end{aligned}
$$
+ Specifically for the example code: $\bold{output}[:,:,i_{0},i_{1},:] = \bold{inputT0}[:,:,\bold{index}[$i_{0}$,$i_{1}$],:], s.t. 0 \le i_{0} < 3, 0 \le i_{1} < 2$.

### ELEMENT mode

+ Refer to ModeElement.py
+ Use Element mode to gather elements.

+ Setting mode=trt.GatherMode.ELEMENT, axis=2.

+ Computation process:
  + Refer to [Onnx GatherElements Operation](https://github.com/onnx/onnx/blob/master/docs/Operators.md#GatherElements)
  + The shape of data tensor, index tensor and output tensor are the same, $\bold{data}[d_{0},d_{1},...,d_{r-1}], \bold{index}[d_{0},d_{1},...,d_{r-1}], \bold{output}[d_{0},d_{1},...,d_{r-1}] (dim=r)$, setting $axis=p$ ($0 \le p < r$).
  + Using grammar in numpy, define $i_{j}$, s.t. $ 0 \le i_{j} < d_{j}$, the process of computation can be translated as (the $i$ on the left side of the equals sign and the $\bold{index}[...]$ on the right side of the equal sign both locate at the dimension of $d_{p}$).
$$
\begin{aligned}
\bold{output}[i_{0},i_{1},...,i_{p-1},i_{p},i_{p+1},...,i_{r-1}] = \bold{data}[i_{0},i_{1},...,i_{p-1},\bold{index}[i_{0},i_{1},...,i_{p-1},i_{p},i_{p+1},...,i_{r-1}],i_{p+1},...,i_{r-1}]
\end{aligned}
$$
  + Specifically for the example code: $\bold{output}[:,:,i,:] = \bold{inputT0}[:,:,\bold{index}[:,:,i,:],:], s.t. 0 \le i < 4$.

### Gather ND mode & num_elementwise_dims

+ Refer to ModeND.py
+ Use ND mode to gather elements.

+ Setting mode=trt.GatherMode.ND without setting parameter num_elementwise_dims (default value: 0).

+ 指定 mode=trt.GatherMode.ND，指定 num_elementwise_dims=1，输入张量 0 与初始范例代码相同，输入张量 1 形状 (1,2,3), shape of output tensor 0: (1,2)。两个输入张量的最高 1 维必须相同，索引张量从次高维开始在数据张量中查找
$$
输入张量 1 =
\left[\begin{matrix}
    \left[\begin{matrix}
        0 & 1 &  2 \\
        0 & 2 & -1
    \end{matrix}\right] \\
\end{matrix}\right] \\
输出张量 =
\left[\begin{matrix}
    1.  & 24.
\end{matrix}\right]
$$

+ 指定 mode=trt.GatherMode.ND，指定 num_elementwise_dims=2，输入张量 0 与初始范例代码相同，输入张量 1 形状 (1,3,2), shape of output tensor 0: (1,3)。两个输入张量的最高 2 维必须相同，索引张量从季高维开始在数据张量中查找
$$
输入张量 1 =
\left[\begin{matrix}
    \left[\begin{matrix}
        2 & 1 \\
        3 & 0 \\
        1 & 2
    \end{matrix}\right]
\end{matrix}\right] \\
输出张量 =
\left[\begin{matrix}
    21. & 130. & 212.
\end{matrix}\right]
$$

+ 指定 mode=trt.GatherMode.ND，指定 num_elementwise_dims=3，输入张量 0 与初始范例代码相同，输入张量 1 形状 (1,3,4,1), shape of output tensor 0: (1,3,4)。两个输入张量的最高 3 维必须相同，索引张量从叔高维开始在数据张量中查找
$$
输入张量 1 =
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            0 \\ 0 \\ 0 \\ 0
        \end{matrix}\right]
        \left[\begin{matrix}
            2 \\ 2 \\ 2 \\ 2
        \end{matrix}\right]
        \left[\begin{matrix}
            1 \\ 1 \\ 1 \\ 1
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right] \\
输出张量 =
\left[\begin{matrix}
    \left[\begin{matrix}
          0. &  10. &  20. &  30. \\
        102. & 112. & 122. & 132. \\
        201. & 211. & 221. & 231.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 指定 mode=trt.GatherMode.ND，指定 num_elementwise_dims=2，输入张量 0 与初始范例代码相同，输入张量 1 形状 (1,3,1), shape of output tensor 0: (1,3,5)
$$
输入张量 1 =
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            2
        \end{matrix}\right]
        \left[\begin{matrix}
            3
        \end{matrix}\right]
        \left[\begin{matrix}
            1
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right] \\
输出张量 =
\left[\begin{matrix}
    \left[\begin{matrix}
         20. &  21. &  22. &  23. &  24. \\
        130. & 131. & 132. & 133. & 134. \\
        210. & 211. & 212. & 213. & 214. 
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 含义：参考 [Onnx GatherND 算子](https://github.com/onnx/onnx/blob/master/docs/Operators.md#GatherND) 和 TensorRT 说明　[link](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_gather_layer.html)
  + 数据张量形状 $data[d_{0},d_{1},...,d_{r-1}]$（$dim=r$），索引张量形状 $index[a_{0},a_{1},...,a_{q-1}]$（$dim=q$），指定 $nElementwiseDim=p$（要求 $a_{q-1}$ 为构建期常量），则
  + 输出张量维度 $dim(output) = q + r - index.shape[-1] - 1 - nElementwiseDim$（以下 $nElementwiseDim$ 简记为 $nB$，它的含义是“被当成 Batch 维的维度数”）
  + 要求 $nB < min(r,q)$，否则报错。即要求跳过的维数不能超过 $data$ 和 $index$ 维数中的较小者
  + 要求 $data.shape[:nB] = index.shape[:nB]$，否则报错。即要求 $data$ 和 $index$ 的形状的前 $nB$ 维尺寸都相等（都当做 $Batch$ 维）
  + 要求 $index.shape[-1] \le r - nB$，否则报错。即要求“$index$ 跳过 $nB$ 维后的剩余维度数”（真实索引维数）不能超出“$data$ 跳过 $nB$ 维后的剩余维度数”
  + 对于 $index$ 中第 j 维的索引 $i_{j}$（$0 \le j < q$） 要求 $-d_{j} \le index[:,:,...,i_{j},:,:,...,:] \le d_{j}-1$，即可以使用负的索引号
  + 命 $N = a_{0}*a_{1}*...*a_{nB-1}$，即 $data$ 和 $index$ 的所有 $Batch$ 维元素数
  + （onnx 文档的解释）If indices_shape[-1] == r-b, since the rank of indices is q, indices can be thought of as N (q-b-1)-dimensional tensors containing 1-D tensors of dimension r-b, where N is an integer equals to the product of 1 and all the elements in the batch dimensions of the indices_shape. Let us think of each such r-b ranked tensor as indices_slice. Each scalar value corresponding to data[0:b-1,indices_slice] is filled into the corresponding location of the (q-b-1)-dimensional tensor to form the output tensor
  + （onnx 文档的解释）If indices_shape[-1] < r-b, since the rank of indices is q, indices can be thought of as N (q-b-1)-dimensional tensor containing 1-D tensors of dimension < r-b. Let us think of each such tensors as indices_slice. Each tensor slice corresponding to data[0:b-1, indices_slice , :] is filled into the corresponding location of the (q-b-1)-dimensional tensor to form the output tensor

+ 计算公式：
$$
output[i_{0},i_{1},...,i_{nB-1},i_{nB},i_{nB+1},...,i_{q-2}] \\
=data[i_{0},i_{1},...,i_{nB-1},index[i_{0},i_{1},...,i_{nB-1},i_{nB},i_{nB+1},...,i_{q-2}]]
$$
+ 式子中下标当 $0 \le j < nB$ 时 $0 \le i_{j} < d_{j}$，当 $ nB \le j < q - 1$ 时 $0 \le i_{j} < a_{j}$
+ 式子中 $output$ 索引的前 $nB$ 项来自公共 $Batch$ 部分（一共 $nB$ 个），以后索引来自 $index$ 跳过 $Batch$ 维后的部分（一共 $q-2-(nB-1)$ 个），两部分总共 $q-1$ 项
+ 式子中 $index$ 只索引了前 $q-1$ 维，$index[...]$ 实际上是个 $a_{q-1}$ 维的张量
+ 若 $a_{q-1} = r - nB$（即真实索引维数等于 data 剩余维度数）:
  + 此时由公式计算的输出张量维度 $\mathbf{dim}(output) = q + r - a_{q-1} - 1 - nB = q + r - (r - nB) - 1 - nB = q - 1$
  + 由于此时 $a_{q-1} = r - nB$，所以 $data[...]$ 中总索引深度为 $nB + a_{q-1} = nB + (r - nB) = r$，恰好定位到 $data$ 的一个元素上
  + 对于上面的范例代码（考虑 num_elementwise_dims=1 那一组）：
        + $r = 4, q = 3, nB = 1，a_{q-1} = 3 = r - nB$，首维 1 被当成 $Batch$ 维，$\mathbf{dim}(output) = q - 1 = 2$
        + output[:,i] = data[:,index[:,i]] = [12,24]，其中
        + index[0,0] = [0,1,2]，所以 output[0,0] = data[0,0,1,2] = 12
        + index[0,1] = [0,2,-1]，所以 output[0,1] = data[0,0,2,-1] = 24

+ 若 $a_{q-1} < r - nB$（即真实索引维数小于 data 剩余维度数），记 $nD = r - nB - a_{q-1}$
  + 此时由公式计算的输出张量维度 $\mathbf{dim}(output) = q + r - a_{q-1} - 1 - nB = q + r - (r - nB - nD) - 1 - nB = q - 1 + nD$
  + 由于此时 $a_{q-1} = r - nB - nD$，所以 $data[...]$ 中总索引深度为 $nB + a_{q-1} = nB + (r - nB - nD) = r - nD$，每个索引将会定位到 $data$ 末尾 nD 维子张量上
  + 对于上面的范例代码（考虑 num_elementwise_dims=2 的后一组）：
        + $r = 4, q = 3, nB = 2，a_{q-1} = 1 < r - nB，nD = r - nB - a_{q-1} = 1$，首两维被当成 $Batch$ 维，$\mathbf{dim}(output) = q - 1 + nD = 3$
        + output[:,i] = data[:,i,index[:,i]]，其中
        + index[0,0] = [2]，所以 output[0,0] = data[0,0,2] = [20,21,22,2324]
        + index[0,1] = [3]，所以 output[0,1] = data[0,1,3] = [130,131,132,133,134]
        + index[0,2] = [1]，所以 output[0,2] = data[0,2,1] = [210,211,212,213,214]

---

+ 不满足 index.shape[-1] $\le$ r - n*ElementwiseDim 时报错（报错信息的“ -” 写成了 “+”）：

```txt
[TRT] [E] 1: [gatherNode.cpp::computeGatherNDOutputExtents::110] Error Code 1: Internal Error (invalid dimension in GatherND indices[-1] > rank(data) + nbElementWiseDims)
```

+ 不满足 nB < min(q,r) 时报错：

```txt
[TRT] [E] 3: (Unnamed Layer* 0) [Gather]: nbElementWiseDims must between 0 and rank(data)-1 inclusive for GatherMode::kND
```

+ 不满足 data.shape[:nB] == index.shape[:nB] 时报错：

```txt
[TRT] [E] 4: [graphShapeAnalyzer.cpp::processCheck::581] Error Code 4: Internal Error ((Unnamed Layer* 0) [Gather]: dimensions not compatible for Gather with GatherMode = kND)
```

---

## Another example of GatherND mode

+ Refer to AnotherExampleOfGatherND.py
---

## add_gather_v2 (since TensorRT 8.5)

+ A new API for adding Gather Layer similar to add_gather, which the parameter **mode** enters constructor, and **axis** leaves constructor.