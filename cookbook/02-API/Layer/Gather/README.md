# Gather layer

+ Gather layer.

+ Steps to run.

```bash
python3 main.py
```

+ Attributes

| Name | Description | Default | Range |
| :--: | :---------- | :-----: | :---: |
| axis | The axis to gather elements from. | 0 | 0 ≤ axis < rank(input) |
| mode | Specifies gather behavior (DEFAULT, ELEMENT, ND). DEFAULT is similar to ONNX Gather; ELEMENT is similar to ONNX GatherElements; ND is similar to ONNX GatherND. | DEFAULT | — |
| num_elementwise_dims | The number of leading dimensions of indices tensor to be handled elementwise. | 0 | 0 ≤ num_elementwise_dims ≤ 1 |

+ Available values of `trt.GatherMode`.

|  Name   |                                    Comment                                     |
| :-----: | :---------------------------------------------------------------------------: |
| DEFAULT | Similar to ONNX `Gather`: replace the `axis` dimension of data with the index shape. |
| ELEMENT | Similar to ONNX `GatherElements`: data, index and output share the same shape.  |
|   ND    | Similar to ONNX `GatherND`: the last dimension of index addresses into data.    |

+ Input / output data types and shapes:
  + Input data tensor `T1` in [bool, int4, int8, int32, int64, float8, float16, float32, bfloat16]; output tensor is also `T1`.
  + Index tensor `T2` in [int32, int64]; every element `I_j` must satisfy `0 <= I_j < input_dimensions[axis]` (negative indices allowed).

## Default mode

+ Computation process:
  + Refer to [Onnx Gather operator](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gather)
  + Input data tensor: $\bold{data}[d_{0},d_{1},...,d_{r-1}] \quad (r \ge 1)$
  + Input index tensor: $\bold{index}[a_{0},a_{1},...,a_{q-1}] \quad (q \ge 1)$
  + Gather axis: $\bold{p} \quad (0 \le p < r)$
  + Number of elementwise dimension: $\bold{nED} \quad (0 \le nED \le 1)$
  + Output tensor with nED == 0: $\bold{output}[d_{0},d_{1},...,d_{p-1},a_{0},a_{1},...,a_{q-1},d_{p+1},d_{p+2},...,d_{r-1}] \quad (dim=r+q-1)$
  + Output tensor with nED > 0: $\bold{output}[d_{0},d_{1},...,d_{p-1},a_{nED},a_{nED+1},...,a_{q-1},d_{p+1},d_{p+2},...,d_{r-1}] \quad (dim=r+q-1-nED)$
  + When $p=0$, the length of the highest dimension is $a_{0}$.
  + The dimension of $d_{p}$ in the $\bold{output}$ is replaced with the dimensions of $\bold{index}$ ($a_{0},a_{1},...,a_{q-1}$).
  + For the each element $\bold{e}$ in the $\bold{index}$, the $\bold{e}$ th element at the $\bold{input}$ 's $p$ th dimension is extracted.
  + Stating with syntax in numpy ($i_{j}$, s.t. $0 \le i_{j} < a_{j}$, locate at the dimension of $d_{p}$):

$$
\begin{aligned}
\bold{output}[:,:,...,:,i_{nED},i_{nED+1},...,i_{q-1},:,:,...,:] = \bold{data}[:,:,...,:,\bold{index}[i_{nED},i_{nED+1},...,i_{q-1}],:,:,...,:]
\end{aligned}
$$

+ For the example code:

$$
\bold{output}\left[:,:,i_{0},i_{1},:\right] = \bold{tensor}\left[:,:,\bold{tensor1}\left[i_{0},i_{1}\right],:\right], \quad s.t. 0 \le i_{0} < 3, 0 \le i_{1} < 2
$$

+ Case Default mode with num_elementwise_axis == 1

## Element mode

+ Computation process:
  + Refer to [Onnx GatherElements operator](https://github.com/onnx/onnx/blob/master/docs/Operators.md#GatherElements)
  + The shape of input data tensor, input index tensor and output tensor are the same as $[d_{0},d_{1},...,d_{r-1}] \quad (dim=r \ge 1)$
  + Stating with syntax in numpy ($i_{j}$, s.t. $ 0 \le i_{j} < d_{j}$, locate at the dimension of $d_{p}$).

$$
\begin{aligned}
\bold{output}[i_{0},i_{1},...,i_{p-1},i_{p},i_{p+1},...,i_{r-1}] = \bold{data}[i_{0},i_{1},...,i_{p-1},\bold{index}[i_{0},i_{1},...,i_{p-1},i_{p},i_{p+1},...,i_{r-1}],i_{p+1},...,i_{r-1}]
\end{aligned}
$$

+ For the example code:

$$
\bold{output}[:,:,i,:] = \bold{tensor}[:,:,\bold{tensor1}[:,:,i,:],:], \quad s.t. 0 \le i < 4
$$

## ND mode

+ Computation process:
  + Refer to [Onnx GatherND operator](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND), and [TensorRT Gather operator](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_gather_layer.html)
  + Input data tensor: $\bold{data}[d_{0},d_{1},...,d_{r-1}] \quad (r \ge 1)$
  + Input index tensor: $\bold{index}[a_{0},a_{1},...,a_{q-1}] \quad (q \ge 1)$
  + $a_{q-1}$ must be a build-time constant.
  + Number of elementwise dimension: $\bold{nED} \quad (0 \le nED \le 1)$
  + Shape of output tensor: $dim(output) = q + r - a_{q-1} - 1 - nED$
  + $nED < min(r,q)$, i.e., elementwise dimension must be less than the smaller dimension of $data$ and $index$.
  + $data.shape[:nED] = index.shape[:nED]$, i.e., the first $nED$ dimensions of $data$ and $index$ must be the same.
  + $a_{q-1} \le r - nED$, i.e., the rank of real index must be less or equal to remain $data$ skiiping $nED$
  + $-d_{j} \le index[:,:,...,i_{j},:,:,...,:] \le d_{j}-1$ for $j$ th value in $index$, i.e., negetive index value can be used.
  + Stating with syntax in numpy:
$$
output[i_{0},i_{1},...,i_{nED-1},i_{nED},i_{nED+1},...,i_{q-2}] \\
=data[i_{0},i_{1},...,i_{nED-1},index[i_{0},i_{1},...,i_{nED-1},i_{nED},i_{nED+1},...,i_{q-2}]]
$$

+ If $a_{q-1} = r - nED$,
  + Rank of output: $q + r - a_{q-1} - 1 - nED = q - 1$
  + Real rank of index: $nED + a_{q-1} = r$, which each index locates the exact one element in $data$.

+ If $a_{q-1} < r - nED$ (asumming $a_{q-1} = r - nED - nD$),
  + Rank of output: $q + r - a_{q-1} - 1 - nED = q - 1 + nD$
  + Real rank of index: $nED + a_{q-1} = r - nD$, which each index locates a tenosr of rank $nD$ in $data$.

## ND mode with num_elementwise_axis == 1
