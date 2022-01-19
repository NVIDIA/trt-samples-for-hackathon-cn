# 目录
+ <a href="#Activation 层">Activation 层</a>（**add_activation**）
  - type & alpha & beta
+ <a href="#Concatenation 层">Concatenation 层</a>（**add_concatenation**）
  - axis
+ <a href="#Constant 层">Constant 层</a>（**add_constant**）
  - weight & shape
+ <a href="#Convolution 层">Convolution 层</a>（**add_convolution**）
  - kernel_size & num_output_maps & kernel & bias
  - stride
  - padding
  - pre_padding
  - post_padding
  - padding_mode
  - dilation
  - num_groups
  - kernel_size_nd & stride_nd & padding_nd & dilation_nd（**add_convolution_nd**，三维卷积）
+ <a href="#Deconvolution 层">Deconvolution 层</a>（**add_deconvolution**）topk
  - kernel_size & num_output_maps & kernel & bias
  - stride
  - padding
  - pre_padding
  - post_padding
  - padding_mode
  - num_groups
  - kernel_size_nd & stride_nd & padding_nd（**add_deconvolution_nd**，三维反卷积）
+ <a href="#Element Wise 层">Element Wise 层</a>（**add_elementwise**）
  - op
+ <a href="#Fill 层">Fill 层</a>（**add_fill**）
  - 线性填充与 set_input
  - 均匀随机填充与 set_input
+ <a href="#Fully Connected 层">Fully Connected 层</a>（**add_fully_connected**）
  - num_output_channels & kernel& bias
+ <a href="#Gather 层">Gather 层</a>（**add_gather**）
  - axis
  - num_elementwise_dims
+ <a href="#Identity 层">Identity 层</a>（**add_identity**）
+ <a href="#Loop 结构">Loop 结构</a>（**add_loop**）
  - for 型循环，两种输出
  - while 型循环，两种输出
  - while 型循环 + slice 层的 bug
  - iterator 迭代层
+ <a href="#基于 Loop 的 RNN">基于 Loop 的 RNN</a>
  - ReLU RNN
  - 单层单向 LSTM
  - 单层双向 LSTM
+ <a href="#LRN 层">LRN 层</a>（**add_lrn**）
  - window_size & alpha & beta & k
+ <a href="#Matrix Multiply 层">Matrix Multiply 层</a>（**add_matrix_multiply**）
  - 乘数广播
  - op0 & op1
  - 矩阵乘向量
  - transpose0 & transpose1（**add_matrix_multiply_deprecated**）
+ <a href="#padding 层">Padding 层</a>（**add_padding**）
  - pre_padding & post_padding
  - pre_padding_nd & post_padding_nd
+ <a href="#Parametric ReLU 层">Parametric ReLU 层</a>（**add_parametric_relu**）
+ <a href="#Pooling 层">Pooling 层</a>（**add_pooling**）
  - type
  - window_size
  - stride
  - padding
  - pre_padding
  - post_padding
  - padding_mode
  - blend_factor
  - average_count_excludes_padding
  - window_size_nd & stride_nd & padding_nd（**add_pooling_nd**，三维池化）
+ <a href="#Ragged Soft Max 层">Ragged Soft Max 层</a>（**add_ragged_softmax**）
+ <a href="#Reduce 层">Reduce 层</a>（**add_reduce**）
  - op
  - axes
  - keep_dims
+ <a href="#Resize 层">Resize 层</a>（**add_resize**）
  - shape
  - scales
  - resize_mode
  - align_corners
+ <a href="#RNN 层">RNN 层</a>（**add_rnn_v2**）
  - num_layers & hidden_size & max_seq_length & data_length & op
  - seq_lengths
  - input mode
  - direction
  - hidden_state
  - cell state（单向 LSTM 的例子）
  - 双向 LSTM 的例子
  - weight & bias（**add_rnn**）
+ <a href="#Scale 层">Scale 层</a>（**add_scale**）
  - mode & scale & shift & power
  - CHANNEL 和 ELEMENTWISE 级的 scale
  - channel_axis（**add_scale_nd**）
+ <a href="#Select 层">Select 层</a>（**add_select**）
+ <a href="#Shape 层">Shape 层</a>（**add_shape**）
+ <a href="#Shuffle 层">Shuffle 层</a>（**add_shuffle**）
  - first_transpose
  - reshape_dims
  - second_transpose
  - 组合使用的例子
  - zero_is_placeholder
  - 静态 set_input
  - 动态 set_input
  - 在dynamic shape 模式中进行 reshape
+ <a href="#Slice 层">Slice 层</a>（**add_slice**）
  - start & shape & stride
  - mode
  - 静态 set_input
  - 动态 set_input
+ <a href="#Soft Max 层">Soft Max 层</a>（**add_softmax**）
  - axes
+ <a href="#Top K 层">Top K 层</a>（**add_topk**）
  - op
  - k
  - axes
+ <a href="#Unary 层">Unary 层</a>（**add_unary**）
  - op

<div style="page-break-after:always;"></div>
## Activation 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn         = 3                                                                                     # 输入张量 HWC
wIn         = 3
cIn         = 1
data        = np.arange(-4,5,dtype=np.float32).reshape(cIn,hIn,wIn)                                 # 输入张量

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (1, hIn, wIn))
    print("inputTensor->", inputTensor.shape)

    #-----------------------------------------------------------------------------------------------# 可替换部分
    act = network.add_activation(inputTensor, trt.ActivationType.RELU)                              # 使用 ReLU 激活函数
    print("act(default)->", act.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(act.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")

    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)

    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()

    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)

if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (1, 3, 3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        -4. & -3. & -2. \\ -1. & 0. & 1. \\ 2. & 3. & 4.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 3, 3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0. & 0. & 0. \\ 0. & 0. & 1. \\ 2. & 3. & 4.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### type & alpha & beta
```python
    act = network.add_activation(inputTensor, trt.ActivationType.RELU)                              # 替换部分
    act.type    = trt.ActivationType.CLIP                                                           # 激活函数类型，可覆盖函数 add_activation 的参数
    act.alpha   = -2                                                                                # 部分激活函数需要的 1 到 2 个参数，.aplha 和 .beta 默认值均为 0
    act.beta    = 2
    print("act(default)->", act.get_output(0).shape)
```

+ 输出张量 (1, 3, 3)，改用 Clip 激活函数使输出值限制在 -2 到 2 之间
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        -2. & -2. & -2. \\ -1. & 0. & 1. \\ 2. & 2. & 2.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 注意：可用的激活函数类型
| trt.ActivationType 名 | 原名                          | 表达式                                                       |
| :-------------------- | :---------------------------- | :----------------------------------------------------------- |
| RELU                  | Rectified Linear activation   | $f\left(x\right) = \max\left(0,x\right)$                     |
| HARD_SIGMOID          | Hard sigmoid activation       | $f\left(x\right) = \max\left(0,\min\left(1, alpha * x + beta\right)\right)$ |
| THRESHOLDED_RELU      | Thresholded Relu activation   | $f\left(x\right) = \left\{\begin{aligned} x \ \left(x \gt alpha \right) \\ 0 \ \left(x \textcolor[rgb]{1,0,0}{\le} alpha\right) \end{aligned}\right.$ |
| TANH                  | Hyperbolic Tangent activation | $f\left(x\right) = \tanh\left(x\right)$                      |
| LEAKY_RELU            | Leaky Relu activation         | $f\left(x\right) = \left\{\begin{aligned} x \ \left(x \ge 0 \right) \\ alpha * x \ \left(x \lt 0 \right) \end{aligned}\right.$ |
| SCALED_TANH           | Scaled Tanh activation        | $f\left(x\right) = alpha * \tanh\left( beta * x \right)$     |
| CLIP                  | Clip activation               | $f\left(x\right) = \max\left(alpha, \min\left(beta,x\right)\right)$ |
| SOFTPLUS              | Softplus activation           | $f\left(x\right) = alpha * \log\left(\exp\left(beta * x\right) + 1\right)$ |
| SIGMOID               | Sigmoid activation            | $f\left(x\right) = \frac{1}{1 + \exp\left( -x \right)}$      |
| SELU                  | Selu activation               | $f\left(x\right) = \left\{\begin{aligned} beta * x \ \ \left(x \ge 0 \right) \\ beta * alpha * \left( \exp\left(x\right)-1\right) \ \left(x \lt 0 \right) \end{aligned}\right.$ |
| ELU                   | Elu activation                | $f\left(x\right) = \left\{\begin{aligned} x \ \ \left(x \ge 0 \right) \\ alpha * \left( \exp\left(x\right)-1\right) \ \left(x \lt 0 \right) \end{aligned}\right.$ |
| SOFTSIGN              | Softsign activation           | $f\left(x\right) = \frac{x}{1 + \left|x\right|}$             |

<div style="page-break-after:always;"></div>
## Concatenation 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 4                                                                                         # 输入张量 HWC
wIn     = 5
cIn     = 3
data    = np.arange(+cIn*hIn*wIn,dtype=np.float32).reshape(cIn,hIn,wIn)                             # 输入张量

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)

    #-----------------------------------------------------------------------------------------------# 可替换部分
    conca = network.add_concatenation([inputTensor,inputTensor])
    print("conca->", conca.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(conca.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")

    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)

    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()

    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)

if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (3, 4, 5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0. & 1. & 2. & 3. & 4. \\ 5. & 6. & 7. & 8. & 9. \\ 10. & 11. & 12. & 13. & 14. \\ 15. & 16. & 17. & 18. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 21. & 22. & 23. & 24. \\ 25. & 26. & 27. & 28. & 29 \\ 30. & 31. & 32. & 33. & 34. \\ 35. & 36. & 37. & 38. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 41. & 42. & 43. & 44. \\ 45. & 46. & 47. & 48. & 49 \\ 50. & 51. & 52. & 53. & 54. \\ 55. & 56. & 57. & 58. & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (6, 4, 5)，默认在最高“非batch”维上连接
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0. & 1. & 2. & 3. & 4. \\ 5. & 6. & 7. & 8. & 9. \\ 10. & 11. & 12. & 13. & 14. \\ 15. & 16. & 17. & 18. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 21. & 22. & 23. & 24. \\ 25. & 26. & 27. & 28. & 29 \\ 30. & 31. & 32. & 33. & 34. \\ 35. & 36. & 37. & 38. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 41. & 42. & 43. & 44. \\ 45. & 46. & 47. & 48. & 49 \\ 50. & 51. & 52. & 53. & 54. \\ 55. & 56. & 57. & 58. & 59.
    \end{matrix}\right]
    \left[\begin{matrix}
        0. & 1. & 2. & 3. & 4. \\ 5. & 6. & 7. & 8. & 9. \\ 10. & 11. & 12. & 13. & 14. \\ 15. & 16. & 17. & 18. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 21. & 22. & 23. & 24. \\ 25. & 26. & 27. & 28. & 29 \\ 30. & 31. & 32. & 33. & 34. \\ 35. & 36. & 37. & 38. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 41. & 42. & 43. & 44. \\ 45. & 46. & 47. & 48. & 49 \\ 50. & 51. & 52. & 53. & 54. \\ 55. & 56. & 57. & 58. & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### axis
```python
    conca = network.add_concatenation([inputTensor,inputTensor])                                    # 替换部分
    conca.axis = 0                                                                                  # 指定连接维度，默认值为 0
    print("conca->", conca.get_output(0).shape)
```

+ 输出张量 (6, 4, 5)，其中指定 axis = 0，在最高“非batch”维（C 维）上连接，与初始代码相同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0. & 1. & 2. & 3. & 4. \\ 5. & 6. & 7. & 8. & 9. \\ 10. & 11. & 12. & 13. & 14. \\ 15. & 16. & 17. & 18. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 21. & 22. & 23. & 24. \\ 25. & 26. & 27. & 28. & 29 \\ 30. & 31. & 32. & 33. & 34. \\ 35. & 36. & 37. & 38. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 41. & 42. & 43. & 44. \\ 45. & 46. & 47. & 48. & 49 \\ 50. & 51. & 52. & 53. & 54. \\ 55. & 56. & 57. & 58. & 59.
    \end{matrix}\right]
    \left[\begin{matrix}
        0. & 1. & 2. & 3. & 4. \\ 5. & 6. & 7. & 8. & 9. \\ 10. & 11. & 12. & 13. & 14. \\ 15. & 16. & 17. & 18. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 21. & 22. & 23. & 24. \\ 25. & 26. & 27. & 28. & 29 \\ 30. & 31. & 32. & 33. & 34. \\ 35. & 36. & 37. & 38. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 41. & 42. & 43. & 44. \\ 45. & 46. & 47. & 48. & 49 \\ 50. & 51. & 52. & 53. & 54. \\ 55. & 56. & 57. & 58. & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (3, 8, 5)，其中指定 axis = 1，在次高“非batch”维（H 维）上连接
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0. & 1. & 2. & 3. & 4. \\ 5. & 6. & 7. & 8. & 9. \\ 10. & 11. & 12. & 13. & 14. \\ 15. & 16. & 17. & 18. & 19.\\
        0. & 1. & 2. & 3. & 4. \\ 5. & 6. & 7. & 8. & 9. \\ 10. & 11. & 12. & 13. & 14. \\ 15. & 16. & 17. & 18. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 21. & 22. & 23. & 24. \\ 25. & 26. & 27. & 28. & 29 \\ 30. & 31. & 32. & 33. & 34. \\ 35. & 36. & 37. & 38. & 39.\\
        20. & 21. & 22. & 23. & 24. \\ 25. & 26. & 27. & 28. & 29 \\ 30. & 31. & 32. & 33. & 34. \\ 35. & 36. & 37. & 38. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 41. & 42. & 43. & 44. \\ 45. & 46. & 47. & 48. & 49 \\ 50. & 51. & 52. & 53. & 54. \\ 55. & 56. & 57. & 58. & 59.\\
        40. & 41. & 42. & 43. & 44. \\ 45. & 46. & 47. & 48. & 49 \\ 50. & 51. & 52. & 53. & 54. \\ 55. & 56. & 57. & 58. & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (3, 4, 10)，其中指定 axis = 2，在季高“非batch”维（W 维）上连接
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0. & 1. & 2. & 3. & 4. & 0. & 1. & 2. & 3. & 4. \\ 5. & 6. & 7. & 8. & 9. & 5. & 6. & 7. & 8. & 9. \\ 
        10. & 11. & 12. & 13. & 14. & 10. & 11. & 12. & 13. & 14. \\ 15. & 16. & 17. & 18. & 19. &  15. & 16. & 17. & 18. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 21. & 22. & 23. & 24. & 20. & 21. & 22. & 23. & 24. \\ 25. & 26. & 27. & 28. & 29. & 25. & 26. & 27. & 28. & 29 \\
        30. & 31. & 32. & 33. & 34. & 30. & 31. & 32. & 33. & 34. \\ 35. & 36. & 37. & 38. & 39. & 35. & 36. & 37. & 38. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 41. & 42. & 43. & 44. & 40. & 41. & 42. & 43. & 44. \\ 45. & 46. & 47. & 48. & 49. & 45. & 46. & 47. & 48. & 49. \\
        50. & 51. & 52. & 53. & 54. & 50. & 51. & 52. & 53. & 54. \\ 55. & 56. & 57. & 58. & 59. & 55. & 56. & 57. & 58. & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$

<div style="page-break-after:always;"></div>
## Constant 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 4                                                                                         # 张量 HWC
wIn     = 5
cIn     = 3

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()

    #-----------------------------------------------------------------------------------------------# 可替换部分
    const = network.add_constant((cIn,hIn,wIn), np.arange(cIn*hIn*wIn,dtype=np.float32))
    print("const->", const.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(const.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")

    context = engine.create_execution_context()
    stream  = cuda.Stream()
    out1_h  = np.empty(engine.get_binding_shape(0),dtype = trt.nptype(engine.get_binding_dtype(0)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)

    context.execute_async(1, [int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()

    print("out1_h:", out1_h.shape)
    print(out1_h)

if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输出张量 (3, 4, 5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0. & 1. & 2. & 3. & 4. \\ 5. & 6. & 7. & 8. & 9. \\ 10. & 11. & 12. & 13. & 14. \\ 15. & 16. & 17. & 18. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 21. & 22. & 23. & 24. \\ 25. & 26. & 27. & 28. & 29 \\ 30. & 31. & 32. & 33. & 34. \\ 35. & 36. & 37. & 38. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 41. & 42. & 43. & 44. \\ 45. & 46. & 47. & 48. & 49 \\ 50. & 51. & 52. & 53. & 54. \\ 55. & 56. & 57. & 58. & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### weight & shape
```python
    const = network.add_constant((1,), np.zeros(1,dtype=np.float32))                                # 替换部分
    const.weights = np.arange(cIn*hIn*wIn,dtype=np.float32)                                         # 常张量内容，可覆盖函数 add_constant 的参数
    const.shape   = (cIn,hIn,wIn)                                                                   # 常张量形状，可覆盖函数 add_constant 的参数
    print("const->", const.get_output(0).shape)
```

+ 输出张量 (3, 4, 5)，与初始代码相同

$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0. & 1. & 2. & 3. & 4. \\ 5. & 6. & 7. & 8. & 9. \\ 10. & 11. & 12. & 13. & 14. \\ 15. & 16. & 17. & 18. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 21. & 22. & 23. & 24. \\ 25. & 26. & 27. & 28. & 29 \\ 30. & 31. & 32. & 33. & 34. \\ 35. & 36. & 37. & 38. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 41. & 42. & 43. & 44. \\ 45. & 46. & 47. & 48. & 49 \\ 50. & 51. & 52. & 53. & 54. \\ 55. & 56. & 57. & 58. & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$

<div style="page-break-after:always;"></div>
## Convolution 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 6                                                                                         # 输入张量 HWC
wIn     = 9
cIn     = 1
cOut    = 1                                                                                         # 输出张量 C
hW      = 3                                                                                         # 卷积窗口 HW
wW      = 3
data    = np.tile(np.arange(1,1+hW*wW,dtype=np.float32).reshape(hW,wW),(cIn,hIn//hW,wIn//wW)).reshape(cIn,hIn,wIn)  # 输入张量
window  = np.power(10,range(4,-5,-1),dtype=np.float32)                                              # 卷积窗口
bias    = np.zeros(cOut, dtype=np.float32)                                                          # 卷积偏置

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替换部分
    conv = network.add_convolution(inputTensor, cOut, (hW, wW), window, bias)
    print("conv->", conv.get_output(0).shape)                                                       # (cOut,hIn-hW+1,wIn-wW+1)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(conv.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (1, 6, 9)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1. & 2. & 3. & 1. & 2. & 3. & 1. & 2. & 3. \\
        4. & 5. & 6. & 4. & 5. & 6. & 4. & 5. & 6. \\
        7. & 8. & 9. & 7. & 8. & 9. & 7. & 8. & 9. \\
        1. & 2. & 3. & 1. & 2. & 3. & 1. & 2. & 3. \\
        4. & 5. & 6. & 4. & 5. & 6. & 4. & 5. & 6. \\
        7. & 8. & 9. & 7. & 8. & 9. & 7. & 8. & 9. \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 4, 7)，默认没有边缘垫 0，步长为 1
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \textcolor[rgb]{0,0.5,0}{12345.6789} & \textcolor[rgb]{0,0,1}{23156.4897}  & 31264.5978 & 12345.6789 & 23156.4897  & 31264.5978 & 12345.6789 \\
        45678.9123 & 56489.7231 & 64597.8312 & 45678.9123 & 56489.7231 & 64597.8312 & 45678.9123 \\
        78912.3456  & 89723.1564  & 97831.2645  & 78912.3456  & 89723.1564  & 97831.2645  & 78912.3456  \\
        12345.6789 & 23156.4897  & 31264.5978 & 12345.6789 & 23156.4897  & 31264.5978 & 12345.6789
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：卷积结果中各元素的**个位**代表得出该值时卷积窗口的中心位置，其他各位代表参与计算的周围元素。受限于 float32 精度，运行结果无法完整展示 9 位有效数字，以上结果矩阵手工调整了这部分显示，以展示理想运行结果。后续各参数讨论中的输出矩阵不再作调整，而是显示再有舍入误差的原始结果。

$$
\left[\quad\begin{matrix}
    \begin{matrix}{\boxed{
        \begin{matrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{matrix}
    }}\end{matrix}\ 
    \begin{matrix} 1 & \cdots \\ 4 \\ 7\end{matrix}\\
    \begin{matrix} \ \, 1 & 2 & 3 & 1 & \cdots \\ \ \,\vdots & & & \vdots \end{matrix}
\end{matrix}\right]
\otimes
{\boxed{\begin{matrix}
    10^{4} & 10^{3} & 10^{2} \\ 10^{1} & 1 & 10^{-1} \\ 10^{-2} & 10^{-3} & 10^{-4}
\end{matrix}}}
= \textcolor[rgb]{0,0.5,0}{12345.6789},

\left[\quad\begin{matrix}
    \begin{matrix} 1 \\ 4 \\ 7\end{matrix}\ 
    \begin{matrix}{\boxed{
        \begin{matrix}2 & 3 & 1 \\ 5 & 6 & 4 \\ 8 & 9 & 7\end{matrix}
    }}\end{matrix}\
    \begin{matrix} \cdots \\ \\ \\ \end{matrix}\\
    \begin{matrix} 1 & 2 & 3 & 1 & \cdots \\ \vdots & & & \vdots \end{matrix}
\end{matrix}\right]
\otimes
{\boxed{\begin{matrix}
    10^{4} & 10^{3} & 10^{2} \\ 10^{1} & 1 & 10^{-1} \\ 10^{-2} & 10^{-3} & 10^{-4}
\end{matrix}}}
=\textcolor[rgb]{0,0,1}{23156.4897}
$$

---
### kernel_size & num_output_maps & kernel & bias
```python
    fakeWeightAndBias = np.zeros(1, dtype=np.float32)                                               # 替换部分
    conv = network.add_convolution(inputTensor, 1, (1, 1), fakeWeightAndBias, fakeWeightAndBias)    # 先填入一些参数，后续再修改
    conv.kernel_size        = (hW, wW)                                                              # 卷积窗口尺寸，可覆盖函数 add_convolution 的参数
    conv.num_output_maps    = cOut                                                                  # 卷积输出通道数，可覆盖函数 add_convolution 的参数
    conv.kernel             = window                                                                # 卷积窗口权值，可覆盖函数 add_convolution 的参数
    conv.bias               = bias                                                                  # 卷积偏置，可覆盖函数 add_constant 的参数
    print("conv->", conv.get_output(0).shape)                                                       # (cOut,hIn-hW+1,wIn-wW+1)
```

+ 输出张量 (1, 4, 7)，与初始代码相同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        12345.6789 & 23156.4897 & 31264.5978 & 12345.6789 & 23156.4897 & 31264.5978 & 12345.6789 \\
        45678.9123 & 56489.7231 & 64597.8312 & 45678.9123 & 56489.7231 & 64597.8312 & 45678.9123 \\
        78912.3456 & 89723.1564 & 97831.2645 & 78912.3456 & 89723.1564 & 97831.2645 & 78912.3456 \\
        12345.6789 & 23156.4897 & 31264.5978 & 12345.6789 & 23156.4897 & 31264.5978 & 12345.6789
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### stride
```python
    hS = wS = 2                                                                                     # 替换部分
    conv = network.add_convolution(inputTensor, cOut, (hW, wW), window, bias)
    conv.stride = (hS,wS)                                                                           # 卷积核移动步长，默认值为 (1,1)
    print("conv->", conv.get_output(0).shape)                                                       # (cOut,(hIn-hW+1)//hS,(wIn-wW+1)//wS)
```

+ 输出张量 (1, 2, 7)，其中指定 stride=(2,1)，H 维跨步 2
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        12345.679 & 23156.49 & 31264.598 & 12345.679 & 23156.49 & 31264.598 & 12345.679 \\
        78912.34  & 89723.16 & 97831.27  & 78912.34  & 89723.16 & 97831.27  & 78912.34
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 4, 4)，其中指定 stride=(1,2)，W 维跨步 2
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        12345.679 & 31264.598 & 23156.49  & 12345.679 \\
        45678.914 & 64597.832 & 56489.723 & 45678.914 \\
        78912.34  & 97831.27  & 89723.16  & 78912.34  \\
        12345.679 & 31264.598 & 23156.49  & 12345.679
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 2, 4)，其中指定 stride=(2,2)，HW 维跨步均为 2
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        12345.679 & 31264.598 & 23156.49 & 12345.679 \\
        78912.34  & 97831.27  & 89723.16 & 78912.34 
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### padding
```python
    hP = wP = 1                                                                                     # 替换部分
    conv = network.add_convolution(inputTensor, cOut, (hW, wW), window, bias)
    conv.padding = (hP, wP)                                                                         # 四周垫 0 层数，默认值为 (0,0)
    print("conv->", conv.get_output(0).shape)                                                       # (cOut,hIn-hW+hP*2+1,wIn-wW+wP*2+1)
```

+ 输出张量 (1, 6, 7)，其中指定 padding=(1,0)，H 维垫 1 层元素 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
           12.3456&    23.1564&    31.2645&    12.3456&    23.1564&    31.2645&    12.3456\\
        12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 \\
        45678.914 & 56489.723 & 64597.832 & 45678.914 & 56489.723 & 64597.832 & 45678.914 \\
        78912.34  & 89723.16  & 97831.27  & 78912.34  & 89723.16  & 97831.27  & 78912.34  \\
        12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 \\
        45678.9   & 56489.7   & 64597.8   & 45678.9   & 56489.7   & 64597.8   & 45678.9   \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 4, 9)，其中指定 padding=(0,1)，W 维垫 1 层元素 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1204.5078 & 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23056.09  \\
        4507.801  & 45678.914 & 56489.723 & 64597.832 & 45678.914 & 56489.723 & 64597.832 & 45678.914 & 56089.023 \\
        7801.2046 & 78912.34  &  89723.16 &  97831.27 & 78912.34  & 89723.16  & 97831.27  & 78912.34  & 89023.055 \\
        1204.5078 & 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23056.09
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 6, 9)，其中指定 padding=(1,1)，HW 维均垫 1 层元素 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
           1.2045&    12.3456&    23.1564&    31.2645&    12.3456&    23.1564&    31.2645&    12.3456&   23.056  \\
        1204.5078& 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23056.09  \\ 
        4507.801 & 45678.914 & 56489.723 & 64597.832 & 45678.914 & 56489.723 & 64597.832 & 45678.914 & 56089.023 \\
        7801.2046& 78912.34  & 89723.16  & 97831.27  & 78912.34  & 89723.16  & 97831.27  & 78912.34  & 89023.055 \\
        1204.5078& 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23056.09  \\
        4507.8   & 45678.9   & 56489.7   & 64597.8   & 45678.9   & 56489.7   & 64597.8   & 45678.9   & 56089.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### pre_padding
```python
    hPre = wPre = 1                                                                                 # 替换部分
    conv = network.add_convolution(inputTensor, cOut, (hW, wW), window, bias)
    conv.pre_padding = (hPre, wPre)                                                                 # 顶部和左侧垫 0 层数，默认值为 (0,0)
    print("conv->", conv.get_output(0).shape)                                                       # (cOut,hIn-hW+hPre+1,wIn-wW+wPre+1)
```

+ 输出张量 (1, 5, 7)，其中指定 pre_padding=(1,0)，H 维头部垫 1 层元素 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
           12.3456&    23.1564&    31.2645&    12.3456&    23.1564&    31.2645&    12.3456\\
        12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 \\
        45678.914 & 56489.723 & 64597.832 & 45678.914 & 56489.723 & 64597.832 & 45678.914 \\
        78912.34  & 89723.16  & 97831.27  & 78912.34  & 89723.16  & 97831.27  & 78912.34  \\
        12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 4, 8)，其中指定 pre_padding=(0,1)，W 维头部垫 1 层元素 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1204.5078& 12345.679&  23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 \\
        4507.801 & 45678.914&  56489.723 & 64597.832 & 45678.914 & 56489.723 & 64597.832 & 45678.914 \\
        7801.2046& 78912.34 &  89723.16  & 97831.27  & 78912.34  & 89723.16  & 97831.27  & 78912.34  \\
        1204.5078& 12345.679&  23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 5, 8)，其中指定 pre_padding=(1,1)，HW 维头部均垫 1 层元素 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1204.5078& 12345.679&  23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 \\
        4507.801 & 45678.914&  56489.723 & 64597.832 & 45678.914 & 56489.723 & 64597.832 & 45678.914 \\
        7801.2046& 78912.34 &  89723.16  & 97831.27  & 78912.34  & 89723.16  & 97831.27  & 78912.34  \\
        1204.5078& 12345.679&  23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### post_padding
```python
    hPost = wPost = 1                                                                               # 替换部分
    conv = network.add_convolution(inputTensor, cOut, (hW, wW), window, bias)
    conv.post_padding = (hPost, wPost)                                                              # 底部和右侧垫 0 层数，默认值为 (0,0)
    print("conv->", conv.get_output(0).shape)                                                       # (cOut,hIn-hW+hPost+1,wIn-wW+wPost+1)
```

+ 输出张量 (1, 5, 7)，其中指定 post_padding=(1,0)，H 维尾部垫 1 层元素 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 \\
        45678.914 & 56489.723 & 64597.832 & 45678.914 & 56489.723 & 64597.832 & 45678.914 \\
        78912.34  & 89723.16  & 97831.27  & 78912.34  & 89723.16  & 97831.27  & 78912.34  \\
        12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 \\
        45678.9   & 56489.7   & 64597.8   & 45678.9   & 56489.7   & 64597.8   & 45678.9  
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 4, 8)，其中指定 post_padding=(0,1)，W 维尾部垫 1 层元素 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23056.09  \\
        45678.914 & 56489.723 & 64597.832 & 45678.914 & 56489.723 & 64597.832 & 45678.914 & 56089.023 \\
        78912.34  & 89723.16  & 97831.27  & 78912.34  & 89723.16  & 97831.27  & 78912.34  & 89023.055 \\
        12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23056.09
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 5, 8)，其中指定 post_padding=(1,1)，HW 维尾部均垫 1 层元素 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23056.09  \\
        45678.914 & 56489.723 & 64597.832 & 45678.914 & 56489.723 & 64597.832 & 45678.914 & 56089.023 \\
        78912.34  & 89723.16  & 97831.27  & 78912.34  & 89723.16  & 97831.27  & 78912.34  & 89023.055 \\
        12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 & 23056.09  \\
        45678.9   & 56489.7   & 64597.8   & 45678.9   & 56489.7   & 64597.8   & 45678.9   & 56089.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### padding_mode
```python
    conv = network.add_convolution(inputTensor, cOut, (hW, wW), window, bias)                       # 替换部分
    conv.stride = (2,2)                                                                             # 加上卷积步长，以便观察结果变化
    conv.padding_mode = trt.PaddingMode.SAME_UPPER                                                  # 垫 0 方案，优先级高于 padding，pre_padding 和 post_padding，默认值为 None
    print("conv->", conv.get_output(0).shape)
```

+ 输出张量 (1, 3, 5)，其中指定 padding_mode = **trt.PaddingMode.SAME_UPPER**
  目标尺寸 $ \left( h',w' \right) = \left( \lceil \frac{hIn}{hS} \rceil, \lceil \frac{wIn}{wS} \rceil \right) $，为此 H 维需要总垫 0 层数 $ hP = \left( h' -1 \right)hS + hW - hIn $，然后取 $ hPre = \textcolor[rgb]{1,0,0}{\lfloor \frac{hP}{2} \rfloor} $，$ hPost = \textcolor[rgb]{1,0,0}{\lceil \frac{hP}{2} \rceil} $，W 维类似
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1204.5078& 23156.49  &  12345.679 & 31264.598 & 23056.09  \\
        7801.2046& 89723.16  &  78912.34  & 97831.27  & 89023.055 \\
        4507.8   & 56489.7   &  45678.9   & 64597.8   & 56089.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 3, 5)，其中指定 padding_mode = **trt.PaddingMode.SAME_LOWER**
  目标尺寸同上，然后 $ hPre = \textcolor[rgb]{1,0,0}{\lceil \frac{hP}{2} \rceil} $，$ hPost = \textcolor[rgb]{1,0,0}{\lfloor \frac{hP}{2} \rfloor} $，W 维类似
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1.2045   & 23.1564   & 12.3456   & 31.2645   & 23.056    \\
        4507.801 & 56489.723 & 45678.914 & 64597.832 & 56089.023 \\
        1204.5078& 23156.49  & 12345.679 & 31264.598 & 23056.09 
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 3, 4)，其中指定 padding_mode = **trt.PaddingMode.EXPLICIT_ROUND_UP**
    目标尺寸 $ \left( h',w' \right) = \left( \lceil \frac{hIn - \frac{hW-1}{2}}{hS} \rceil, \lceil \frac{wIn - \frac{wW-1}{2}}{wS} \rceil \right) $，为此 H 维需要总垫 0 层数 $ hP = \left( h' -1 \right)hS + hW - hIn $，全部垫在底部 $ hPost = hP $，W 维类似
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        12345.679 & 31264.598 & 23156.49 & 12345.679 \\
        78912.34  & 97831.27  & 89723.16 & 78912.34  \\
        45678.9   & 64597.8   & 56489.7  & 45678.9  
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 2, 4)，其中指定 padding_mode = **trt.PaddingMode.EXPLICIT_ROUND_DOWN**，不做任何垫 0 处理，如果卷积窗口超出原张量边界，则舍弃该位置的卷积计算
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        12345.679 & 31264.598 & 23156.49 & 12345.679 \\
        78912.34  & 97831.27  & 89723.16 & 78912.34 
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量形状 **None**，其中指定 padding_mode = **trt.PaddingMode.CAFFE_ROUND_UP**，不做任何垫 0 处理，如果卷积窗口超出原张量边界，则报错

> [TensorRT] ERROR: ../rtSafe/cuda/caskConvolutionRunner.cpp (62) - Cask Error in createConvolution: 3 (isConsistent)

+ 输出张量 (1, 2, 4)，其中指定 padding_mode = trt.PaddingMode.CAFFE_ROUND_UP 且预先手动调整输入张量尺寸为 (1, 5, 9)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        12345.679 & 31264.598 & 23156.49 & 12345.679 \\
        78912.34  & 97831.27  & 89723.16 & 78912.34 
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 2, 4)，其中指定 padding_mode = **trt.PaddingMode.CAFFE_ROUND_DOWN**，不做任何垫 0 处理，如果卷积窗口超出原张量边界，则舍弃该位置的卷积计算
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        12345.679 & 31264.598 & 23156.49 & 12345.679 \\
        78912.34  & 97831.27  & 89723.16 & 78912.34 
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### dilation
```python
    hD = wD = 2                                                                                     # 替换部分
    conv = network.add_convolution(inputTensor, cOut, (hW, wW), window, bias)
    conv.dilation = (hD, wD)                                                                        # 卷积核扩张度，表示卷积核相邻两元素在该轴上的间隔，默认值为 (1,1)
    print("conv->", conv.get_output(0).shape)                                                       # (cOut,hIn-hW-2*hD+3,wIn-wW-2*wD+3)
```

+ 输出张量 (1, 2, 7)，其中指定 dilation=(2,1)，卷积核在 H 维上元素间隔为 2
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        12378.946 & 23189.756 & 31297.865 & 12378.946 & 23189.756 & 31297.865 & 12378.946 \\
        45612.38  & 56423.188 & 64531.297 & 45612.38  & 56423.188 & 64531.297 & 45612.38
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 4, 5)，其中指定 dilation=(1,2)，卷积核在 W 维上元素间隔为 2
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        13246.58  & 21354.688 & 32165.498 & 13246.58  & 21354.688 \\
        46579.816 & 54687.918 & 65498.734 & 46579.816 & 54687.918 \\
        79813.25  & 87921.35  & 98732.17  & 79813.25  & 87921.35  \\
        13246.58  & 21354.688 & 32165.498 & 13246.58  & 21354.688
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 2, 5)，其中指定 dilation=(2,2)，卷积核在 HW 维上元素间隔均为 2
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        13279.847 & 21387.955 & 32198.766 & 13279.847 & 21387.955 \\
        46513.277 & 54621.387 & 65432.2   & 46513.277 & 54621.387
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### num_groups
```python
# 调整部分参数
cIn         = 2
numberGroup = 2
cOut        = numberGroup
data        = np.tile(np.arange(1,1+hW*wW,dtype=np.float32).reshape(hW,wW),(cIn,hIn//hW,wIn//wW)).reshape(cIn,hIn,wIn)
window      = np.power(10,range(4,-5,-1),dtype=np.float32)
window      = np.concatenate([window,-window],0)                                                    # 卷积窗口通道数必须能被分组数整除
bias        = np.full(cOut, 0, dtype=np.float32)
```
```python
    conv = network.add_convolution(inputTensor, cOut, (hW, wW), window, bias)                       # 替换部分
    conv.num_groups = numberGroup                                                                   # 分组数，默认值为 1
    print("conv->", conv.get_output(0).shape)                                                       # (cOut,hIn-hW+1,wIn-wW+1)
```

+ 输出张量 (2, 4, 7)，其中指定 num_groupds=2，输入张量和卷积核均在 C 维上被均分为 2 组，各自卷积后再拼接到一起
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679 \\
        45678.914 & 56489.723 & 64597.832 & 45678.914 & 56489.723 & 64597.832 & 45678.914 \\
        78912.34  & 89723.16  & 97831.27  & 78912.34  & 89723.16  & 97831.27  & 78912.34  \\
        12345.679 & 23156.49  & 31264.598 & 12345.679 & 23156.49  & 31264.598 & 12345.679
    \end{matrix}\right]\\
    \left[\begin{matrix}
        -12345.679 & -23156.49  & -31264.598 & -12345.679 & -23156.49  & -31264.598 & -12345.679 \\
        -45678.914 & -56489.723 & -64597.832 & -45678.914 & -56489.723 & -64597.832 & -45678.914 \\
        -78912.34  & -89723.16  & -97831.27  & -78912.34  & -89723.16  & -97831.27  & -78912.34  \\
        -12345.679 & -23156.49  & -31264.598 & -12345.679 & -23156.49  & -31264.598 & -12345.679
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### kernel_size_nd & stride_nd & padding_nd & dilation_nd（add_convolution_nd）
```python
# 三维卷积的例子，调整部分参数
cIn     = 2
cOut    = 1
data    = np.tile(np.arange(1,1+hW*wW,dtype=np.float32).reshape(hW,wW),(cIn,hIn//hW,wIn//wW)).reshape(cIn,hIn,wIn)
window  = np.power(10,range(4,-5,-1),dtype=np.float32)
window  = np.concatenate([window,-window],0)
bias    = np.full(cOut, 0, dtype=np.float32)
```
```python
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (1,cIn, hIn, wIn))           # 替换部分，要求输入张量维度是 4
    #...
    conv = network.add_convolution_nd(inputTensor, 1, (2,hW, wW), window, bias)                     # 注意卷积核是 3 维的
    conv.kernel_size_nd = (2,hW,wW)                                                                 # 卷积核尺寸
    conv.stride_nd      = (1, 1, 1)                                                                 # 卷积核移动步长，默认值为 (1,1,1)
    conv.padding_nd     = (0, 0, 0)                                                                 # 四周垫 0 层数，默认值为 (0,0,0)
    conv.dilation_nd    = (1, 1, 1)                                                                 # 卷积核扩展数，默认值为 (1,1,1)
    print("conv->", conv.get_output(0).shape)                                                       # (1,cOut,hIn-hW+1,wIn-wW+1)
```

+ 输出张量 (1, 1, 4, 7)，相当于把前面 num_groups 例子中结果的两个通道加在一起，得到了 0 的结果
$$
\left[\begin{matrix}
\left[\begin{matrix}
    \left[\begin{matrix}
        -0.00018907 &  0.00053437 & -0.00014376 & -0.00018907 &  0.00053437 & -0.00014376 & -0.00018907 \\
         0.00176249 & -0.00044376 &  0.00083124 &  0.00176249 & -0.00044376 &  0.00083124 &  0.00176249 \\
        -0.00185    & -0.00015    &  0.0089375  & -0.00185    & -0.00015    &  0.0089375  & -0.00185    \\
        -0.00018907 &  0.00053437 & -0.00014376 & -0.00018907 &  0.00053437 & -0.00014376 & -0.00018907
    \end{matrix}\right]
\end{matrix}\right]
\end{matrix}\right]
$$

<div style="page-break-after:always;"></div>
## Deconvolution 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 3                                                                                         # 输入张量 HWC
wIn     = 3
cIn     = 1
cOut    = 1                                                                                         # 输出张量 C
hW      = 3                                                                                         # 卷积窗口 HW
wW      = 3
data    = np.arange(1,1+hIn*wIn,dtype=np.float32).reshape(cIn,hIn,wIn)                              # 输入张量
window  = np.power(10,range(4,-5,-1),dtype=np.float32)                                              # 卷积窗口
bias    = np.full(cOut, 0, dtype=np.float32)                                                        # 卷积偏置

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替换部分
    deconv = network.add_deconvolution(inputTensor, cOut, (hW, wW), window, bias)
    print("deconv->", deconv.get_output(0).shape)                                                   # (cOut,hIn+hW-1,wIn+wW-1)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(deconv.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (1, 3, 3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1. & 2. & 3. \\
        4. & 5. & 6. \\
        7. & 8. & 9.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 5, 5)，默认没有边缘垫 0，步长为 1
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \textcolor[rgb]{0,0.5,0}{10000.0000}& 21000.0000& 32100.0000& 3200.0000&  300.0000\\
        40010.0000& 54021.0000& 65432.1000& 6503.2   &  600.3000\\
        70040.0100& 87054.0210& \textcolor[rgb]{0,0,1}{98765.4321}& 9806.5032&  900.6003 \\
           70.0400&    87.0540&    98.76540&   9.8065&    0.9006\\
            0.0700&     0.0870&     0.09867&   0.0098&    \textcolor[rgb]{1,0,0}{0.0009}
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 注意：反卷积结果中各元素的**个位**代表得出该值时卷积窗口的中心位置，其他各位代表参与计算的周围元素，注意反卷积核是倒序的。受限于 float32 精度，运行结果无法完整展示 9 位有效数字，以上结果矩阵手工调整了这部分显示，以展示理想运行结果。后续各参数讨论中的输出矩阵不再作调整，而是显示再有舍入误差的原始结果。
$$
\left[\begin{matrix}
    \begin{matrix}{\boxed{
        \begin{matrix} \ & \ & \ \\ \ & \  & \ \\ \ & \ & 1 \end{matrix}
    }}\end{matrix}\ 
    \begin{matrix} & \\ & \\ 2 & 3\end{matrix}\\
    \begin{matrix} \ \ \ \ \ \ \, & \  & 4 & 5 & 6 & \\ \ \ \ \ & & 7 & 8 & 9\end{matrix}
\end{matrix}\right]
\otimes
{\boxed{\begin{matrix}
    10^{-4} & 10^{-3} & 10^{-2} \\ 10^{-1} & 1 & 10^{1} \\ 10^{2} & 10^{3} & 10^{4}
\end{matrix}}}
= \textcolor[rgb]{0,0.5,0}{10000.},

\left[\quad\begin{matrix}\\
    \begin{matrix}{\boxed{
        \begin{matrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{matrix}
    }}\end{matrix}\\
    \begin{matrix}\ \end{matrix}
\end{matrix}\quad\right]
\otimes
{\boxed{\begin{matrix}
    10^{-4} & 10^{-3} & 10^{-2} \\ 10^{-1} & 1 & 10^{1} \\ 10^{2} & 10^{3} & 10^{4}
\end{matrix}}}
=\textcolor[rgb]{0,0,1}{98765.4321},

\left[\quad\begin{matrix}
    \begin{matrix}
    	1 & 2 & 3 & \ & \ \ \\ 4 & 5 & 6 & \ & \ \ 
   	\end{matrix}\\
   	\begin{matrix}
    	7 & 8 \\ \ & \ \\ & \
   	\end{matrix}\
   	{\boxed{
        \begin{matrix} \ 9 & \ & \ \\ \ & \ & \ \\ \ & \ & \ \end{matrix}
    }}
\end{matrix}\quad\right]
\otimes
{\boxed{\begin{matrix}
    10^{-4} & 10^{-3} & 10^{-2} \\ 10^{-1} & 1 & 10^{1} \\ 10^{2} & 10^{3} & 10^{4}
\end{matrix}}}
=\textcolor[rgb]{1,0,0}{0.0009}
$$

---
### kernel_size & num_output_maps & kernel & bias
```python
    fakeWeightAndBias = np.zeros(1, dtype=np.float32)                                               # 替换部分
    deconv = network.add_deconvolution(inputTensor, 1, (1, 1), fakAndBias, fakeWeightAndBias)
    deconv.kernel_size        = (hW, wW)                                                            # 反卷积窗口尺寸，可覆盖函数 add_deconvolution 的参数
    deconv.num_output_maps    = cOut                                                                # 翻卷技术处通道数，可覆盖函数 add_deconvolution 的参数
    deconv.kernel             = window                                                              # 反卷积窗口权值，可覆盖函数 add_deconvolution 的参数
    deconv.bias               = bias                                                                # 反卷积偏置，可覆盖函数 add_deconvolution 的参数
    print("deconv->", deconv.get_output(0).shape)                                                   # (cOut,hIn+hW-1,wIn+wW-1)
```

+ 输出张量形状(1, 5, 5)，与初始代码相同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        10000.   & 21000.    & 32100.        & 3200.       & 300.    \\
        40010.   & 54021.    & 65432.1       & 6503.2      & 600.3   \\
        70040.01 & 87054.02  & 98765.43      & 9806.503    & 900.6003\\
           70.04 &    87.054 &    98.765396  &    9.8064995&   0.9006\\
            0.07 &     0.087 &     0.09869999&    0.0098   &   0.0009
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### stride
```python
    hS = wS = 2                                                                                     # 替换部分
    deconv = network.add_deconvolution(inputTensor, cOut, (hW, wW), window, bias)
    deconv.stride = (hS, wS)                                                                        # 反卷积核移动步长，默认值为 (1,1)
    print("deconv->", deconv.get_output(0).shape)                                                   # (cOut,(hIn-1)*hS+hW,(wIn-1)*wS+wW)
```

+ 输出张量 (1, 9, 6)，其中指定 stride=(2,1)，H 维跨步 2
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        10000.    & 21000.    & 32100.        & 3200.    &  300.        \\
           10.    &    21.    &    32.1       &    3.2   &    0.3       \\
        40000.01  & 54000.02  & 65400.035     & 6500.0034&  600.0003    \\
           40.    &    54.         65.4       &    6.5   &    0.6       \\
        70000.04  & 87000.055 & 98700.07      & 9800.007 &  900.0006    \\
           70.    &    87.    &    98.7       &    9.8   &    0.90000004\\
            0.07  &     0.087 &     0.09869999&    0.0098&    0.0009
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 5, 7)，其中指定 stride=(1,2)，W 维跨步 2
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        10000.    & 1000. &    20100.    & 2000.    & 30200.        & 3000.    & 300.    \\
        40010.    & 4001. &    50420.1   & 5002.    & 60530.2       & 6003.    & 600.3   \\
        70040.01  & 7004.001 & 80750.42  & 8005.002 & 90860.53      & 9006.003 & 900.6003\\
           70.04  &    7.004 &    80.7504&    8.005 &    90.860504  &    9.006 &   0.9006\\
            0.07  &    0.007 &     0.0807&    0.008 &     0.09079999&    0.009 &   0.0009
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 7, 7)，其中指定 stride=(2,2)，HW 维均跨步 2
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        10000.    & 1000.    & 20100.    & 2000.    & 30200.        & 3000.    & 300.        \\
           10.    &    1.    &    20.1   &    2.    &    30.2       &    3.    &   0.3       \\
        40000.01  & 4000.001 & 50400.02  & 5000.002 & 60500.03      & 6000.003 & 600.0003    \\
           40.    &    4.    &    50.4   &    5.    &    60.5       &    6.    &   0.6       \\
        70000.04  & 7000.004 & 80700.05  & 8000.005 & 90800.06      & 9000.006 & 900.0006    \\
           70.    &    7.    &    80.7   &    8.    &    90.8       &    9.    &   0.90000004\\
            0.07  &    0.007 &     0.0807&    0.008 &     0.09079999&    0.009 &   0.0009
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### padding
```python
    hP = wP = 1                                                                                     # 替换部分
    deconv = network.add_deconvolution(inputTensor, cOut, (hW, wW), window, bias)
    deconv.padding = (hP, wP)                                                                       # 垫 0 层减少数，默认值为 (0,0)
    print("deconv->", deconv.get_output(0).shape)                                                   # (cOut,hIn+hW-hP*2-1,wIn-wW+wP*2-1)
```

+ 输出张量 (1, 3, 5)，其中指定 padding=(1,0)，H 维垫 0 减少 1 层
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        40010.    & 54021.    & 65432.1     & 6503.2      &   600.3   \\
        70040.01  & 87054.02  & 98765.43    & 9806.503    & 900.6003  \\
           70.04  &    87.054 &    98.765396&    9.8064995&     0.9006
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 5, 3)，其中指定 padding=(0,1)，W 维垫 0 减少 1 层
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        21000.    & 32100.        & 3200.       \\
        54021.    & 65432.1       & 6503.2      \\
        87054.02  & 98765.43      & 9806.503    \\
           87.054 &    98.765396  &    9.8064995\\
            0.087 &     0.09869999&    0.0098
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 3, 3)，其中指定 padding=(1,1)，HW 维垫 0 均减少 1 层
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        54021.    & 65432.1     &  6503.2  \\
        87054.02  & 98765.43    & 9806.503 \\
           87.054 &    98.765396& 9.8064995
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### pre_padding
```python
    hPre = wPre = 1                                                                                 # 替换部分
    deconv = network.add_deconvolution(inputTensor, cOut, (hW, wW), window, bias)
    deconv.pre_padding = (hPre, wPre)                                                               # 顶部和左侧垫 0 层减少数，默认值为 (0,0)
    print("deconv->", deconv.get_output(0).shape)                                                   # (cOut,hIn+hW-hPre-1,wIn+wW-wPre-1)
```

+ 输出张量 (1, 4, 5)，其中指定 pre_padding=(1,0)，H 维头部垫 0 减少 1 层
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        40010.    & 54021.    & 65432.1      &  6503.2      & 600.3   \\
        70040.01  & 87054.02  & 98765.43     &  9806.503    & 900.6003\\
           70.04  &    87.054 &    98.765396 &     9.8064995&   0.9006\\
            0.07  &     0.087 &    0.09869999&     0.0098   &   0.0009
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 5, 4)，其中指定 pre_padding=(0,1)，W 维头部垫 0 减少 1层
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        21000.    & 32100.        & 3200.       & 300.    \\
        54021.    & 65432.1       & 6503.2      & 600.3   \\
        87054.02  & 98765.43      & 9806.503    & 900.6003\\
           87.054 &    98.765396  &    9.8064995&   0.9006\\
            0.087 &     0.09869999&    0.0098   &   0.0009
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 4, 4)，其中指定 pre_padding=(1,1)，HW 维头部垫 0 均减少 1 层
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        54021.    & 65432.1       &  6503.2      & 600.3   \\ 
        87054.02  & 98765.43      &  9806.503    & 900.6003\\
           87.054 &    98.765396  &     9.8064995&   0.9006\\
            0.087 &     0.09869999&     0.0098   &   0.0009
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### post_padding
```python
    hPost = wPost = 1                                                                               # 替换部分
    deconv = network.add_deconvolution(inputTensor, cOut, (hW, wW), window, bias)
    deconv.post_padding = (hPost, wPost)                                                            # 底部和右侧垫 0 层减少数，默认值为 (0,0)
    print("deconv->", deconv.get_output(0).shape)                                                   # (cOut,hIn+hW-hPost-1,wIn+wW-Post-1)
```

+ 输出张量 (1, 4, 5)，其中指定 post_padding=(1,0)，H 维尾部垫 0 减少 1 层
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        10000.    & 21000.    & 32100.      & 3200.       & 300.    \\
        40010.    & 54021.    & 65432.1     & 6503.2      & 600.3   \\
        70040.01  & 87054.02  & 98765.43    & 9806.503    & 900.6003\\
           70.04  &    87.054 &    98.765396&    9.8064995&   0.9006
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 5, 4)，其中指定 post_padding=(0,1)，W 维尾部垫 0减少 1 层
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        10000.    & 21000.    & 32100.        & 3200.       \\  
        40010.    & 54021.    & 65432.1       & 6503.2      \\
        70040.01  & 87054.02  & 98765.43      & 9806.503    \\
           70.04  &    87.054 &    98.765396  &    9.8064995\\
            0.07  &     0.087 &     0.09869999&    0.0098
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 4, 4)，其中指定 post_padding=(1,1)，HW 维尾部垫 0 均减少 1 层
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        10000.    & 21000.    & 32100.      & 3200.       \\
        40010.    & 54021.    & 65432.1     & 6503.2      \\
        70040.01  & 87054.02  & 98765.43    & 9806.503    \\
           70.04  &    87.054 &    98.765396&    9.8064995
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### padding_mode
```python
    deconv = network.add_deconvolution(inputTensor, cOut, (hW, wW), window, bias)                   # 替换部分
    deconv.stride = (2, 2)
    deconv.padding_mode = trt.PaddingMode.SAME_UPPER                                                # 垫 0 层减少方案，默认值为 None
    print("deconv->", deconv.get_output(0).shape)
```
+ 输出张量 (1, 6, 6)，其中指定 padding_mode = **trt.PaddingMode.SAME_UPPER**
  目标尺寸 $ \left( h',w' \right) = \left( hIn \cdot hS, wIn \cdot wS \right) $，为此 H 维垫 0 需要减少数 $ hP = \left( hIn -1 \right)hS + hW - h' $，然后取 $ hPre = \textcolor[rgb]{1,0,0}{\lfloor \frac{hP}{2} \rfloor} $，$ hPost = \textcolor[rgb]{1,0,0}{\lceil \frac{hP}{2} \rceil} $，W 维类似
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        10000. & 1000. & 20100. & 2000. & 30200. & 3000. \\
        10. & 1. & 20.1 & 2. & 30.2 & 3. \\
        40000.01 & 4000.001 & 50400.02 & 5000.002 & 60500.03 & 6000.003 \\
        40. & 4. & 50.4 & 5. & 60.5 & 6. \\
        70000.04 &  7000.004 & 80700.05 & 8000.005 & 90800.06 & 9000.006 \\
        70. & 7. & 80.7 & 8. & 90.8 & 9.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 6, 6)，其中指定 padding_mode = **trt.PaddingMode.SAME_LOWER**
  目标尺寸同上，然后 $ hPre = \textcolor[rgb]{1,0,0}{\lceil \frac{hP}{2} \rceil} $，$ hPost = \textcolor[rgb]{1,0,0}{\lfloor \frac{hP}{2} \rfloor} $，W 维类似
$$
\left[\begin{matrix}
    \left[\begin{matrix}s
        1. & 20.1 & 2. & 30.2 & 3. & 0.3 \\
        4000.001 & 50400.02 & 5000.002 & 60500.03 & 6000.003 & 600.0003 \\
        4. & 50.4 & 5. & 60.5 & 6. & 0.6 \\
        7000.004 & 80700.05 & 8000.005 & 90800.06 & 9000.006 & 900.0006 \\
        7. & 80.7 & 8. & 90.8 & 9. & 0.90000004 \\
        0.007 & 0.0807 & 0.008 & 0.09079999 & 0.009 & 0.0009
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 7, 7)，其中指定 padding_mode = **trt.PaddingMode.EXPLICIT_ROUND_UP**
    目标尺寸 $ \left( h',w' \right) = \left( \left( hIn - 1 \right) hS + hW, \left( wIn - 1 \right) wS + wW \right) $，不做任何垫 0 减少处理
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        10000. & 1000. & 20100. & 2000. & 30200. & 3000. & 300. \\
        10. & 1. & 20.1 & 2. & 30.2 & 3. & 0.3 \\
        40000.01 & 4000.001 & 50400.02 & 5000.002 & 60500.03 & 6000.003 & 600.0003 \\
        40. & 4. & 50.4 & 5. & 60.5 & 6. & 0.6 \\
        70000.04 & 7000.004 & 80700.05 & 8000.005 & 90800.06 & 9000.006 & 900.0006 \\
        70. & 7. & 80.7 & 8. & 90.8 & 9. & 0.90000004 \\
        0.07 & 0.007 & 0.0807 & 0.008 & 0.09079999 & 0.009 & 0.0009
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 7, 7)，其中指定 padding_mode = **trt.PaddingMode.EXPLICIT_ROUND_DOWN**，目标尺寸同上，不做任何垫 0 减少处理
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        10000. & 1000. & 20100. & 2000. & 30200. & 3000. & 300. \\
        10. & 1. & 20.1 & 2. & 30.2 & 3. & 0.3 \\
        40000.01 & 4000.001 & 50400.02 & 5000.002 & 60500.03 & 6000.003 & 600.0003 \\
        40. & 4. & 50.4 & 5. & 60.5 & 6. & 0.6 \\
        70000.04 & 7000.004 & 80700.05 & 8000.005 & 90800.06 & 9000.006 & 900.0006 \\
        70. & 7. & 80.7 & 8. & 90.8 & 9. & 0.90000004 \\
        0.07 & 0.007 & 0.0807 & 0.008 & 0.09079999 & 0.009 & 0.0009
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1,7,7)，其中指定 padding_mode = **trt.PaddingMode.CAFFE_ROUND_UP**，目标尺寸同上，不做任何垫 0 处理
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        10000. & 1000. & 20100. & 2000. & 30200. & 3000. & 300. \\
        10. & 1. & 20.1 & 2. & 30.2 & 3. & 0.3 \\
        40000.01 & 4000.001 & 50400.02 & 5000.002 & 60500.03 & 6000.003 & 600.0003 \\
        40. & 4. & 50.4 & 5. & 60.5 & 6. & 0.6 \\
        70000.04 & 7000.004 & 80700.05 & 8000.005 & 90800.06 & 9000.006 & 900.0006 \\
        70. & 7. & 80.7 & 8. & 90.8 & 9. & 0.90000004 \\
        0.07 & 0.007 & 0.0807 & 0.008 & 0.09079999 & 0.009 & 0.0009
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1,7,7)，其中指定 padding_mode = **trt.PaddingMode.CAFFE_ROUND_DOWN**，目标尺寸同上，不做任何垫 0 处理
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        10000. & 1000. & 20100. & 2000. & 30200. & 3000. & 300. \\
        10. & 1. & 20.1 & 2. & 30.2 & 3. & 0.3 \\
        40000.01 & 4000.001 & 50400.02 & 5000.002 & 60500.03 & 6000.003 & 600.0003 \\
        40. & 4. & 50.4 & 5. & 60.5 & 6. & 0.6 \\
        70000.04 & 7000.004 & 80700.05 & 8000.005 & 90800.06 & 9000.006 & 900.0006 \\
        70. & 7. & 80.7 & 8. & 90.8 & 9. & 0.90000004 \\
        0.07 & 0.007 & 0.0807 & 0.008 & 0.09079999 & 0.009 & 0.0009
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### num_groups
```python
# 调整部分参数
cIn         = 2
numberGroup = 2
cOut        = 2
data        = np.arange(1,1+hIn*wIn,dtype=np.float32).reshape(1,hIn,wIn)
data        = np.concatenate([data,data],0)
window      = np.power(10,range(4,-5,-1),dtype=np.float32)
window      = np.concatenate([window,-window],0)
bias        = np.full(cOut, 0, dtype=np.float32)
```
```python
    deconv = network.add_deconvolution(inputTensor, cOut, (hW, wW), window, bias)                   # 替换部分
    deconv.num_groups = cOut                                                                        # 分组数，默认值为 1
    print("deconv->", deconv.get_output(0).shape)                                                   # (cOut, hIn+hW-1, wIn+wW-1)
```

+ 输出张量 (2, 4, 7)，其中指定 num_groupds=2，输入张量和反卷积核均在 C 维上被均分为 2 组，各自反卷积后再拼接到一起
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        10000. & 21000. & 32100. & 3200. & 300. \\
        40010. & 54021. & 65432.1 & 6503.2 & 600.3 \\
        70040.01 & 87054.02 & 98765.43 & 9806.503 & 900.6003 \\
        70.04 & 87.054 & 98.76539 & 9.8064995 & 0.9006 \\
        0.07 & 0.087 & 0.09869999 & 0.0098 & 0.0009
    \end{matrix}\right]
    \left[\begin{matrix}
        -10000. & -21000. & -32100. & -3200. & -300. \\
        -40010. & -54021. & -65432.1 & -6503.2 & -600.3 \\
        -70040.01 & -87054.02 & -98765.43 & -9806.503 & -900.6003 \\
        -70.04 & -87.054 & -98.76539 & -9.8064995 & -0.9006 \\
        -0.07 & -0.087 & -0.09869999 & -0.0098 & -0.0009
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### kernel_size_nd & stride_nd & padding_nd（add_deconvolution_nd）
```python
# 三维反卷积的例子，调整部分参数
cIn     = 2
cOut    = 1
data    = np.arange(1,1+hIn*wIn,dtype=np.float32).reshape(1,hIn,wIn)
data    = np.concatenate([data,data],0)
window  = np.power(10,range(4,-5,-1),dtype=np.float32)
window  = np.concatenate([window,-window],0)
bias    = np.full(cOut, 0, dtype=np.float32)
```
```python
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (1,cIn, hIn, wIn))           # 替换部分，要求输入张量维度是 4
    #...
    deconv = network.add_deconvolution_nd(inputTensor, cOut, (2, hW, wW), window, bias)             # 注意反卷积核是 3 维的
    deconv.kernel_size_nd = (2,hW,wW)                                                               # 反卷积核尺寸
    deconv.stride_nd      = (1, 1, 1)                                                               # 反卷积移动步长，默认值为 (1,1,1)
    deconv.padding_nd     = (0, 0, 0)                                                               # 四周垫 0 减少数，默认值为 (0,0,0)
    print("deconv->", deconv.get_output(0).shape)                                                   # (cOut, hIn+hW-1, wIn+wW-1)
```

+ 输出张量 (1, 3, 5, 5)，除了 C 维两端的两个反卷积结果外，C 维中间层的结果相当于两端的两个通道加在一起，得到了 0 的结果
$$
\left[\begin{matrix}
\left[\begin{matrix}
    \left[\begin{matrix}
        10000. & 21000. & 32100. & 3200. & 300. \\
        40010. & 54021. & 65432.1 & 6503.2 & 600.3 \\
        70040.01 & 87054.02 & 98765.43 & 9806.503 & 900.6003 \\
        70.04 & 87.054 & 98.76539 & 9.8064995 & 0.9006 \\
        0.07 & 0.087 & 0.09869999 & 0.0098 & 0.0009
    \end{matrix}\right]
    \left[\begin{matrix}
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0.
    \end{matrix}\right]
    \left[\begin{matrix}
        -10000. & -21000. & -32100. & -3200. & -300. \\
        -40010. & -54021. & -65432.1 & -6503.2 & -600.3 \\
        -70040.01 & -87054.02 & -98765.43 & -9806.503 & -900.6003 \\
        -70.04 & -87.054 & -98.76539 & -9.8064995 & -0.9006 \\
        -0.07 & -0.087 & -0.09869999 & -0.0098 & -0.0009
    \end{matrix}\right]
\end{matrix}\right]
\end{matrix}\right]
$$

<div style="page-break-after:always;"></div>
## Element Wise 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 4                                                                                         # 输入张量 HWC
wIn     = 5
cIn     = 3
data    = np.arange(cIn*hIn*wIn,dtype=np.float32).reshape(cIn,hIn,wIn)                              # 输入张量

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替换部分
    ele = network.add_elementwise(inputTensor,inputTensor,trt.ElementWiseOperation.SUM)
    print("ele->", ele.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(ele.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (3, 4, 5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0.  & 1.  & 2.  & 3.  & 4. \\
        5.  & 6.  & 7.  & 8.  & 9. \\
        10. & 11. & 12. & 13. & 14.\\
        15. & 16. & 17. & 18. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 21. & 22. & 23. & 24. \\
        25. & 26. & 27. & 28. & 29. \\
        30. & 31. & 32. & 33. & 34. \\
        35. & 36. & 37. & 38. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 41. & 42. & 43. & 44. \\
        45. & 46. & 47. & 48. & 49. \\
        50. & 51. & 52. & 53. & 54. \\
        55. & 56. & 57. & 58. & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (3, 4, 5)，每个元素变成原来的 2 倍
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0. &   2. &   4. &   6. &   8. \\
        10. &  12. &  14. &  16. &  18. \\
        20. &  22. &  24. &  26. &  28. \\
        30. &  32. &  34. &  36. &  38.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. &  42. &  44. &  46. &  48. \\
        50. &  52. &  54. &  56. &  58. \\
        60. &  62. &  64. &  66. &  68. \\
        70. &  72. &  74. &  76. &  78.
    \end{matrix}\right]
    \left[\begin{matrix}
        80. &  82. &  84. &  86. &  88. \\
        90. &  92. &  94. &  96. &  98. \\
        100. & 102. & 104. & 106. & 108. \\
        110. & 112. & 114. & 116. & 118. \\
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### op
```python
    ele = network.add_elementwise(inputTensor,inputTensor,trt.ElementWiseOperation.SUM)             # 替换部分
    ele.op = trt.ElementWiseOperation.SUB                                                           # 指定运算种类
    print("ele->", ele.get_output(0).shape)
```

+ 输出张量 (3, 4, 5)，每个元素变成 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0.
    \end{matrix}\right]
    \left[\begin{matrix}
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0.
    \end{matrix}\right]
    \left[\begin{matrix}
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 注意：可用的计算类型
| trt.ElementWiseOperation 名 | $f\left(a,b\right)$    |
| :-------------------------- | :--------------------- |
| EQUAL                       | a == b                 |
| DIV                         | a / b                  |
| SUB                         | a - b                  |
| POW                         | a \*\* b ($a^{b}$)     |
| LESS                        | a < b                  |
| OR                          | a or b                 |
| MIN                         | $\min\left(a,b\right)$ |
| FLOOR_DIV                   | a // b                 |
| GREATER                     | a > b                  |
| XOR                         | a ^ b (a xor b)        |
| MAX                         | $\max\left(a,b\right)$ |
| AND                         | a and b                |
| PROD                        | a * b                  |
| SUM                         | a + b                  |

<div style="page-break-after:always;"></div>
## Fill 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hOut     = 4                                                                                        # 输出张 HWC
wOut     = 5
cOut     = 3

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))      # 使用显式 batch 模式

    #-----------------------------------------------------------------------------------------------# 可替换部分
    fill = network.add_fill((cOut,hOut,wOut), trt.FillOperation.LINSPACE)
    print("fill->", fill.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(fill.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    out1_h  = np.empty(engine.get_binding_shape(0),dtype = trt.nptype(engine.get_binding_dtype(0)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
            
    context.execute_async(1, [int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 3, linewidth = 200, suppress = True)
    run()
```
+ 输出张量 (3, 4, 5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0.
    \end{matrix}\right]
    \left[\begin{matrix}
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0.
    \end{matrix}\right]
    \left[\begin{matrix}
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 注意：包含内部错误
> [TensorRT] INTERNAL ERROR: Assertion failed: dims.nbDims == 1 && "Alpha and beta tensor should be set when output an ND tensor" ../rtExt/cuda/cudaFillRunner.cpp:43

---
### 线性填充与 set_input
```python
    fill = network.add_fill((2,3,5), trt.FillOperation.LINSPACE)                                    # 替换部分，使用线性填充
    
    in0 = network.add_constant((4,),np.array([1,cOut,hOut,wOut],dtype=np.int32))                    # 形状张量，可覆盖函数 add_fill 的参数
    in1 = network.add_constant((),np.array([1000],dtype=np.float32))                                # 初值标量
    in2 = network.add_constant((4,),np.array([0,100,10,1],dtype=np.float32))                        # 增量张量，要求该一维张量的长度等于形状张量的维数
    
    fill.set_input(0,in0.get_output(0))
    fill.set_input(1,in1.get_output(0))
    fill.set_input(2,in2.get_output(0))
    print("fill->", fill.get_output(0).shape)
```

+ 输出张量 (3, 4, 5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1000. & 1001. & 1002. & 1003. & 1004. \\
        1010. & 1011. & 1012. & 1013. & 1014. \\
        1020. & 1021. & 1022. & 1023. & 1024. \\
        1030. & 1031. & 1032. & 1033. & 1034.
    \end{matrix}\right]
    \left[\begin{matrix}
        1100. & 1101. & 1102. & 1103. & 1104. \\
        1110. & 1111. & 1112. & 1113. & 1114. \\
        1120. & 1121. & 1122. & 1123. & 1124. \\
        1130. & 1131. & 1132. & 1133. & 1134.
    \end{matrix}\right]
    \left[\begin{matrix}
        1200. & 1201. & 1202. & 1203. & 1204. \\
        1210. & 1211. & 1212. & 1213. & 1214. \\
        1220. & 1221. & 1222. & 1223. & 1224. \\
        1230. & 1231. & 1232. & 1233. & 1234.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### 均匀随机填充与 set_input
```python
    fill = network.add_fill((7,11), trt.FillOperation.RANDOM_UNIFORM)                               # 替换部分，使用均匀随机数填充
    
    in0 = network.add_constant((4,),np.array([1,cOut,hOut,wOut],dtype=np.int32))                    # 形状张量，可覆盖函数 add_fill 的参数
    in1 = network.add_constant((),np.array([-10],dtype=np.float32))                                 # 最小值标量
    in2 = network.add_constant((),np.array([10],dtype=np.float32))                                  # 最大指标量
    
    fill.set_input(0,in0.get_output(0))
    fill.set_input(1,in1.get_output(0))
    fill.set_input(2,in2.get_output(0))
    print("fill->", fill.get_output(0).shape)
```

+ 输出张量 (3, 4, 5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        -9.167 &  0.371 & -7.195  & 2.565 &  9.814 \\
        -9.129 & -2.469 &  4.144  & 0.099 & -6.012 \\
         9.422 & -9.963 & -2.179  & 8.372 & -9.639 \\
         6.106 &  6.965 & -0.493  & 6.938 & -7.576
    \end{matrix}\right]
    \left[\begin{matrix}
         6.567 &  5.466 & -6.148 & -7.764 & -5.719 \\
         4.527 &  1.752 & -7.469 &  1.24  & -6.424 \\
        -9.2   &  3.142 &  9.268 &  9.176 & -6.118 \\
        -1.818 &  5.001 & -3.764 &  9.836 & -9.384
    \end{matrix}\right]
    \left[\begin{matrix}
        -6.398 &  7.669 & -6.942 & -7.131 &  8.463 \\
        -0.08  & -7.027 &  9.608 &  2.046 & -7.655 \\
         1.096 & -4.69  &  7.327 & -6.187 &  3.415 \\
        -5.887 & -2.402 & -6.263 & -1.868 & -4.79
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 注意：
  - 随机数种子固定，按相同形状和数值范围生成的随机数是相同的
  - 建成 engine 后其中数值为常数，多次运行数值均相同

<div style="page-break-after:always;"></div>
## Fully Connected 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn         = 4                                                                                     # 输入张量 HWC
wIn         = 5
cIn         = 3
cOut        = 2                                                                                     # 输出张量 C
data        = np.arange(cIn*hIn*wIn,dtype=np.float32).reshape(cIn,hIn,wIn)                          # 输入张量
weight      = np.ones(cIn*hIn*wIn), dtype=np.float32)                                               # 全连接权值
weight      = np.concatenate([weight, -weight],0)
bias        = np.zeros(cOut, dtype=np.float32)                                                      # 全连接偏置

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替换部分
    cf = network.add_fully_connected(inputTensor, cOut, weight, bias)
    print("fullCon(default)->", cf.get_output(0).shape)                                             # (cOut,1)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(cf.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (3, 4, 5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0. & 1. & 2. & 3. & 4. \\
        5. & 6. & 7. & 8. & 9. \\
        10. & 11. & 12. & 13. & 14. \\
        15. & 16. & 17. & 18. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 21. & 22. & 23. & 24. \\
        25. & 26. & 27. & 28. & 29. \\
        30. & 31. & 32. & 33. & 34. \\
        35. & 36. & 37. & 38. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 41. & 42. & 43. & 44. \\
        45. & 46. & 47. & 48. & 49. \\
        50. & 51. & 52. & 53. & 54. \\
        55. & 56. & 57. & 58. & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (2, 1, 1)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1770. \\ -1770.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：官方文档写的是 $output = X \cdot W^{T} + bias$
$$
\begin{aligned}
    output &= W.reshape(cOut,cIn*hIn*wIn) \cdot X.reshape(cIn*hIn*wIn,1) + bias\\
    &= \left[\begin{matrix} 1 & 1 & 1 & \cdots & 1 \\ -1 & -1 & -1 & \cdots & -1 \end{matrix}\right]
    \left[\begin{matrix} 0 \\ 1 \\ 2 \\ \vdots \\ 59 \end{matrix}\right] + \left[\begin{matrix} 0 \\ 0 \\ 0 \\ \vdots \\ 0 \end{matrix}\right]\\
    &= \left[\begin{matrix} 1770 \\ -1770 \end{matrix}\right]
\end{aligned}
$$

---
### num_output_channels & kernel& bias
```python
    fakeWeightAndBias = np.zeros(1, dtype=np.float32)                                               # 替换部分
    cf = network.add_fully_connected(inputTensor, 1, fakeWeightAndBias, fakeWeightAndBias)          # 先填入一些参数，后续再修改
    cf.num_output_channels = cOut                                                                   # 输出通道数，可覆盖函数 add_fully_connected 的参数
    cf.kernel = weight                                                                              # 全连接权值，可覆盖函数 add_fully_connected 的参数
    cf.bias = bias                                                                                  # 全连接偏置，可覆盖函数 add_fully_connected 的参数
    print("fullCon->", cf.get_output(0).shape)                                                      # (cOut,1)
```

+ 输出张量 (2, 1, 1)，与初始代码相同

$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1770. \\ -1770.
    \end{matrix}\right]
\end{matrix}\right]
$$

<div style="page-break-after:always;"></div>
## Gather 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 4                                                                                         # 输入张量 HWC
wIn     = 5
cIn     = 3
lenIndex= 3                                                                                         # 下标张量长度
data    = np.arange(cIn).reshape(cIn,1,1)*100 + np.arange(hIn).reshape(1,hIn,1)*10 + np.arange(wIn).reshape(1,1,wIn)
data    = data.astype(np.float32)                                                                   # 输入张量
index   = np.array([1,0,2],dtype=np.int32)                                                          # 下标张量

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    indexTensor = network.add_input('indexTensor', trt.DataType.INT32, (lenIndex,))
    print("inputTensor->", inputTensor.shape)

    #-----------------------------------------------------------------------------------------------# 可替换部分
    ga = network.add_gather(inputTensor, indexTensor, 0)
    print("ga->", ga.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(ga.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    in2_h   = np.ascontiguousarray(index.reshape(-1))
    in2_d   = cuda.mem_alloc(in2_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(2),dtype = trt.nptype(engine.get_binding_dtype(2)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    cuda.memcpy_htod_async(in2_d, in2_h, stream)
    context.execute_async(1, [int(in1_d), int(in2_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (3, 4, 5)，百位表示 C 维编号，十位表示 H 维编号，个位表示 W 维编号
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         0. &  1. &  2. &  3. &  4. \\
        10. & 11. & 12. & 13. & 14. \\
        20. & 21. & 22. & 23. & 24. \\
        30. & 31. & 32. & 33. & 34.
    \end{matrix}\right]
    \left[\begin{matrix}
        100. & 101. & 102. & 103. & 104. \\
        110. & 111. & 112. & 113. & 114. \\
        120. & 121. & 122. & 123. & 124. \\
        130. & 131. & 132. & 133. & 134.
    \end{matrix}\right]
    \left[\begin{matrix}
        200. & 201. & 202. & 203. & 204. \\
        210. & 211. & 212. & 213. & 214. \\
        220. & 221. & 222. & 223. & 224. \\
        230. & 231. & 232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (3, 4, 5)，在最高“非batch”维（C 维）上按照下标张量重排顺序
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        100. & 101. & 102. & 103. & 104. \\
        110. & 111. & 112. & 113. & 114. \\
        120. & 121. & 122. & 123. & 124. \\
        130. & 131. & 132. & 133. & 134.
    \end{matrix}\right]
    \left[\begin{matrix}
         0. &  1. &  2. &  3. &  4. \\
        10. & 11. & 12. & 13. & 14. \\
        20. & 21. & 22. & 23. & 24. \\
        30. & 31. & 32. & 33. & 34.
    \end{matrix}\right]
    \left[\begin{matrix}
        200. & 201. & 202. & 203. & 204. \\
        210. & 211. & 212. & 213. & 214. \\
        220. & 221. & 222. & 223. & 224. \\
        230. & 231. & 232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### axis
```python
    gatherAxis = 0                                                                                  # 替换部分
    ga = network.add_gather(inputTensor, indexTensor, 0)
    ga.axis = gatherAxis                                                                            # 重排轴号，默认值 0
    print("ga->", ga.get_output(0).shape)
```

+ 输出张量 (3, 4, 5)，其中指定 axis=0，在最高“非batch”维（C 维）上按照下标张量重排顺序，与初始代码相同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        100. & 101. & 102. & 103. & 104. \\
        110. & 111. & 112. & 113. & 114. \\
        120. & 121. & 122. & 123. & 124. \\
        130. & 131. & 132. & 133. & 134.
    \end{matrix}\right]
    \left[\begin{matrix}
         0. &  1. &  2. &  3. &  4. \\
        10. & 11. & 12. & 13. & 14. \\
        20. & 21. & 22. & 23. & 24. \\
        30. & 31. & 32. & 33. & 34.
    \end{matrix}\right]
    \left[\begin{matrix}
        200. & 201. & 202. & 203. & 204. \\
        210. & 211. & 212. & 213. & 214. \\
        220. & 221. & 222. & 223. & 224. \\
        230. & 231. & 232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (3, 3, 5)，其中指定 axis=1，在次高“非batch”维（H 维）上按照下标张量重排顺序
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         10. & 11. & 12. & 13. & 14. \\
          0. &  1. &  2. &  3. &  4. \\
         20. & 21. & 22. & 23. & 24. \\
    \end{matrix}\right]
    \left[\begin{matrix}
        110. & 111. & 112. & 113. & 114. \\
        100. & 101. & 102. & 103. & 104. \\
        120. & 121. & 122. & 123. & 124. \\
    \end{matrix}\right]
    \left[\begin{matrix}
        210. & 211. & 212. & 213. & 214. \\
        200. & 201. & 202. & 203. & 204. \\
        220. & 221. & 222. & 223. & 224. \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (3, 4, 3)，其中指定 axis=2，在季高“非batch”维（W 维）上按照下标张量重排顺序
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         1. &  0. &  2. \\
        11. & 10. & 12. \\
        21. & 20. & 22. \\
        31. & 30. & 32.
    \end{matrix}\right]
    \left[\begin{matrix}
        101. & 100. & 102. \\
        111. & 110. & 112. \\
        121. & 120. & 122. \\
        131. & 130. & 132.
    \end{matrix}\right]
    \left[\begin{matrix}
        201. & 200. & 202. \\
        211. & 210. & 212. \\
        221. & 220. & 222. \\
        231. & 230. & 232.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### num_elementwise_dims
```python
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))      # 使用显式 batch 模式

    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (1,cIn, hIn, wIn))           # 替换部分
    indexTensor = network.add_input('indexTensor', trt.DataType.INT32, (1,lenIndex,))
    ga = network.add_gather(inputTensor, indexTensor, 1)
    ga.num_elementwise_dims = 1                                                                     # 显式状态下的重排轴号，默认值为 0
    print("ga->", ga.get_output(0).shape)
```

+ 输出张量 (3, 4, 5)，其中指定 num_elementwise_dims = 1 ，显式 batch size 模式下第 1 轴为 C 维，结果与初始代码相同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        100. & 101. & 102. & 103. & 104. \\
        110. & 111. & 112. & 113. & 114. \\
        120. & 121. & 122. & 123. & 124. \\
        130. & 131. & 132. & 133. & 134.
    \end{matrix}\right]
    \left[\begin{matrix}
         0. &  1. &  2. &  3. &  4. \\
        10. & 11. & 12. & 13. & 14. \\
        20. & 21. & 22. & 23. & 24. \\
        30. & 31. & 32. & 33. & 34.
    \end{matrix}\right]
    \left[\begin{matrix}
        200. & 201. & 202. & 203. & 204. \\
        210. & 211. & 212. & 213. & 214. \\
        220. & 221. & 222. & 223. & 224. \\
        230. & 231. & 232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right]
$$

<div style="page-break-after:always;"></div>
## Identity 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 4                                                                                         # 输入张量 HWC
wIn     = 5
cIn     = 3
data    = np.arange(cIn*hIn*wIn,dtype=np.float32).reshape(cIn,hIn,wIn)                              # 输入张量

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替换部分
    id = network.add_identity(inputTensor)
    print("id->", id.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(id.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (3, 4, 5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         0. &  1. &  2. &  3. &  4. \\
         5. &  6. &  7. &  8. &  9. \\
        10. & 11. & 12. & 13. & 14. \\
        15. & 16. & 17. & 18. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 21. & 22. & 23. & 24. \\
        25. & 26. & 27. & 28. & 29. \\
        30. & 31. & 32. & 33. & 34. \\
        35. & 36. & 37. & 38. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 41. & 42. & 43. & 44. \\
        45. & 46. & 47. & 48. & 49. \\
        50. & 51. & 52. & 53. & 54. \\
        55. & 56. & 57. & 58. & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (3, 4, 5)，与输入张量一模一样
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         0. &  1. &  2. &  3. &  4. \\
         5. &  6. &  7. &  8. &  9. \\
        10. & 11. & 12. & 13. & 14. \\
        15. & 16. & 17. & 18. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 21. & 22. & 23. & 24. \\
        25. & 26. & 27. & 28. & 29. \\
        30. & 31. & 32. & 33. & 34. \\
        35. & 36. & 37. & 38. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 41. & 42. & 43. & 44. \\
        45. & 46. & 47. & 48. & 49. \\
        50. & 51. & 52. & 53. & 54. \\
        55. & 56. & 57. & 58. & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 注意，identity 常用于转换精度，以及 iterator 层（见“基于 Loop 的 RNN”部分）

<div style="page-break-after:always;"></div>
## Loop 结构

### 初始代码，for 型循环，两种输出
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 4                                                                                         # 输入张量 HWC
wIn     = 5
cIn     = 3
data    = np.ones(cIn*hIn*wIn,dtype=np.float32).reshape(cIn,hIn,wIn)                                # 输入张量

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))      # 使用显式 batch 模式
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (1, cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)

    #-----------------------------------------------------------------------------------------------# 可替换部分
    loop    = network.add_loop()                                                                    # 仿照 "int m=1,temp=m;for(int i=0;i<t;i++){temp+=temp};m=temp;"
    
    t       = 10                                                                                    # 循环次数
    limit   = network.add_constant((),np.array([t],dtype=np.int32))
    loop.add_trip_limit(limit.get_output(0),trt.TripLimit.COUNT)                                    # 指定 for 型循环，"要么执行 i=0，要么执行 i++，然后判断 i < t"

    rLayer  = loop.add_recurrence(inputTensor)                                                      # 循环入口，"temp = m"

    h1 = network.add_elementwise(rLayer.get_output(0),rLayer.get_output(0),trt.ElementWiseOperation.SUM)    # 循环体，"temp += temp"

    rLayer.set_input(1,h1.get_output(0))                                                            # 返回循环头，rLayer 的第 0 输入是 inputTensor，h1 是第 1 输入

    loopOutput1 = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)          # 第一种循环输出，只要最终结果，"m = temp"
            
    loopOutput2 = loop.add_loop_output(h1.get_output(0), trt.LoopOutput.CONCATENATE, 0)             # 第二种循环输出，保留所有中间结果，若这里不是传入 h1 而是传入 rLayer，则输出“包含原始输入而不包含最后一次迭代结果”
    loopOutput2.set_input(1,limit.get_output(0))                                                    # 时间长度张量交给 loopOutput2，若这里传入张量的值 v < t，则结果保留前 v 次迭代，若 v > t，则尾部用 0 填充

    print("loop->", loopOutput1.get_output(0).shape, loopOutput2.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(loopOutput1.get_output(0))
    network.mark_output(loopOutput2.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
    out2_h  = np.empty(engine.get_binding_shape(2),dtype = trt.nptype(engine.get_binding_dtype(2)))
    out2_d  = cuda.mem_alloc(out2_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d), int(out2_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    cuda.memcpy_dtoh_async(out2_h, out2_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    print("out2_h:", out2_h.shape)
    print(out2_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 3, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (1, 3, 4, 5)
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

+ 输出张量 1 形状 (10, 1, 3, 4, 5)，（实际是 loopOutput2），在输入张量的最前面增加一维，将每次迭代的输出依次放入
$$
\left[\begin{matrix}
    \left[\begin{matrix}
    \left[\begin{matrix}
    \left[\begin{matrix}
          2. &   2. &   2. &   2. &   2. \\
          2. &   2. &   2. &   2. &   2. \\
          2. &   2. &   2. &   2. &   2. \\
          2. &   2. &   2. &   2. &   2.
    \end{matrix}\right]
    \left[\begin{matrix}
          2. &   2. &   2. &   2. &   2. \\
          2. &   2. &   2. &   2. &   2. \\
          2. &   2. &   2. &   2. &   2. \\
          2. &   2. &   2. &   2. &   2.
    \end{matrix}\right]
    \left[\begin{matrix}
          2. &   2. &   2. &   2. &   2. \\
          2. &   2. &   2. &   2. &   2. \\
          2. &   2. &   2. &   2. &   2. \\
          2. &   2. &   2. &   2. &   2.
    \end{matrix}\right]
    \end{matrix}\right]
    \end{matrix}\right]\\
    \left[\begin{matrix}
    \left[\begin{matrix}
    \left[\begin{matrix}
          4. &   4. &   4. &   4. &   4. \\
          4. &   4. &   4. &   4. &   4. \\
          4. &   4. &   4. &   4. &   4. \\
          4. &   4. &   4. &   4. &   4.
    \end{matrix}\right]
    \left[\begin{matrix}
          4. &   4. &   4. &   4. &   4. \\
          4. &   4. &   4. &   4. &   4. \\
          4. &   4. &   4. &   4. &   4. \\
          4. &   4. &   4. &   4. &   4.
    \end{matrix}\right]
    \left[\begin{matrix}
          4. &   4. &   4. &   4. &   4. \\
          4. &   4. &   4. &   4. &   4. \\
          4. &   4. &   4. &   4. &   4. \\
          4. &   4. &   4. &   4. &   4.
    \end{matrix}\right]
    \end{matrix}\right]
    \end{matrix}\right]\\
	\mathbf{ \cdots }\\
    \left[\begin{matrix}
    \left[\begin{matrix}
    \left[\begin{matrix}
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512.
    \end{matrix}\right]
    \left[\begin{matrix}
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512.
    \end{matrix}\right]
    \left[\begin{matrix}
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512.
    \end{matrix}\right]
    \end{matrix}\right]
    \end{matrix}\right]\\
    \left[\begin{matrix}
    \left[\begin{matrix}
    \left[\begin{matrix}
        1024. & 1024. & 1024. & 1024. & 1024. \\
        1024. & 1024. & 1024. & 1024. & 1024. \\
        1024. & 1024. & 1024. & 1024. & 1024. \\
        1024. & 1024. & 1024. & 1024. & 1024.
    \end{matrix}\right]
    \left[\begin{matrix}
        1024. & 1024. & 1024. & 1024. & 1024. \\
        1024. & 1024. & 1024. & 1024. & 1024. \\
        1024. & 1024. & 1024. & 1024. & 1024. \\
        1024. & 1024. & 1024. & 1024. & 1024.
    \end{matrix}\right]
    \left[\begin{matrix}
        1024. & 1024. & 1024. & 1024. & 1024. \\
        1024. & 1024. & 1024. & 1024. & 1024. \\
        1024. & 1024. & 1024. & 1024. & 1024. \\
        1024. & 1024. & 1024. & 1024. & 1024.
    \end{matrix}\right]
    \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 2 形状 (1, 3, 4, 5)，（实际是 loopOutput2），循环最终的结果
$$
\left[\begin{matrix}
\left[\begin{matrix}
    \left[\begin{matrix}
        1024. & 1024. & 1024. & 1024. & 1024. \\
        1024. & 1024. & 1024. & 1024. & 1024. \\
        1024. & 1024. & 1024. & 1024. & 1024. \\
        1024. & 1024. & 1024. & 1024. & 1024.
    \end{matrix}\right]
    \left[\begin{matrix}
        1024. & 1024. & 1024. & 1024. & 1024. \\
        1024. & 1024. & 1024. & 1024. & 1024. \\
        1024. & 1024. & 1024. & 1024. & 1024. \\
        1024. & 1024. & 1024. & 1024. & 1024.
    \end{matrix}\right]
    \left[\begin{matrix}
        1024. & 1024. & 1024. & 1024. & 1024. \\
        1024. & 1024. & 1024. & 1024. & 1024. \\
        1024. & 1024. & 1024. & 1024. & 1024. \\
        1024. & 1024. & 1024. & 1024. & 1024.
    \end{matrix}\right]
\end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：
|    t     |  继续循环？  | rLayer 元素值 |   h1 元素值   | rLayer 更新元素值 |
| :------: | :----------: | :-----------: | :-----------: | :---------------: |
|    0     | $\checkmark$ |       1       |       2       |         2         |
|    1     | $\checkmark$ |       2       |       4       |         4         |
|    2     | $\checkmark$ |       4       |       8       |         8         |
| $\cdots$ |   $\cdots$   |   $\cdots$    |   $\cdots$    |     $\cdots$      |
|    9     | $\checkmark$ |      512      |     1024      |       1024        |
|    10    |   $\times$   |   输出 1024   | 输出 2 ~ 1024 |                   |

+ 注意，LAST_VALUE 和 CONCATENATE 两种输出，可以只使用其中一个或两者同时使用（需要标记为两个不同的 loopOutput 层）
+ 注意，无 iterator 的循环不能将结果 CONCATENATE 到其他维度，也不能使用 REVERSE 输出，否则报错：
> [TensorRT] ERROR: ../builder/myelin/codeGenerator.cpp (338) - Myelin Error in compileGraph: 64 (myelinProgramAnalysisError : No scan input/output tensor to derive shape for tensor(Unnamed Loop_ 0)_U0^(Unnamed Layer_ 5) [LoopOutput] copy

---
### while 型循环，两种输出
```python
    loop = network.add_loop()                                                                       # 替换部分，仿照 "int m=1,temp=m;{m=temp;temp+=temp;}while{temp!=1024}"
    
    rLayer = loop.add_recurrence(inputTensor)                                                       # 循环头部，"temp = m"

    h1 = network.add_elementwise(rLayer.get_output(0),rLayer.get_output(0),trt.ElementWiseOperation.SUM)    # 循环体，"temp += temp"
    
    rLayer.set_input(1,h1.get_output(0))                                                            # 返回循环头

    w    = np.zeros(cIn*hIn*wIn,dtype=np.float32)
    w[0] = 1
    b    = np.zeros(1,dtype=np.float32)
    c0 = network.add_fully_connected(h1.get_output(0),1,w,b)                                        # 获取 h1 的第一个元素，调整为 BOOL 型标量以便判断循环条件
    c1 = network.add_shuffle(c0.get_output(0))
    c1.reshape_dims = ()
    c_ = network.add_constant((),np.array([1024],dtype=np.float32))
    c2 = network.add_elementwise(c1.get_output(0),c_.get_output(0),trt.ElementWiseOperation.SUB)
    c3 = network.add_identity(c2.get_output(0))
    c3.set_output_type(0,trt.DataType.BOOL)
    #print(c3.get_output(0).shape,c3.get_output_type(0))                                            # 标量，BOOL 型
        
    loop.add_trip_limit(c3.get_output(0),trt.TripLimit.WHILE)                                       # 判断 temp!=1024，当条件为 False 时退出循环
    
    loopOutput1 = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)          # 第一种循环输出，只要最终结果的 "m"
    
    loopOutput2 = loop.add_loop_output(h1.get_output(0), trt.LoopOutput.CONCATENATE, 0)             # 第二种循环输出，保留所有中间结果，注意传入 h1 而不是 rLayer
    length = network.add_constant((),np.array([10],dtype=np.int32))                                 # 需要保留的循环次数，这里仅循环 9 次，超出部分会被 0 填充
    loopOutput2.set_input(1,length.get_output(0))

    print("loop->", loopOutput1.get_output(0).shape, loopOutput2.get_output(0).shape)
```

+ 输出张量 1 形状 (10, 1, 3, 4, 5)，（实际是 loopOutput2），在输入张量的最前面增加一维，将每次迭代的输出依次放入
$$
\left[\begin{matrix}
    \left[\begin{matrix}
    \left[\begin{matrix}
    \left[\begin{matrix}
          2. &   2. &   2. &   2. &   2. \\
          2. &   2. &   2. &   2. &   2. \\
          2. &   2. &   2. &   2. &   2. \\
          2. &   2. &   2. &   2. &   2.
    \end{matrix}\right]
    \left[\begin{matrix}
          2. &   2. &   2. &   2. &   2. \\
          2. &   2. &   2. &   2. &   2. \\
          2. &   2. &   2. &   2. &   2. \\
          2. &   2. &   2. &   2. &   2.
    \end{matrix}\right]
    \left[\begin{matrix}
          2. &   2. &   2. &   2. &   2. \\
          2. &   2. &   2. &   2. &   2. \\
          2. &   2. &   2. &   2. &   2. \\
          2. &   2. &   2. &   2. &   2.
    \end{matrix}\right]
    \end{matrix}\right]
    \end{matrix}\right]\\
    \left[\begin{matrix}
    \left[\begin{matrix}
    \left[\begin{matrix}
          4. &   4. &   4. &   4. &   4. \\
          4. &   4. &   4. &   4. &   4. \\
          4. &   4. &   4. &   4. &   4. \\
          4. &   4. &   4. &   4. &   4.
    \end{matrix}\right]
    \left[\begin{matrix}
          4. &   4. &   4. &   4. &   4. \\
          4. &   4. &   4. &   4. &   4. \\
          4. &   4. &   4. &   4. &   4. \\
          4. &   4. &   4. &   4. &   4.
    \end{matrix}\right]
    \left[\begin{matrix}
          4. &   4. &   4. &   4. &   4. \\
          4. &   4. &   4. &   4. &   4. \\
          4. &   4. &   4. &   4. &   4. \\
          4. &   4. &   4. &   4. &   4.
    \end{matrix}\right]
    \end{matrix}\right]
    \end{matrix}\right]\\
	\mathbf{ \cdots }\\
    \left[\begin{matrix}
    \left[\begin{matrix}
    \left[\begin{matrix}
        256. & 256. & 256. & 256. & 256. \\
        256. & 256. & 256. & 256. & 256. \\
        256. & 256. & 256. & 256. & 256. \\
        256. & 256. & 256. & 256. & 256.
    \end{matrix}\right]
    \left[\begin{matrix}
        256. & 256. & 256. & 256. & 256. \\
        256. & 256. & 256. & 256. & 256. \\
        256. & 256. & 256. & 256. & 256. \\
        256. & 256. & 256. & 256. & 256.
    \end{matrix}\right]
    \left[\begin{matrix}
        256. & 256. & 256. & 256. & 256. \\
        256. & 256. & 256. & 256. & 256. \\
        256. & 256. & 256. & 256. & 256. \\
        256. & 256. & 256. & 256. & 256.
    \end{matrix}\right]
    \end{matrix}\right]
    \end{matrix}\right]\\
    \left[\begin{matrix}
    \left[\begin{matrix}
    \left[\begin{matrix}
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512.
    \end{matrix}\right]
    \left[\begin{matrix}
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512.
    \end{matrix}\right]
    \left[\begin{matrix}
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512.
    \end{matrix}\right]
    \end{matrix}\right]
    \end{matrix}\right]\\
    \left[\begin{matrix}
    \left[\begin{matrix}
    \left[\begin{matrix}
          0. &   0. &   0. &   0. &   0. \\
          0. &   0. &   0. &   0. &   0. \\
          0. &   0. &   0. &   0. &   0. \\
          0. &   0. &   0. &   0. &   0.
    \end{matrix}\right]
    \left[\begin{matrix}
          0. &   0. &   0. &   0. &   0. \\
          0. &   0. &   0. &   0. &   0. \\
          0. &   0. &   0. &   0. &   0. \\
          0. &   0. &   0. &   0. &   0.
    \end{matrix}\right]
    \left[\begin{matrix}
          0. &   0. &   0. &   0. &   0. \\
          0. &   0. &   0. &   0. &   0. \\
          0. &   0. &   0. &   0. &   0. \\
          0. &   0. &   0. &   0. &   0.
    \end{matrix}\right]
    \end{matrix}\right]
    \end{matrix}\right]\\
\end{matrix}\right]
$$

+ 输出张量 2 形状 (1, 3, 4, 5)，（实际是 loopOutput1），循环最终的 rLayer 结果
$$
\left[\begin{matrix}
\left[\begin{matrix}
    \left[\begin{matrix}
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512.
    \end{matrix}\right]
    \left[\begin{matrix}
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512.
    \end{matrix}\right]
    \left[\begin{matrix}
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512. \\
        512. & 512. & 512. & 512. & 512.
    \end{matrix}\right]
\end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：
| rLayer 元素值 |  h1 元素值   |  c3 值   |  继续循环？  |
| :-----------: | :----------: | :------: | :----------: |
|       1       |      2       |    1     | $\checkmark$ |
|       2       |      4       |    1     | $\checkmark$ |
|       4       |      8       |    1     | $\checkmark$ |
|   $\cdots$    |   $\cdots$   | $\cdots$ |   $\cdots$   |
|      512      |     1024     |    0     |   $\times$   |
|   输出 512    | 输出 2 ~ 512 |          |              |

+ 注意，可用的循环类型
| tensorrt.TripLimit 名 | 说明                     |
| :-------------------- | :----------------------- |
| COUNT                 | for 型循环，给定循环次数 |
| WHILE                 | while 型循环             |

---
### while 型循环 + slice 层的 bug
```python
    loop = network.add_loop()
    
    rLayer = loop.add_recurrence(inputTensor)

    h1 = network.add_elementwise(rLayer.get_output(0),rLayer.get_output(0),trt.ElementWiseOperation.SUM)
    
    rLayer.set_input(1,h1.get_output(0))

    c0 = network.add_slice(h1.get_output(0),(0,0,0,0),(1,1,1,1),(1,1,1,1))                          # 使用 slice 而不是全连接来取第一个元素
    c1 = network.add_shuffle(c0.get_output(0))
    c1.reshape_dims = ()
    c_ = network.add_constant((),np.array([1024],dtype=np.float32))
    c2 = network.add_elementwise(c1.get_output(0),c_.get_output(0),trt.ElementWiseOperation.SUB)
    c3 = network.add_identity(c2.get_output(0))
    c3.set_output_type(0,trt.DataType.BOOL)
        
    loop.add_trip_limit(c3.get_output(0),trt.TripLimit.WHILE)
    
    loopOutput1 = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)
    
    loopOutput2 = loop.add_loop_output(rLayer.get_output(0), h1.LoopOutput.CONCATENATE, 0)
    length = network.add_constant((),np.array([10],dtype=np.int32))
    loopOutput2.set_input(1,length.get_output(0))

    print("loop->", loopOutput1.get_output(0).shape, loopOutput2.get_output(0).shape)
```

+ 结果：数组第一个元素为剩余元素的平方，仅循环 5 次便退出，原因是 slice 层没有新建张量，而是取了原张量的引用，对其计算会影响原张量中的值

---
### iterator 迭代层
```python
# iterator 部分所有代码
data    = np.ones(cIn*hIn*wIn).reshape(cIn,hIn,wIn)*np.arange(1,1+cIn).reshape(cIn,1,1)             # 输入张量
data    = data.astype(np.float32)                                                                   # 不明原因，合并两行 data 会导致错误
```
```python
    loop = network.add_loop()                                                                       # 替换部分

    it = loop.add_iterator(inputTensor, 1, False)                                                   # 用 inputTensor 制造一个迭代器，在 C 维上每次正向抛出 1 层 (1,hIn,wIn)
    it.axis = 1                                                                                     # 指定抛出的轴号，从 batch 维为 0 开始往右递增，可覆盖函数 add_iterator 的参数
    print(it.reverse)                                                                               # 是否反序抛出（见后面样例），参数只用于输出，不可修改

    limit = network.add_constant((),np.array([cIn],dtype=np.int32))
    loop.add_trip_limit(limit.get_output(0),trt.TripLimit.COUNT)
    
    h0 = network.add_constant([1,hIn,wIn],np.zeros(hIn*wIn,dtype=np.float32))                       # 存储中间结果的张量，必须在循环外初始化好
    rLayer = loop.add_recurrence(h0.get_output(0))                                                  # rLayer 的第 0 输入要求来自循环外
    
    h1 = network.add_elementwise(it.get_output(0),rLayer.get_output(0),trt.ElementWiseOperation.SUM)

    rLayer.set_input(1, h1.get_output(0))                                                           # 循环内层计算结果作为 rLayer 第 1 输入

    loopOutput1 = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)          # 只保留最后输出，index 参数被忽略

    loopOutput2 = loop.add_loop_output(h1.get_output(0), trt.LoopOutput.CONCATENATE, 0)             # 保留所有中间输出，index 可以使用其他参数（例子见后面）
    length = network.add_constant((),np.array([cIn],dtype=np.int32))
    loopOutput2.set_input(1,length.get_output(0))

    print("loop->", loopOutput1.get_output(0).shape, loopOutput2.get_output(0).shape)
```

+ 输入张量 (3, 4, 5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
    \left[\begin{matrix}
        2. & 2. & 2. & 2. & 2. \\
        2. & 2. & 2. & 2. & 2. \\
        2. & 2. & 2. & 2. & 2. \\
        2. & 2. & 2. & 2. & 2.
    \end{matrix}\right]
    \left[\begin{matrix}
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (3, 1, 4, 5)，（实际是 loopOutput2），先加 1 再加 2 再加 3
$$
\left[\begin{matrix}
    \left[\begin{matrix}
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
    \end{matrix}\right]
    \left[\begin{matrix}
    \left[\begin{matrix}
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3.
    \end{matrix}\right]
    \end{matrix}\right]
    \left[\begin{matrix}
    \left[\begin{matrix}
        6. & 6. & 6. & 6. & 6. \\
        6. & 6. & 6. & 6. & 6. \\
        6. & 6. & 6. & 6. & 6. \\
        6. & 6. & 6. & 6. & 6.
    \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 2 形状 (1, 4, 5)，（实际是 loopOutput1）
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        6. & 6. & 6. & 6. & 6. \\
        6. & 6. & 6. & 6. & 6. \\
        6. & 6. & 6. & 6. & 6. \\
        6. & 6. & 6. & 6. & 6.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 使用 REVERSE 模式（将 CONCATENATE 换成 REVERSE）形状 (3, 1, 4, 5)，相当于将 CONCATENATE 的结果在最高维上倒序
$$
\left[\begin{matrix}
    \left[\begin{matrix}
    \left[\begin{matrix}
        6. & 6. & 6. & 6. & 6. \\
        6. & 6. & 6. & 6. & 6. \\
        6. & 6. & 6. & 6. & 6. \\
        6. & 6. & 6. & 6. & 6.
    \end{matrix}\right]
    \end{matrix}\right]
    \left[\begin{matrix}
    \left[\begin{matrix}
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3.
    \end{matrix}\right]
    \end{matrix}\right]
    \left[\begin{matrix}
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
    \end{matrix}\right]

\end{matrix}\right]
$$

+ 注意，可用的输出类型
| tensorrt.LoopOutput 名 | 说明                                         |
| :--------------------- | :------------------------------------------- |
| LAST_VALUE             | 仅保留最后一个输出                           |
| CONCATENATE            | 保留指定长度的中间输出（从第一次循环向后）   |
| REVERSE                | 保留执行长度的中间输出（从最后一次循环向前） |

```python
    loop = network.add_loop()                                                                       # 替换部分，迭代器在 H 维上抛出

    it = loop.add_iterator(inputTensor, 2, False)                                                   # index 改成 2

    limit = network.add_constant((),np.array([hIn],dtype=np.int32))                                 # 循环次数变为 hIn
    loop.add_trip_limit(limit.get_output(0),trt.TripLimit.COUNT)
    
    h0 = network.add_constant([1,cIn,wIn],np.zeros(cIn*wIn,dtype=np.float32))                       # 中间结果张量 尺寸变为 [1,cIn,wIn]
    rLayer = loop.add_recurrence(h0.get_output(0))
    
    h1 = network.add_elementwise(it.get_output(0),rLayer.get_output(0),trt.ElementWiseOperation.SUM)

    rLayer.set_input(1, h1.get_output(0))

    loopOutput1 = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)

    loopOutput2 = loop.add_loop_output(h1.get_output(0), trt.LoopOutput.CONCATENATE, 0)
    length = network.add_constant((),np.array([hIn],dtype=np.int32))                                # 保存长度变为 hIn
    loopOutput2.set_input(1,length.get_output(0))

    print("loop->", loopOutput1.get_output(0).shape, loopOutput2.get_output(0).shape)
```

+ 输出张量 1 形状 (4, 1, 3, 5)，（实际是 loopOutput2），REVERSE 型不再展示
$$
\left[\begin{matrix}
    \left[\begin{matrix}
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. \\
        2. & 2. & 2. & 2. & 2. \\
        3. & 3. & 3. & 3. & 3.
    \end{matrix}\right]
    \end{matrix}\right]
    \left[\begin{matrix}
    \left[\begin{matrix}
        2. & 2. & 2. & 2. & 2. \\
        4. & 4. & 4. & 4. & 4. \\
        6. & 6. & 6. & 6. & 6.
    \end{matrix}\right]
    \end{matrix}\right]
    \left[\begin{matrix}
    \left[\begin{matrix}
        3. & 3. & 3. & 3. & 3. \\
        6. & 6. & 6. & 6. & 6. \\
        9. & 9. & 9. & 9. & 9.
    \end{matrix}\right]
    \end{matrix}\right]
    \left[\begin{matrix}
    \left[\begin{matrix}
        4. &  4. &  4. &  4. &  4. \\
        8. &  8. &  8. &  8. &  8. \\
       12. & 12. & 12. & 12. & 12.
    \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 2 形状 (1, 3, 5)，（实际是 loopOutput1）
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         4. &  4. &  4. &  4. &  4. \\
         8. &  8. &  8. &  8. &  8. \\
        12. & 12. & 12. & 12. & 12.
    \end{matrix}\right]
\end{matrix}\right]
$$
```python
    loop = network.add_loop()                                                                       # 替换部分，迭代器在 W 维上抛

    it = loop.add_iterator(inputTensor, 3, False)                                                   # index 改成 3

    limit = network.add_constant((),np.array([wIn],dtype=np.int32))                                 # 循环次数变为 wIn
    loop.add_trip_limit(limit.get_output(0),trt.TripLimit.COUNT)
    
    h0 = network.add_constant([1,cIn,hIn],np.zeros(cIn*hIn,dtype=np.float32))                       # 中间结果张量 尺寸变为 [1,cIn,hIn]
    rLayer = loop.add_recurrence(h0.get_output(0))
    
    h1 = network.add_elementwise(it.get_output(0),rLayer.get_output(0),trt.ElementWiseOperation.SUM)

    rLayer.set_input(1, h1.get_output(0))

    loopOutput1 = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)

    loopOutput2 = loop.add_loop_output(h1.get_output(0), trt.LoopOutput.CONCATENATE, 0)
    length = network.add_constant((),np.array([wIn],dtype=np.int32))                                # 保存长度变为 wIn
    loopOutput2.set_input(1,length.get_output(0))

    print("loop->", loopOutput1.get_output(0).shape, loopOutput2.get_output(0).shape)
```

+ 输出张量 1 形状 (5, 1, 3, 4)，（实际是 loopOutput2）
$$
\left[\begin{matrix}
    \left[\begin{matrix}
    \left[\begin{matrix}
         1. &  1. &  1. &  1. \\
         2. &  2. &  2. &  2. \\
         3. &  3. &  3. &  3.
    \end{matrix}\right]
    \end{matrix}\right]
    \left[\begin{matrix}
    \left[\begin{matrix}
         2. &  2. &  2. &  2. \\
         4. &  4. &  4. &  4. \\
         6. &  6. &  6. &  6.
    \end{matrix}\right]
    \end{matrix}\right]
    \left[\begin{matrix}
    \left[\begin{matrix}
         3. &  3. &  3. &  3. \\
         6. &  6. &  6. &  6. \\
         9. &  9. &  9. &  9.
    \end{matrix}\right]
    \end{matrix}\right]
    \left[\begin{matrix}
    \left[\begin{matrix}
         4. &  4. &  4. &  4. \\
         8. &  8. &  8. &  8. \\
        12. & 12. & 12. & 12.
    \end{matrix}\right]
    \end{matrix}\right]
    \left[\begin{matrix}
    \left[\begin{matrix}
         5. &  5. &  5. &  5. \\
        10. & 10. & 10. & 10. \\
        15. & 15. & 15. & 15.
    \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 2 形状 (1, 3, 4)，（实际是 loopOutput1）
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         5. &  5. &  5. &  5. \\
        10. & 10. & 10. & 10. \\
        15. & 15. & 15. & 15.
    \end{matrix}\right]
\end{matrix}\right]
$$

```python
    loop = network.add_loop()                                                                       # 替换部分

    it = loop.add_iterator(inputTensor, 1, True)                                                    # 在 C 维上使用反序抛出

    limit = network.add_constant((),np.array([cIn],dtype=np.int32))
    loop.add_trip_limit(limit.get_output(0),trt.TripLimit.COUNT)
    
    h0 = network.add_constant([1,hIn,wIn],np.zeros(hIn*wIn,dtype=np.float32))                       # 存储中间结果的张量，必须在循环外初始化好
    rLayer = loop.add_recurrence(h0.get_output(0))                                                  # rLayer 的第 0 输入要求来自循环外
    
    h1 = network.add_elementwise(it.get_output(0),rLayer.get_output(0),trt.ElementWiseOperation.SUM)

    rLayer.set_input(1, h1.get_output(0))                                                           # 循环内层计算结果作为 rLayer 第 1 输入

    loopOutput1 = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)

    loopOutput2 = loop.add_loop_output(h1.get_output(0), trt.LoopOutput.CONCATENATE, 0)
    length = network.add_constant((),np.array([cIn],dtype=np.int32))
    loopOutput2.set_input(1,length.get_output(0))

    print("loop->", loopOutput1.get_output(0).shape, loopOutput2.get_output(0).shape)
```

+ 输出张量 1 形状 (3, 1, 4, 5)，（实际是 loopOutput2），先加了 3 再加 2 最后加 1
$$
\left[\begin{matrix}
    \left[\begin{matrix}
    \left[\begin{matrix}
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3.
    \end{matrix}\right]
    \end{matrix}\right]
    \left[\begin{matrix}
    \left[\begin{matrix}
        5. & 5. & 5. & 5. & 5. \\
        5. & 5. & 5. & 5. & 5. \\
        5. & 5. & 5. & 5. & 5. \\
        5. & 5. & 5. & 5. & 5.
    \end{matrix}\right]
    \end{matrix}\right]
    \left[\begin{matrix}
    \left[\begin{matrix}
        6. & 6. & 6. & 6. & 6. \\
        6. & 6. & 6. & 6. & 6. \\
        6. & 6. & 6. & 6. & 6. \\
        6. & 6. & 6. & 6. & 6.
    \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 2 形状 (1, 3, 4)，（实际是 loopOutput1），最终结果与正序相同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        6. & 6. & 6. & 6. & 6. \\
        6. & 6. & 6. & 6. & 6. \\
        6. & 6. & 6. & 6. & 6. \\
        6. & 6. & 6. & 6. & 6.
    \end{matrix}\right]
\end{matrix}\right]
$$
```python
    loop = network.add_loop()                                                                       # 替换部分，CONCATENATE 到其他维度上，其他部分与 iterator 的初始代码相同

    it = loop.add_iterator(inputTensor, 1, False)

    limit = network.add_constant((),np.array([cIn],dtype=np.int32))
    loop.add_trip_limit(limit.get_output(0),trt.TripLimit.COUNT)
    
    h0 = network.add_constant([1,hIn,wIn],np.zeros(hIn*wIn,dtype=np.float32))
    rLayer = loop.add_recurrence(h0.get_output(0))
    
    h1 = network.add_elementwise(it.get_output(0),rLayer.get_output(0),trt.ElementWiseOperation.SUM)

    rLayer.set_input(1, h1.get_output(0))

    length = network.add_constant((),np.array([cIn],dtype=np.int32))
    loopOutput1 = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.CONCATENATE, 1)         # LAST_VALUE 模式 index 参数被忽略，这里仅展示 CONTENATE 和 REVERSE
    loopOutput1.set_input(1,length.get_output(0))
    loopOutput2 = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.REVERSE, 1)
    loopOutput2.set_input(1,length.get_output(0))

    print("loop->", loopOutput1.get_output(0).shape, loopOutput2.get_output(0).shape)
```

+ 输出张量 1 形状 (1, 3, 4, 5)，（实际是 loopOutput2），在第 1 维上进行连接
$$
\left[\begin{matrix}
\left[\begin{matrix}
    \left[\begin{matrix}
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3.
    \end{matrix}\right]
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
    \left[\begin{matrix}
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0.
    \end{matrix}\right]
\end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (1, 3, 4, 5)，（实际是 loopOutput1），在第 1 维上进行连接
$$
\left[\begin{matrix}
\left[\begin{matrix}
    \left[\begin{matrix}
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0.
    \end{matrix}\right]
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
    \left[\begin{matrix}
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3.
    \end{matrix}\right]
\end{matrix}\right]
\end{matrix}\right]
$$

<div style="page-break-after:always;"></div>
## 基于 Loop 的 RNN

### 简单的 ReLU RNN，所有输入输出数据与”RNN 层“ 保持一致，只有实现不同
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 4                                                                                         # 输入张量 HWC
wIn     = 7
cIn     = 3
lenH    = 5                                                                                         # 隐藏层元素宽度
data    = np.ones(cIn*hIn*wIn,dtype=np.float32).reshape(cIn,hIn,wIn)                                # 输入张量
weight  = np.ones((wIn+lenH,lenH),dtype=np.float32)                                                 # RNN 变换阵（TensorFlow 格式）
bias    = np.zeros(lenH, dtype=np.float32)                                                          # RNN 偏置

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network(1<<0)
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (1, cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替换部分
    wX, wH = np.split(weight, [wIn])
    wX     = network.add_constant([1, wIn, lenH], wX)
    wH     = network.add_constant([1, lenH,lenH], wH)
    b      = network.add_constant([1, cIn, lenH], np.tile(bias,(cIn,1)))
    h0     = network.add_constant([1, cIn, lenH], np.zeros(cIn*lenH,dtype=np.float32))              # 初始隐藏状态

    loop = network.add_loop()

    it = loop.add_iterator(inputTensor, 2, False)                                                   # 每次抛出 inputTensor 的 H 维的一层 (1,cIn,wIn)

    limit = network.add_constant((),np.array([hIn],dtype=np.int32))
    loop.add_trip_limit(limit.get_output(0),trt.TripLimit.COUNT)
    
    rLayer = loop.add_recurrence(h0.get_output(0))                                                  # 循环体
   
    temp1 = network.add_matrix_multiply(it.get_output(0), trt.MatrixOperation.NONE, wX.get_output(0), trt.MatrixOperation.NONE)

    temp2 = network.add_matrix_multiply(rLayer.get_output(0), trt.MatrixOperation.NONE, wH.get_output(0), trt.MatrixOperation.NONE)

    temp3 = network.add_elementwise(temp1.get_output(0), temp2.get_output(0), trt.ElementWiseOperation.SUM)

    temp4 = network.add_elementwise(temp3.get_output(0), b.get_output(0), trt.ElementWiseOperation.SUM)

    temp5 = network.add_activation(temp4.get_output(0),trt.ActivationType.RELU)

    rLayer.set_input(1, temp5.get_output(0))

    loopOutput1 = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)          # 形状 (1,3,5)，3 个独立输出，每个输出 1 个最终隐藏状态，每个隐藏状态 5 维坐标
    loopOutput2 = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.CONCATENATE, 1)         # 形状 (1,4,3,5)，3 个独立输出，每个输出 4 个隐藏状态，每个隐藏状态 5 维坐标
    length = network.add_constant((),np.array([hIn],dtype=np.int32))
    loopOutput2.set_input(1,length.get_output(0))
    print("loop->", loopOutput1.get_output(0).shape, loopOutput2.get_output(0).shape)

    output1 = network.add_shuffle(loopOutput1.get_output(0))                                        # 调整形状为 (3,1,5)，与 RNNv2 层的结果相同
    output1.first_transpose = (1,0,2)
    output2 = network.add_shuffle(loopOutput2.get_output(0))                                        # 调整形状为 (1,3,4,5)，与 RNNv2 层的结果相同
    output2.first_transpose = (0,2,1,3)
    output2.reshape_dims    = (cIn,hIn,lenH)
    print("outpout->", output1.get_output(0).shape, output2.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(output1.get_output(0))
    network.mark_output(output2.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
    out2_h  = np.empty(engine.get_binding_shape(2),dtype = trt.nptype(engine.get_binding_dtype(2)))
    out2_d  = cuda.mem_alloc(out2_h.nbytes)
            
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d),int(out2_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    cuda.memcpy_dtoh_async(out2_h, out2_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    print("out2_h:", out2_h.shape)
    print(out2_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (1, 3, 4, 7)，3 个独立输入，每个输入 4 个单词，每个单词 7 维坐标
$$
\left[\begin{matrix}
\left[\begin{matrix}
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
\end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (3, 1, 5)，3 个独立输出，每个输出 1 个最终隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]\\
    \left[\begin{matrix}
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]\\
    \left[\begin{matrix}
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 2 形状 (3, 4, 5)，3 个独立输出，每个输出 4 个隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \textcolor[rgb]{0,0.5,0}{7.} & \textcolor[rgb]{0,0.5,0}{7.} & \textcolor[rgb]{0,0.5,0}{7.} & \textcolor[rgb]{0,0.5,0}{7.} & \textcolor[rgb]{0,0.5,0}{7.} \\
        \textcolor[rgb]{0,0,1}{42.} & \textcolor[rgb]{0,0,1}{42.} & \textcolor[rgb]{0,0,1}{42.} & \textcolor[rgb]{0,0,1}{42.} & \textcolor[rgb]{0,0,1}{42.} \\
        217. & 217. & 217. & 217. & 217. \\
       1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
    \left[\begin{matrix}
          7. &   7. &   7. &   7. &   7. \\
         42. &  42. &  42. &  42. &  42. \\
        217. & 217. & 217. & 217. & 217. \\
       1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
    \left[\begin{matrix}
          7. &   7. &   7. &   7. &   7. \\
         42. &  42. &  42. &  42. &  42. \\
        217. & 217. & 217. & 217. & 217. \\
       1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：
$$
\begin{aligned}
h_{1}&=\textbf{ReLU}\left(W_{i,X}\cdot x_{1}+W_{i,H}\cdot h_{0}+b_{i,X}+b_{i,H}\right)\\
&=\textbf{ReLU}
\left(
  \left[\begin{matrix}
   1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1
  \end{matrix}\right]
  \left[\begin{matrix}
   1\\1\\1\\1\\1\\1\\1
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1
  \end{matrix}\right]
  \left[\begin{matrix}
   0\\0\\0\\0\\0\\0\\0
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   0\\0\\0\\0\\0\\0\\0
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   0\\0\\0\\0\\0\\0\\0
  \end{matrix}\right]
\right)\\
&=\textbf{ReLU}\left(\left(7,7,7,7,7\right)^\mathrm{T}\right)\\
&=\left(\textcolor[rgb]{0,0.5,0}{7},\textcolor[rgb]{0,0.5,0}{7},\textcolor[rgb]{0,0.5,0}{7},\textcolor[rgb]{0,0.5,0}{7},\textcolor[rgb]{0,0.5,0}{7}
  \right)^\mathrm{T}\\
\\\hfill\\
h_{2}&=\textbf{ReLU}\left(W_{i,X}\cdot x_{2}+W_{i,H}\cdot h_{1}+b_{i,X}+b_{i,H}\right)\\
&=\textbf{ReLU}
\left(
  \left[\begin{matrix}
   1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1   
  \end{matrix}\right]
  \left[\begin{matrix}
   1\\1\\1\\1\\1\\1\\1
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1
  \end{matrix}\right]
  \left[\begin{matrix}
   7\\7\\7\\7\\7\\7\\7
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   0\\0\\0\\0\\0\\0\\0
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   0\\0\\0\\0\\0\\0\\0
  \end{matrix}\right]
\right)\\
&=\textbf{ReLU}\left(\left(42,42,42,42,42\right)^\mathrm{T}\right)\\
&=\left(
    \textcolor[rgb]{0,0,1}{42},\textcolor[rgb]{0,0,1}{42},\textcolor[rgb]{0,0,1}{42},\textcolor[rgb]{0,0,1}{42},\textcolor[rgb]{0,0,1}{42}
  \right)^\mathrm{T}
\end{aligned}
$$

---
### 单层单向 LSTM
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 4                                                                                         # 输入张量 HWC
wIn     = 7
cIn     = 3
lenH    = 5                                                                                         # 隐藏层元素宽度
data    = np.ones(cIn*hIn*wIn,dtype=np.float32).reshape(cIn,hIn,wIn)                                # 输入张量
weight  = np.ones((wIn+lenH,lenH*4),dtype=np.float32)                                               # RNN 变换阵（TensorFlow 格式）
bias    = np.zeros(lenH*4, dtype=np.float32)                                                        # RNN 偏置（TensorFlow 格式）

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network(1<<0)
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (1, cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替换部分
    def gate(network, x, wx, h, wh, b, isSigmoid):
        temp1 = network.add_matrix_multiply(x, trt.MatrixOperation.NONE, wx, trt.MatrixOperation.NONE)
        temp2 = network.add_matrix_multiply(h, trt.MatrixOperation.NONE, wh, trt.MatrixOperation.NONE)
        temp3 = network.add_elementwise(temp1.get_output(0), temp2.get_output(0), trt.ElementWiseOperation.SUM)    
        temp4 = network.add_elementwise(temp3.get_output(0), b, trt.ElementWiseOperation.SUM)        
        return network.add_activation(temp4.get_output(0), [trt.ActivationType.SIGMOID if isSigmoid else trt.ActivationType.TANH][0])        
        
    wX, wH = np.split(weight, [wIn])
    wIX, wCX, wFX, wOX = [network.add_constant([1, wIn, lenH], w.transpose().reshape(-1)) for w in np.split(wX, 4, axis=1)]
    wIH, wCH, wFH, wOH = [network.add_constant([1, lenH,lenH], w.transpose().reshape(-1)) for w in np.split(wH, 4, axis=1)]
    bI,  bC,  bF,  bO  = [network.add_constant([1, cIn, lenH], np.tile(b,(cIn,1)))        for b in np.split(bias, 4)]
        
    h0 = network.add_constant([1, cIn, lenH], np.zeros(cIn*lenH,dtype=np.float32))                  # 初始隐藏状态
    c0 = network.add_constant([1, cIn, lenH], np.zeros(cIn*lenH,dtype=np.float32))                  # 初始细胞状态

    loop = network.add_loop()

    limit = network.add_constant((),np.array([hIn],dtype=np.int32))
    loop.add_trip_limit(limit.get_output(0),trt.TripLimit.COUNT)

    it = loop.add_iterator(inputTensor, 2, False)                                                   # 每次抛出 inputTensor 的 H 维的一层 (1,cIn,wIn)，反向 LSTM 要多一个反抛的迭代器
    x  = network.add_identity(it.get_output(0))                                                     # x 要被多次使用
    h  = loop.add_recurrence(h0.get_output(0))                                                      # 一个 loop 中有多个循环变量
    c  = loop.add_recurrence(h0.get_output(0))
       
    gateI = gate(network, x.get_output(0), wIX.get_output(0), h.get_output(0), wIH.get_output(0), bI.get_output(0), True)
    gateC = gate(network, x.get_output(0), wCX.get_output(0), h.get_output(0), wCH.get_output(0), bC.get_output(0), False)
    gateF = gate(network, x.get_output(0), wFX.get_output(0), h.get_output(0), wFH.get_output(0), bF.get_output(0), True)
    gateO = gate(network, x.get_output(0), wOX.get_output(0), h.get_output(0), wOH.get_output(0), bO.get_output(0), True)

    temp1 = network.add_elementwise(gateF.get_output(0), c.get_output(0), trt.ElementWiseOperation.PROD) 
    temp2 = network.add_elementwise(gateI.get_output(0), gateC.get_output(0), trt.ElementWiseOperation.PROD)
    c_    = network.add_elementwise(temp1.get_output(0), temp2.get_output(0), trt.ElementWiseOperation.SUM)
    temp3 = network.add_activation(c_.get_output(0), trt.ActivationType.TANH)
    h_    = network.add_elementwise(gateO.get_output(0), temp3.get_output(0), trt.ElementWiseOperation.PROD)

    h.set_input(1, h_.get_output(0))
    c.set_input(1, c_.get_output(0))

    loopOutput1 = loop.add_loop_output(h.get_output(0), trt.LoopOutput.LAST_VALUE, 0)               # 形状 (1,3,5)，3 个独立输出，每个输出 1 个最终隐藏状态，每个隐藏状态 5 维坐标
    loopOutput2 = loop.add_loop_output(h_.get_output(0), trt.LoopOutput.CONCATENATE, 1)              # 形状 (1,4,3,5)，3 个独立输出，每个输出 4 个隐藏状态，每个隐藏状态 5 维坐标
    length = network.add_constant((),np.array([hIn],dtype=np.int32))
    loopOutput2.set_input(1,length.get_output(0))
    print("loop->", loopOutput1.get_output(0).shape, loopOutput2.get_output(0).shape)

    output1 = network.add_shuffle(loopOutput1.get_output(0))                                        # 调整形状为 (3,1,5)，与 RNNv2 层的结果相同，可以不要
    output1.first_transpose = (1,0,2)
    output2 = network.add_shuffle(loopOutput2.get_output(0))                                        # 调整形状为 (1,3,4,5)，与 RNNv2 层的结果相同，可以不要
    output2.first_transpose = (0,2,1,3)
    output2.reshape_dims    = (cIn,hIn,lenH)
    print("outpout->", output1.get_output(0).shape, output2.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(output1.get_output(0))
    network.mark_output(output2.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
    out2_h  = np.empty(engine.get_binding_shape(2),dtype = trt.nptype(engine.get_binding_dtype(2)))
    out2_d  = cuda.mem_alloc(out2_h.nbytes)
            
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d),int(out2_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    cuda.memcpy_dtoh_async(out2_h, out2_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    print("out2_h:", out2_h.shape)
    print(out2_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (1, 3, 4, 7)，3 个独立输入，每个输入 4 个单词，每个单词 7 维坐标
$$
\left[\begin{matrix}
\left[\begin{matrix}
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
\end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (3, 1, 5)，3 个独立输出，每个输出 1 个最终隐藏状态，每个隐藏状态 5 维坐标。另外可以将 c 标记为 loop 的输出，以便获取最终或全部细胞状态。

$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0.9993216 & 0.9993216 & 0.9993216 & 0.9993216 & 0.9993216
    \end{matrix}\right]\\
    \left[\begin{matrix}
        0.9993216 & 0.9993216 & 0.9993216 & 0.9993216 & 0.9993216
    \end{matrix}\right]\\
    \left[\begin{matrix}
        0.9993216 & 0.9993216 & 0.9993216 & 0.9993216 & 0.9993216
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 2 形状 (3, 4, 5)，3 个独立输出，每个输出 4 个隐藏状态，每个隐藏状态 5 维坐标

$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \textcolor[rgb]{0,0.5,0}{0.7605171} & \textcolor[rgb]{0,0.5,0}{0.7605171} & \textcolor[rgb]{0,0.5,0}{0.7605171} & \textcolor[rgb]{0,0.5,0}{0.7605171} & \textcolor[rgb]{0,0.5,0}{0.7605171} \\
        \textcolor[rgb]{0,0,1}{0.9639405} & \textcolor[rgb]{0,0,1}{0.9639405} & \textcolor[rgb]{0,0,1}{0.9639405} & \textcolor[rgb]{0,0,1}{0.9639405} & \textcolor[rgb]{0,0,1}{0.9639405} \\
        0.99503773& 0.99503773& 0.99503773& 0.99503773& 0.99503773 \\
        0.9993216 & 0.9993216 & 0.9993216 & 0.9993216 & 0.9993216
    \end{matrix}\right]\\
    \left[\begin{matrix}
        0.7605171 & 0.7605171 & 0.7605171 & 0.7605171 & 0.7605171  \\
        0.9639405 & 0.9639405 & 0.9639405 & 0.9639405 & 0.9639405  \\
        0.99503773& 0.99503773& 0.99503773& 0.99503773& 0.99503773 \\
        0.9993216 & 0.9993216 & 0.9993216 & 0.9993216 & 0.9993216
    \end{matrix}\right]\\
    \left[\begin{matrix}
        0.7605171 & 0.7605171 & 0.7605171 & 0.7605171 & 0.7605171  \\
        0.9639405 & 0.9639405 & 0.9639405 & 0.9639405 & 0.9639405  \\
        0.99503773& 0.99503773& 0.99503773& 0.99503773& 0.99503773 \\
        0.9993216 & 0.9993216 & 0.9993216 & 0.9993216 & 0.9993216
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程，注意这里只用了一个 bias，没有区分 $b_{?,X}$ 和 $b_{?,H}$：
$$
\begin{aligned}
I_{1}=F_{1}=O_{1}=\textbf{sigmoid}\left(W_{?,X}\cdot x_{1}+W_{?,H}\cdot h_{0}+b_{?}\right)&=
  \left(0.999088,0.999088,0.999088,0.999088,0.999088\right)^\mathrm{T}\\
C_{1}=               \textbf{tanh}\left(W_{C,X}\cdot x_{1}+W_{C,H}\cdot h_{0}+b_{C}\right)&=
  \left(0.999998,0.999998,0.999998,0.999998,0.999998\right)^\mathrm{T}\\
c_{1}=F_{1}\cdot c_{0}+I_{1}\cdot C_{1}&=\left(0.999087,0.999087,0.999087,0.999087,0.999087\right)^\mathrm{T}\\
h_{1}=O_{1}\cdot \textbf{tanh}\left(c_{1}\right)&=\left(
  \textcolor[rgb]{0,0.5,0}{0.760517},\textcolor[rgb]{0,0.5,0}{0.760517},\textcolor[rgb]{0,0.5,0}{0.760517},\textcolor[rgb]{0,0.5,0}{0.760517},\textcolor[rgb]{0,0.5,0}{0.760517}
                                                  \right)^\mathrm{T}\\
\hfill\\
I_{2}=F_{2}=O_{2}=\textbf{sigmoid}\left(W_{?,X}\cdot x_{2}+W_{?,H}\cdot h_{1}+b_{?}\right)&=
  \left(0.999979,0.999979,0.999979,0.999979,0.999979\right)^\mathrm{T}\\
C_{2}=               \textbf{tanh}\left(W_{C,X}\cdot x_{2}+W_{C,H}\cdot h_{1}+b_{C}\right)&=
  \left(0.999999,0.999999,0.999999,0.999999,0.999999\right)^\mathrm{T}\\
c_{2}=F_{2}\cdot c_{1}+I_{2}\cdot C_{2}&=\left(1.999046,1.999046,1.999046,1.999046,1.999046\right)^\mathrm{T}\\
h_{2}=O_{2}\cdot \textbf{tanh}\left(c_{2}\right)&=\left(
  \textcolor[rgb]{0,0,1}{0.963940},\textcolor[rgb]{0,0,1}{0.963940},\textcolor[rgb]{0,0,1}{0.963940},\textcolor[rgb]{0,0,1}{0.963940},\textcolor[rgb]{0,0,1}{0.963940}
                                                  \right)^\mathrm{T}\\
\end{aligned}
$$

### 单层双向 LSTM TODO
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 4                                                                                         # 输入张量 HWC
wIn     = 7
cIn     = 3
lenH    = 5                                                                                         # 隐藏层元素宽度
data    = np.ones(cIn*hIn*wIn,dtype=np.float32).reshape(cIn,hIn,wIn)                                # 输入张量
weight  = np.ones((wIn+lenH,lenH*4),dtype=np.float32)                                               # RNN 变换阵（TensorFlow 格式）
bias    = np.zeros(lenH*4, dtype=np.float32)                                                        # RNN 偏置（TensorFlow 格式）

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network(1<<0)
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (1, cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替换部分
    def gate(network, x, wx, h, wh, b, isSigmoid):
        temp1 = network.add_matrix_multiply(x, trt.MatrixOperation.NONE, wx, trt.MatrixOperation.NONE)
        temp2 = network.add_matrix_multiply(h, trt.MatrixOperation.NONE, wh, trt.MatrixOperation.NONE)
        temp3 = network.add_elementwise(temp1.get_output(0), temp2.get_output(0), trt.ElementWiseOperation.SUM)    
        temp4 = network.add_elementwise(temp3.get_output(0), b, trt.ElementWiseOperation.SUM)        
        return network.add_activation(temp4.get_output(0), [trt.ActivationType.SIGMOID if isSigmoid else trt.ActivationType.TANH][0])        
        
    wX, wH = np.split(weight, [wIn])
    wIX, wCX, wFX, wOX = [network.add_constant([1, wIn, lenH], w.transpose().reshape(-1)) for w in np.split(wX, 4, axis=1)]
    wIH, wCH, wFH, wOH = [network.add_constant([1, lenH,lenH], w.transpose().reshape(-1)) for w in np.split(wH, 4, axis=1)]
    bI,  bC,  bF,  bO  = [network.add_constant([1, cIn, lenH], np.tile(b,(cIn,1)))        for b in np.split(bias, 4)]
        
    h0 = network.add_constant([1, cIn, lenH], np.zeros(cIn*lenH,dtype=np.float32))                  # 初始隐藏状态
    c0 = network.add_constant([1, cIn, lenH], np.zeros(cIn*lenH,dtype=np.float32))                  # 初始细胞状态

    loop = network.add_loop()

    limit = network.add_constant((),np.array([hIn],dtype=np.int32))
    loop.add_trip_limit(limit.get_output(0),trt.TripLimit.COUNT)

    it = loop.add_iterator(inputTensor, 2, False)                                                   # 每次抛出 inputTensor 的 H 维的一层 (1,cIn,wIn)，反向 LSTM 要多一个反抛的迭代器
    x  = network.add_identity(it.get_output(0))                                                     # x 要被多次使用
    h  = loop.add_recurrence(h0.get_output(0))                                                      # 一个 loop 中有多个循环变量
    c  = loop.add_recurrence(h0.get_output(0))
       
    gateI = gate(network, x.get_output(0), wIX.get_output(0), h.get_output(0), wIH.get_output(0), bI.get_output(0), True)
    gateC = gate(network, x.get_output(0), wCX.get_output(0), h.get_output(0), wCH.get_output(0), bC.get_output(0), False)
    gateF = gate(network, x.get_output(0), wFX.get_output(0), h.get_output(0), wFH.get_output(0), bF.get_output(0), True)
    gateO = gate(network, x.get_output(0), wOX.get_output(0), h.get_output(0), wOH.get_output(0), bO.get_output(0), True)

    temp1 = network.add_elementwise(gateF.get_output(0), c.get_output(0), trt.ElementWiseOperation.PROD) 
    temp2 = network.add_elementwise(gateI.get_output(0), gateC.get_output(0), trt.ElementWiseOperation.PROD)
    c_    = network.add_elementwise(temp1.get_output(0), temp2.get_output(0), trt.ElementWiseOperation.SUM)
    temp3 = network.add_activation(c_.get_output(0), trt.ActivationType.TANH)
    h_    = network.add_elementwise(gateO.get_output(0), temp3.get_output(0), trt.ElementWiseOperation.PROD)

    h.set_input(1, h_.get_output(0))
    c.set_input(1, c_.get_output(0))

    loopOutput1 = loop.add_loop_output(h.get_output(0), trt.LoopOutput.LAST_VALUE, 0)               # 形状 (1,3,5)，3 个独立输出，每个输出 1 个最终隐藏状态，每个隐藏状态 5 维坐标
    loopOutput2 = loop.add_loop_output(h_.get_output(0), trt.LoopOutput.CONCATENATE, 1)              # 形状 (1,4,3,5)，3 个独立输出，每个输出 4 个隐藏状态，每个隐藏状态 5 维坐标
    length = network.add_constant((),np.array([hIn],dtype=np.int32))
    loopOutput2.set_input(1,length.get_output(0))
    print("loop->", loopOutput1.get_output(0).shape, loopOutput2.get_output(0).shape)

    output1 = network.add_shuffle(loopOutput1.get_output(0))                                        # 调整形状为 (3,1,5)，与 RNNv2 层的结果相同，可以不要
    output1.first_transpose = (1,0,2)
    output2 = network.add_shuffle(loopOutput2.get_output(0))                                        # 调整形状为 (1,3,4,5)，与 RNNv2 层的结果相同，可以不要
    output2.first_transpose = (0,2,1,3)
    output2.reshape_dims    = (cIn,hIn,lenH)
    print("outpout->", output1.get_output(0).shape, output2.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(output1.get_output(0))
    network.mark_output(output2.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
    out2_h  = np.empty(engine.get_binding_shape(2),dtype = trt.nptype(engine.get_binding_dtype(2)))
    out2_d  = cuda.mem_alloc(out2_h.nbytes)
            
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d),int(out2_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    cuda.memcpy_dtoh_async(out2_h, out2_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    print("out2_h:", out2_h.shape)
    print(out2_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (1, 3, 4, 7)，3 个独立输入，每个输入 4 个单词，每个单词 7 维坐标
$$
\left[\begin{matrix}
\left[\begin{matrix}
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
\end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (3, 1, 5)，3 个独立输出，每个输出 1 个最终隐藏状态，每个隐藏状态 5 维坐标。另外可以将 c 标记为 loop 的输出，以便获取最终或全部细胞状态。

$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0.9993216 & 0.9993216 & 0.9993216 & 0.9993216 & 0.9993216
    \end{matrix}\right]\\
    \left[\begin{matrix}
        0.9993216 & 0.9993216 & 0.9993216 & 0.9993216 & 0.9993216
    \end{matrix}\right]\\
    \left[\begin{matrix}
        0.9993216 & 0.9993216 & 0.9993216 & 0.9993216 & 0.9993216
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 2 形状 (3, 4, 5)，3 个独立输出，每个输出 4 个隐藏状态，每个隐藏状态 5 维坐标

$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \textcolor[rgb]{0,0.5,0}{0.7605171} & \textcolor[rgb]{0,0.5,0}{0.7605171} & \textcolor[rgb]{0,0.5,0}{0.7605171} & \textcolor[rgb]{0,0.5,0}{0.7605171} & \textcolor[rgb]{0,0.5,0}{0.7605171} \\
        \textcolor[rgb]{0,0,1}{0.9639405} & \textcolor[rgb]{0,0,1}{0.9639405} & \textcolor[rgb]{0,0,1}{0.9639405} & \textcolor[rgb]{0,0,1}{0.9639405} & \textcolor[rgb]{0,0,1}{0.9639405} \\
        0.99503773& 0.99503773& 0.99503773& 0.99503773& 0.99503773 \\
        0.9993216 & 0.9993216 & 0.9993216 & 0.9993216 & 0.9993216
    \end{matrix}\right]\\
    \left[\begin{matrix}
        0.7605171 & 0.7605171 & 0.7605171 & 0.7605171 & 0.7605171  \\
        0.9639405 & 0.9639405 & 0.9639405 & 0.9639405 & 0.9639405  \\
        0.99503773& 0.99503773& 0.99503773& 0.99503773& 0.99503773 \\
        0.9993216 & 0.9993216 & 0.9993216 & 0.9993216 & 0.9993216
    \end{matrix}\right]\\
    \left[\begin{matrix}
        0.7605171 & 0.7605171 & 0.7605171 & 0.7605171 & 0.7605171  \\
        0.9639405 & 0.9639405 & 0.9639405 & 0.9639405 & 0.9639405  \\
        0.99503773& 0.99503773& 0.99503773& 0.99503773& 0.99503773 \\
        0.9993216 & 0.9993216 & 0.9993216 & 0.9993216 & 0.9993216
    \end{matrix}\right]
\end{matrix}\right]
$$

<div style="page-break-after:always;"></div>
## LRN 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 3                                                                                         # 输入张量 HWC
wIn     = 3
cIn     = 3
data    = np.tile(np.array([1,2,5],dtype=np.float32).reshape(cIn,1,1),(1,hIn,wIn))                  # 输入张量

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替换部分
    lrn = network.add_lrn(inputTensor, 3, 1.0, 1.0, 0.0001)                                         # LRN窗口尺寸 n，参数 alpha，beta，k
    print("lrn->", lrn.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(lrn.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (3, 3, 3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1. & 1. & 1. \\
        1. & 1. & 1. \\
        1. & 1. & 1.
    \end{matrix}\right]
    \left[\begin{matrix}
        2. & 2. & 2. \\
        2. & 2. & 2. \\
        2. & 2. & 2.
    \end{matrix}\right]
    \left[\begin{matrix}
        5. & 5. & 5. \\
        5. & 5. & 5. \\
        5. & 5. & 5.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (3, 3, 3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \textcolor[rgb]{0,0.5,0}{0.59996396} & 0.59996396 & 0.59996396 \\
        0.59996396 & 0.59996396 & 0.59996396 \\
        0.59996396 & 0.59996396 & 0.59996396
    \end{matrix}\right]
    \left[\begin{matrix}
        \textcolor[rgb]{0,0,1}{0.19999799} & 0.19999799 & 0.19999799 \\
        0.19999799 & 0.19999799 & 0.19999799 \\
        0.19999799 & 0.19999799 & 0.19999799
    \end{matrix}\right]
    \left[\begin{matrix}
        \textcolor[rgb]{1,0,0}{0.51723605} & 0.51723605 & 0.51723605 \\
        0.51723605 & 0.51723605 & 0.51723605 \\
        0.51723605 & 0.51723605 & 0.51723605
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：求和元素个数等于 $\lfloor \frac{n}{2} \rfloor$，超出输入张量边界的部分按 0 计算
$$
\frac{1^{2}}{ \left( k + \frac{\alpha}{3} \left( 0^{2} + 1^{2} + 2^{2} \right) \right)^{\beta} }
= \textcolor[rgb]{0,0.5,0}{0.59996396},
\frac{2^{2}}{ \left( k + \frac{\alpha}{3} \left( 1^{2} + 2^{2} + 5^{2} \right) \right)^{\beta} }
= \textcolor[rgb]{0,0,1}{0.19999799},
\frac{5^{2}}{ \left( k + \frac{\alpha}{3} \left( 2^{2} + 5^{2} + 0^{2}\right) \right)^{\beta} }
= \textcolor[rgb]{1,0,0}{0.51723605}
$$

---
### window_size & alpha & beta & k
```python
    lrn = network.add_lrn(inputTensor, 15, 2.0, 3.0, 4.0)                                           # 替换部分，先填入一些参数，后续再修改
    lrn.window_size = 3                                                                             # LRN 窗口尺寸，可覆盖函数 add_lrn 的参数
    lrn.alpha       = 1.0                                                                           # alpha 值，可覆盖函数 add_lrn 的参数
    lrn.beta        = 1.0                                                                           # beta 值，可覆盖函数 add_lrn 的参数
    lrn.k           = 0.0001                                                                        # k 值，可覆盖函数 add_lrn 的参数
    print("lrn->", lrn.get_output(0).shape)
```

+ 输出张量 (3, 3, 3)，与初始代码相同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0.59996396 & 0.59996396 & 0.59996396 \\
        0.59996396 & 0.59996396 & 0.59996396 \\
        0.59996396 & 0.59996396 & 0.59996396
    \end{matrix}\right]
    \left[\begin{matrix}
        0.19999799 & 0.19999799 & 0.19999799 \\
        0.19999799 & 0.19999799 & 0.19999799 \\
        0.19999799 & 0.19999799 & 0.19999799
    \end{matrix}\right]
    \left[\begin{matrix}
        0.51723605 & 0.51723605 & 0.51723605 \\
        0.51723605 & 0.51723605 & 0.51723605 \\
        0.51723605 & 0.51723605 & 0.51723605
    \end{matrix}\right]
\end{matrix}\right]
$$

<div style="page-break-after:always;"></div>
## Matrix Multiply 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 4                                                                                         # 输入张量 HWC
wIn     = 5
cIn     = 3
data    = np.arange(cIn*hIn*wIn,dtype=np.float32).reshape(cIn,hIn,wIn)                              # 输入张量 HWC

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)

    #-----------------------------------------------------------------------------------------------# 可替换部分
    one = network.add_constant((cIn,wIn,hIn),np.ones(cIn*hIn*wIn,dtype=np.float32))

    mm = network.add_matrix_multiply(inputTensor, trt.MatrixOperation.NONE, one.get_output(0), trt.MatrixOperation.NONE)
    print("mm->", mm.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(mm.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (3, 4, 5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         0. &  1. &  2. &  3. &  4. \\
         5. &  6. &  7. &  8. &  9. \\
        10. & 11. & 12. & 13. & 14. \\
        15. & 16. & 17. & 18. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 21. & 22. & 23. & 24. \\
        25. & 26. & 27. & 28. & 29. \\
        30. & 31. & 32. & 33. & 34. \\
        35. & 36. & 37. & 38. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 41. & 42. & 43. & 44. \\
        45. & 46. & 47. & 48. & 49. \\
        50. & 51. & 52. & 53. & 54. \\
        55. & 56. & 57. & 58. & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (3, 4, 4)，各通道上进行矩阵乘法
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         \textcolor[rgb]{0,0.5,0}{10.} & \textcolor[rgb]{0,0.5,0}{10.} & \textcolor[rgb]{0,0.5,0}{10.} & \textcolor[rgb]{0,0.5,0}{10.} \\
         35. & 35. & 35. & 35. \\
         60. & 60. & 60. & 60. \\
         85. & 85. & 85. & 85.
    \end{matrix}\right]
    \left[\begin{matrix}
        110. & 110. & 110. & 110. \\
        135. & 135. & 135. & 135. \\
        160. & 160. & 160. & 160. \\
        185. & 185. & 185. & 185.
    \end{matrix}\right]
    \left[\begin{matrix}
        210. & 210. & 210. & 210. \\
        235. & 235. & 235. & 235. \\
        260. & 260. & 260. & 260. \\
        285. & 285. & 285. & 285.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：
$$
\left[\begin{matrix}
  0 & 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 & 9 \\ 10 & 11 & 12 & 13 & 14 \\ 15 & 16 & 17 & 18 & 19
\end{matrix}\right]
\left[\begin{matrix}
  1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1
\end{matrix}\right]
=
\left[\begin{matrix}
  \textcolor[rgb]{0,0.5,0}{10} & \textcolor[rgb]{0,0.5,0}{10} & \textcolor[rgb]{0,0.5,0}{10} & \textcolor[rgb]{0,0.5,0}{10} \\ 35 & 35 & 35 & 35 \\ 60 & 60 & 60 & 60 \\ 85 & 85 & 85 & 85
\end{matrix}\right]
$$

---
### 乘数广播
```python
    one = network.add_constant((1,wIn,hIn),np.ones(hIn*wIn,dtype=np.float32))                     # 替换部分，one 的通道数为 1

    mm = network.add_matrix_multiply(inputTensor, trt.MatrixOperation.NONE, one.get_output(0), trt.MatrixOperation.NONE)
    print("mm->", mm.get_output(0).shape)
```

+ 输出张量 (3, 4, 4)，乘数 one 的形状由 (1,5,1) 扩充为 (3,5,4) 进行计算，结果与初始代码相同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         10. & 10. & 10. & 10. \\
         35. & 35. & 35. & 35. \\
         60. & 60. & 60. & 60. \\
         85. & 85. & 85. & 85.
    \end{matrix}\right]
    \left[\begin{matrix}
        110. & 110. & 110. & 110. \\
        135. & 135. & 135. & 135. \\
        160. & 160. & 160. & 160. \\
        185. & 185. & 185. & 185.
    \end{matrix}\right]
    \left[\begin{matrix}
        210. & 210. & 210. & 210. \\
        235. & 235. & 235. & 235. \\
        260. & 260. & 260. & 260. \\
        285. & 285. & 285. & 285.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### op0 & op1
```python
    one = network.add_constant((cIn,hIn,wIn),np.ones(cIn*hIn*wIn,dtype=np.float32))                 # 替换部分，one 形状是初始代码版本的转置

    mm = network.add_matrix_multiply(inputTensor, trt.MatrixOperation.TRANSPOSE, one.get_output(0), trt.MatrixOperation.NONE)
    mm.op0 = trt.MatrixOperation.NONE                                                               # 设置乘数特性，默认值 trt.MatrixOperation.NONE
    mm.op1 = trt.MatrixOperation.TRANSPOSE															# one 值是转置版本，这里再转置一次恢复，可覆盖 add_matrix_multiply 的参数
    print("mm->", mm.get_output(0).shape)
```

+ 输出张量 (3, 4, 4)，结果与初始代码相同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         10. & 10. & 10. & 10. \\
         35. & 35. & 35. & 35. \\
         60. & 60. & 60. & 60. \\
         85. & 85. & 85. & 85.
    \end{matrix}\right]
    \left[\begin{matrix}
        110. & 110. & 110. & 110. \\
        135. & 135. & 135. & 135. \\
        160. & 160. & 160. & 160. \\
        185. & 185. & 185. & 185.
    \end{matrix}\right]
    \left[\begin{matrix}
        210. & 210. & 210. & 210. \\
        235. & 235. & 235. & 235. \\
        260. & 260. & 260. & 260. \\
        285. & 285. & 285. & 285.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 注意，可用的选项
| trt.MatrixOperation 名 | 说明                                                       |
| :--------------------- | :--------------------------------------------------------- |
| NONE                   | 默认行为，不作限制                                         |
| VECTOR                 | 指明该参数为向量，不进行元素广播（见下方“矩阵乘向量”部分） |
| TRANSPOSE              | 计算乘法前对该矩阵进行转置                                 |

---
### 矩阵乘向量
```python
    one = network.add_constant((cIn,wIn),np.ones(cIn*wIn,dtype=np.float32))                         # 替换部分，one 比矩阵少 1 维

    mm = network.add_matrix_multiply(inputTensor, trt.MatrixOperation.NONE, one.get_output(0), trt.MatrixOperation.NONE)
    mm.op1 = trt.MatrixOperation.VECTOR																# 表明 one 是向量不是矩阵，防止被广播
    print("mm->", mm.get_output(0).shape)
```

+ 输出张量 (3, 4)，各通道上分别进行矩阵乘向量
$$
\left[\begin{matrix}
    \textcolor[rgb]{0,0.5,0}{10.} & \textcolor[rgb]{0,0.5,0}{35.} & \textcolor[rgb]{0,0.5,0}{60.} & \textcolor[rgb]{0,0.5,0}{85.} \\
    110. & 135. & 160. & 185. \\
    210. & 235. & 260. & 285.
\end{matrix}\right]
$$

+ 计算过程：
$$
\left[\begin{matrix}
  0 & 1 & 2 & 3 & 4\\5 & 6 & 7 & 8 & 9\\10 & 11 & 12 & 13 & 14\\15 & 16 & 17 & 18 & 19
\end{matrix}\right]
\left[\begin{matrix}
  1\\1\\1\\1\\1
\end{matrix}\right]
=
\left[\begin{matrix}
  \textcolor[rgb]{0,0.5,0}{10} \\ \textcolor[rgb]{0,0.5,0}{35} \\ \textcolor[rgb]{0,0.5,0}{60} \\ \textcolor[rgb]{0,0.5,0}{85}
\end{matrix}\right]
$$

---
### transpose0 & transpose1（add_matrix_multiply_deprecated）
```python
    one = network.add_constant((cIn,wIn,hIn),np.ones(cIn*hIn*wIn,dtype=np.float32))

    mm = network.add_matrix_multiply_deprecated(inputTensor, True, one.get_output(0), True)
    mm.transpose0 = False                                                                           # 指明乘数是否转置，可覆盖函数 add_matrix_multiply_deprecated 的参数
    mm.transpose1 = False
    print("mm->", mm.get_output(0).shape)
```

+ 输出张量 (3, 4, 4)，结果与初始代码相同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         10. & 10. & 10. & 10. \\
         35. & 35. & 35. & 35. \\
         60. & 60. & 60. & 60. \\
         85. & 85. & 85. & 85.
    \end{matrix}\right]
    \left[\begin{matrix}
        110. & 110. & 110. & 110. \\
        135. & 135. & 135. & 135. \\
        160. & 160. & 160. & 160. \\
        185. & 185. & 185. & 185.
    \end{matrix}\right]
    \left[\begin{matrix}
        210. & 210. & 210. & 210. \\
        235. & 235. & 235. & 235. \\
        260. & 260. & 260. & 260. \\
        285. & 285. & 285. & 285.
    \end{matrix}\right]
\end{matrix}\right]
$$

<div style="page-break-after:always;"></div>
## padding 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 4                                                                                         # 输入张量 HWC
wIn     = 5                                                                                         # width of input tensor
cIn     = 3                                                                                         # channel of input tensor
data    = np.ones((cIn, hIn, wIn),dtype=np.float32)                                                 # 输入张量

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替换部分
    pad = network.add_padding(inputTensor, (1,2), (3,4))
    print("pad->", pad.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(pad.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (3, 4, 5)
$$
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
        1. & 1. & 1. & 1. & 1.]]]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (3, 8, 11)，在输入张量的上、左、下、右分别垫起了 1、2、3、4 层元素 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.
    \end{matrix}\right]\\
    \left[\begin{matrix}
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.
    \end{matrix}\right]\\
    \left[\begin{matrix}
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### pre_padding & post_padding
```python
    pad = network.add_padding(inputTensor, (0,0), (0,0))                                            # 替换部分
    pad.pre_padding = (1,2)                                                                         # 上侧和左侧垫起元素 0 层数，可覆盖函数 add_padding 的参数
    pad.post_padding = (3,4)                                                                        # 下侧和右侧垫起元素 0 层数，可覆盖函数 add_padding 的参数
    print("pad->", pad.get_output(0).shape)
```

$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.
    \end{matrix}\right]\\
    \left[\begin{matrix}
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.
    \end{matrix}\right]\\
    \left[\begin{matrix}
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 1. & 1. & 1. & 1. & 1. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. \\
        0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0. & 0.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### pre_padding_nd & post_padding_nd
+ 目前只支持 2 维，效果与 pre_padding_nd 和post_padding_nd 相同，将来可能支持更高维度

<div style="page-break-after:always;"></div>
## Parametric ReLU 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 3                                                                                         # 输入张量 HWC
wIn     = 3
cIn     = 3
data    = np.tile(np.arange(-4,5,dtype=np.float32).reshape(1,hIn,wIn),(cIn,1,1))                    # 输入张量

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替换部分
    slope = network.add_constant((cIn,1,1),np.array([0.5,1,2],dtype=np.float32))                    # 斜率张量，可广播到与 inputTensor 相同大小即可，可以控制在全局、单维度、元素的水平上设置斜率
    
    pReLU = network.add_parametric_relu(inputTensor,slope.get_output(0))
    print("pReLU->", pReLU.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(pReLU.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (3, 3, 3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        -4. & -3. & -2. \\
        -1. &  0. &  1. \\
         2. &  3. &  4.
    \end{matrix}\right]
    \left[\begin{matrix}
        -4. & -3. & -2. \\
        -1. &  0. &  1. \\
         2. &  3. &  4.
    \end{matrix}\right]
    \left[\begin{matrix}
        -4. & -3. & -2. \\
        -1. &  0. &  1. \\
         2. &  3. &  4.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (3, 3, 3)，等价于 leaky ReLU，负数部分的斜率被改变
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        -2.  & -1.5. & -1. \\
        -0.5 &  0.   &  1. \\
         2.  &  3.   &  4.
    \end{matrix}\right]
    \left[\begin{matrix}
        -4. & -3. & -2. \\
        -1. &  0. &  1. \\
         2. &  3. &  4.
    \end{matrix}\right]
    \left[\begin{matrix}
        -8. & -6. & -4. \\
        -2. &  0. &  1. \\
         2. &  3. &  4.
    \end{matrix}\right]
\end{matrix}\right]
$$

<div style="page-break-after:always;"></div>
## Pooling 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 6                                                                                         # 输入张量 HWC
wIn     = 9
cIn     = 1
hW      = 2                                                                                         # 池化窗口 HW
wW      = 2
data    = np.tile(np.arange(9,dtype=np.float32).reshape(1,3,3),(cIn,hIn//3,wIn//3))                 # 输入张量

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
        
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替代部分
    pool = network.add_pooling(inputTensor, trt.PoolingType.MAX,(hW, wW))
    print("pool(default)->", pool.get_output(0).shape)                                              # (cIn,hIn//wW,wIn//hW)
    #-----------------------------------------------------------------------------------------------# 可替代部分

    network.mark_output(pool.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (1, 6, 9)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1. & 2. & 3. & 1. & 2. & 3. & 1. & 2. & 3. \\
        4. & 5. & 6. & 4. & 5. & 6. & 4. & 5. & 6. \\
        7. & 8. & 9. & 7. & 8. & 9. & 7. & 8. & 9. \\
        1. & 2. & 3. & 1. & 2. & 3. & 1. & 2. & 3. \\
        4. & 5. & 6. & 4. & 5. & 6. & 4. & 5. & 6. \\
        7. & 8. & 9. & 7. & 8. & 9. & 7. & 8. & 9.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 3, 4)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        5. & 6. & 6. & 5. \\
        8. & 9. & 9. & 8. \\
        8. & 9. & 9. & 8.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### type
```python
    pool = network.add_pooling(inputTensor, trt.PoolingType.MAX,(hW, wW))                           # 替换部分，有 bug
    pool.type = trt.PoolingType.AVERAGE														        # 池化方式，可覆盖函数 add_pooling 的参数
    print("pool(kernel_size & num_output_maps & kernel & bias)->", pool.get_output(0).shape)        # (cIn,hIn//hW,wIn//wW)
```

+ 输出张量 (1, 3, 4)，其中指定平均值池化
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        2.  & 2.5 & 3.  & 2.  \\
        3.5 & 4.  & 4.5 & 3.5 \\
        5.  & 5.5 & 6.  & 5.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 注意：可用的池化方式
| trt.PoolingType 名 | 说明                                                         |
| :----------------- | :----------------------------------------------------------- |
| AVERAGE            | 均值池化                                                     |
| MAX                | 最大值池化                                                   |
| MAX_AVERAGE_BLEND  | 混合池化，(1 - blendFactor) * maxPool + blendFactor * avgPool |

---
### window_size
```python
    pool = network.add_pooling(inputTensor, trt.PoolingType.MAX,(hW, wW))                           # 替换部分，有 bug
    pool.window_size  = (hW, wW)                                                                    # 池化窗口大小，可覆盖函数 add_pooling 的参数
    print("pool(kernel_size & num_output_maps & kernel & bias)->", pool.get_output(0).shape)        # (cIn,hIn//hW,wIn//wW)
```

+ 输出张量 (1, 3, 4)，与初始代码相同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        5. & 6. & 6. & 5. \\
        8. & 9. & 9. & 8. \\
        8. & 9. & 9. & 8.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### stride
```python
    hS = wS = 2                                                                                     # 替代部分
    pool = network.add_pooling(inputTensor, trt.PoolingType.MAX,(hW, wW))
    pool.stride = (1,1)                                                                             # 池化窗口移动步长，默认值等于窗口大小（TRT API 有误）
    print("pool->", pool.get_output(0).shape)                                                       # (cIn,(hIn-hW+1)//hS,(wIn-wW+1)//wS)
```

+ 输出张量 (1, 5, 8)，其中指定 stride=(1,1)，步长可以比 hW 或 wW 更小
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        5. & 6. & 6. & 5. & 6. & 6. & 5. & 6. \\
        8. & 9. & 9. & 8. & 9. & 9. & 8. & 9. \\
        8. & 9. & 9. & 8. & 9. & 9. & 8. & 9. \\
        5. & 6. & 6. & 5. & 6. & 6. & 5. & 6. \\
        8. & 9. & 9. & 8. & 9. & 9. & 8. & 9.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### padding
```python
    hP = wP = 1                                                                                     # 替代部分
    pool = network.add_pooling(inputTensor, trt.PoolingType.MAX,(hW, wW))
    pool.padding = (hP, wP)                                                                         # 四周垫 0 层数，默认值为 (0,0)
    print("pool->", pool.get_output(0).shape)                                                       # (cIn,hIn-hW+hP*2+1,wIn-wW+wP*2+1)
```

+ 输出张量 (1, 4, 4)，其中指定 padding=(1,0)，H 维垫 1 层元素 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        2. & 3. & 3. & 2. \\
        8. & 9. & 9. & 8. \\
        5. & 6. & 6. & 5. \\
        8. & 9. & 9. & 8.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 3, 5)，其中指定 padding=(0,1)，W 维垫 1 层元素 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        4. & 6. & 5. & 6. & 6. \\
        7. & 9. & 8. & 9. & 9. \\
        7. & 9. & 8. & 9. & 9.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 4, 5)，其中指定 padding=(1,1)，HW 维均垫 1 层元素 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1. & 3. & 2. & 3. & 3. \\
        7. & 9. & 8. & 9. & 9. \\
        4. & 6. & 5. & 6. & 6. \\
        7. & 9. & 8. & 9. & 9.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### pre_padding
```python
    hPre = wPre = 1                                                                                 # 替换部分
    pool = network.add_pooling(inputTensor, trt.PoolingType.MAX,(hW, wW))
    pool.pre_padding = (hPre, wPre)                                                                 # 顶部和左侧垫 0 层数，默认值为 (0,0)
    print("pool->", pool.get_output(0).shape)                                                       # (cIn,hIn-hW+hPre+1,wIn-wW+wPre+1)
```

+ 输出张量 (1, 3, 4)，其中指定 pre_padding=(1,0)，H 维头部垫 1 层元素 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        2. & 3. & 3. & 2. \\
        8. & 9. & 9. & 8. \\
        5. & 6. & 6. & 5.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 3, 5)，其中指定 pre_padding=(0,1)，W 维头部垫 1 层元素 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        4. & 6. & 5. & 6. & 6. \\
        7. & 9. & 8. & 9. & 9. \\
        7. & 9. & 8. & 9. & 9.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 3, 5)，其中指定 pre_padding=(1,1)，HW 维头部均垫 1 层元素 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1. & 3. & 2. & 3. & 3. \\
        7. & 9. & 8. & 9. & 9. \\
        4. & 6. & 5. & 6. & 6.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### post_padding
```python
    hPost = wPost = 1                                                                               # 替换部分
    pool = network.add_pooling(inputTensor, trt.PoolingType.MAX,(hW, wW))
    pool.post_padding = (hPost, wPost)                                                              # 底部和右侧垫 0 层数，默认值为 (0,0)
    print("pool->", pool.get_output(0).shape)                                                       # (cIn,hIn-hW+hPost+1,wIn-wW+wPost+1)
```

+ 输出张量 (1, 3, 4)，其中指定 post_padding=(1,0)，H 维尾部垫 1 层元素 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        5. & 6. & 6. & 5. \\
        8. & 9. & 9. & 8. \\
        8. & 9. & 9. & 8.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 3, 5)，其中指定 post_padding=(0,1)，W 维尾部垫 1 层元素 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        5. & 6. & 6. & 5. & 6. \\
        8. & 9. & 9. & 8. & 9. \\
        8. & 9. & 9. & 8. & 9.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 3, 5)，其中指定 post_padding=(1,1)，HW 维尾部均垫 1 层元素 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        5. & 6. & 6. & 5. & 6. \\
        8. & 9. & 9. & 8. & 9. \\
        8. & 9. & 9. & 8. & 9.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### padding_mode
```python
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn-1, wIn))           # 手动去除输入张量的最后一行，以便观察 pad 模式的影响
    #...
    pool = network.add_pooling(inputTensor, trt.PoolingType.MAX,(hW, wW))
    pool.padding_mode = trt.PaddingMode.SAME_UPPER
    print("pool->", pool.get_output(0).shape)                                                       # (cIn,(hIn-hW+?+1)//hS,(wIn-wW+?+1)//wS)
```

+ 输出张量 (1, 3, 5)，其中指定 padding_mode = **trt.PaddingMode.SAME_UPPER**
  目标尺寸 $ \left( h',w' \right) = \left( \lceil \frac{hIn}{hW} \rceil, \lceil \frac{wIn}{wW} \rceil \right) $，为此 H 维需要总垫 0 层数 $ hP = \left( h' \cdot hW - hIn \right) $，然后取 $ hPre = \textcolor[rgb]{1,0,0}{\lfloor \frac{hP}{2} \rfloor} $，$ hPost = \textcolor[rgb]{1,0,0}{\lceil \frac{hP}{2} \rceil} $，W 维类似
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        5. & 6. & 6. & 5. & 6. \\
        8. & 9. & 9. & 8. & 9. \\
        5. & 6. & 6. & 5. & 6.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 3, 5)，其中指定 padding_mode = *trt.PaddingMode.SAME_LOWER**
  目标尺寸同上，然后 $ hPre = \textcolor[rgb]{1,0,0}{\lceil \frac{hP}{2} \rceil} $，$ hPost = \textcolor[rgb]{1,0,0}{\lfloor \frac{hP}{2} \rfloor} $，W 维类似
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1. & 3. & 2. & 3. & 3. \\
        7. & 9. & 8. & 9. & 9. \\
        4. & 6. & 5. & 6. & 6.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 3, 5)，其中指定 padding_mode = **trt.PaddingMode.EXPLICIT_ROUND_UP**
  目标尺寸同上，然后全部垫在底部 $ hPost = hP $，W 维类似
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        5. & 6. & 6. & 5. & 6. \\
        8. & 9. & 9. & 8. & 9. \\
        5. & 6. & 6. & 5. & 6.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 2, 4)，其中指定 padding_mode = **trt.PaddingMode.EXPLICIT_ROUND_DOWN**
  目标尺寸 $ \left( h',w' \right) = \left( \lfloor \frac{hIn}{hW} \rfloor, \lfloor \frac{wIn}{wW} \rfloor \right) $，不做任何垫 0 处理
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        5. & 6. & 6. & 5. \\
        8. & 9. & 9. & 8.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 3, 5)，其中指定 padding_mode = **trt.PaddingMode.CAFFE_ROUND_UP**
  目标尺寸同上，然后全部垫在底部 $ hPost = hP $，W 维类似
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        5. & 6. & 6. & 5. & 6. \\
        8. & 9. & 9. & 8. & 9. \\
        5. & 6. & 6. & 5. & 6.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 2, 4)，其中指定 padding_mode = **trt.PaddingMode.EXPLICIT_ROUND_DOWN**
  目标尺寸 $ \left( h',w' \right) = \left( \lfloor \frac{hIn}{hW} \rfloor, \lfloor \frac{wIn}{wW} \rfloor \right) $，不做任何垫 0 处理
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        5. & 6. & 6. & 5. \\
        8. & 9. & 9. & 8.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### blend_factor
```python
    bF = 0.8                                                                                        # 替换部分
    pool = network.add_pooling(inputTensor, trt.PoolingType.MAX_AVERAGE_BLEND,(hW, wW))
    pool.blend_factor   = bF                                                                        # 均值池化的比例，默认值为 1.0
    print("pool->", pool.get_output(0).shape)
```

+ 输出张量 (1, 3, 4)，其中指定 blend_factor=0.5
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        4.   & 4.75 & 5.   &   4. \\
        6.25 & 7.   & 7.25 & 6.25 \\
        7.   & 7.75 & 8.   &   7. \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        4.   & 4.75 & 5.   &   4. \\
        6.25 & 7.   & 7.25 & 6.25 \\
        7.   & 7.75 & 8.   &   7. \\
    \end{matrix}\right]
\end{matrix}\right]
= bF \cdot
\left[\begin{matrix}
    \left[\begin{matrix}
        3.  & 3.5 & 4.  & 3.  \\
        4.5 & 5.  & 5.5 & 4.5 \\
        6.  & 6.5 & 7.  & 6.
    \end{matrix}\right]
\end{matrix}\right]
+ \left( 1 - bF \right) \cdot
\left[\begin{matrix}
    \left[\begin{matrix}
        5. & 6. & 6. & 5. \\
        8. & 9. & 9. & 8. \\
        8. & 9. & 9. & 8.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### average_count_excludes_padding
```python
    pool = network.add_pooling(inputTensor, trt.PoolingType.AVERAGE,(hW, wW))                       # 替换部分
    pool.padding                        = (1,1)                                                     # 不支持非对称垫 0，如 pre_padding 或 post_padding
    pool.average_count_excludes_padding = False                                                     # 是否排除分母中的垫 0 元素计数，默认值为 True
    print("pool->", pool.get_output(0).shape)
```

+ 输出张量 (1, 4, 5)，其中指定 average_count_excludes_padding=False，均值池化时连着垫 0 一起计算（默认状态池化窗口中若有垫 0 元素，则不计入分母）
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0.25 & 1.25 & 0.75 & 1.  & 1.25 \\
        2.75 & 7.   & 6.   & 6.5 & 7.   \\
        1.25 & 4.   & 3.   & 3.5 & 4.   \\
        1.75 & 4.25 & 3.75 & 4.  & 4.25
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### window_size_nd & stride_nd & padding_nd（add_pooling_nd ）
```python
# 三维池化的例子，调整部分参数
cIn     = 2
cW      = 2
data    = np.tile(np.arange(1,10,dtype=np.float32).reshape(3,3),(2,2,3)).reshape(cIn,hIn,wIn)
data[1]*= 10
```
```python
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (1,cIn, hIn, wIn))           # 替换部分，要求输入张量维度是 4
    #...
    pool = network.add_pooling_nd(inputTensor, trt.PoolingType.MAX,(cW,hW, wW))                     # 注意池化窗口是 3 维的
    pool.window_size_nd = (cW,hW,wW)                                                                # 池化窗口尺寸
    pool.stride_nd      = (1,1,1)                                                                   # 赤化窗口移动步长，默认值为 (1,1,1)
    pool.padding_nd     = (0,0,0)                                                                   # 四周垫 0 层数，默认值为 (0,0,0)
    print("pool->", pool.get_output(0).shape)
```

+ 输入张量 (2, 6, 9)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         1. &  2. &  3. &  1. &  2. &  3. &  1. &  2. &  3. \\
         4. &  5. &  6. &  4. &  5. &  6. &  4. &  5. &  6. \\
         7. &  8. &  9. &  7. &  8. &  9. &  7. &  8. &  9. \\
         1. &  2. &  3. &  1. &  2. &  3. &  1. &  2. &  3. \\
         4. &  5. &  6. &  4. &  5. &  6. &  4. &  5. &  6. \\
         7. &  8. &  9. &  7. &  8. &  9. &  7. &  8. &  9.
    \end{matrix}\right]
    \left[\begin{matrix}
        10. & 20. & 30. & 10. & 20. & 30. & 10. & 20. & 30. \\
        40. & 50. & 60. & 40. & 50. & 60. & 40. & 50. & 60. \\
        70. & 80. & 90. & 70. & 80. & 90. & 70. & 80. & 90. \\
        10. & 20. & 30. & 10. & 20. & 30. & 10. & 20. & 30. \\
        40. & 50. & 60. & 40. & 50. & 60. & 40. & 50. & 60. \\
        70. & 80. & 90. & 70. & 80. & 90. & 70. & 80. & 90.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 1, 3, 4)，最大元素全都来自靠后的通道
$$
\left[\begin{matrix}
\left[\begin{matrix}
    \left[\begin{matrix}
        50. & 60. & 60. & 50. \\
        80. & 90. & 90. & 80. \\
        80. & 90. & 90. & 80.
    \end{matrix}\right]
\end{matrix}\right]
\end{matrix}\right]
$$

<div style="page-break-after:always;"></div>
## Ragged Soft Max 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 4                                                                                         # 输入张量 HWC
wIn     = 5
cIn     = 1
data    = np.ones(cIn*hIn*wIn,dtype=np.float32).reshape(cIn,hIn,wIn)                                #输入张量

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (hIn, wIn))                  # 严格要求输入张量的 non-batch 是 2 维
    print("inputTensor->", inputTensor.shape)

    #-----------------------------------------------------------------------------------------------# 可替换部分
    bound = network.add_constant((hIn,1),np.array([5,4,3,2],dtype=np.int32))

    rsm = network.add_ragged_softmax(inputTensor, bound.get_output(0))
    print("rsm->", rsm.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(rsm.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 3, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (4, 5)
$$
\left[\begin{matrix}
    1. & 1. & 1. & 1. & 1. \\
    1. & 1. & 1. & 1. & 1. \\
    1. & 1. & 1. & 1. & 1. \\
    1. & 1. & 1. & 1. & 1.
\end{matrix}\right]
$$

+ 输出张量 (4, 5)，各行在指定长度（5,4,3,2）上计算了 Soft Max，其余元素变成 0
$$
\left[\begin{matrix}
    0.2   & 0.2   & 0.2   & 0.2  & 0.2 \\
    0.25  & 0.25  & 0.25  & 0.25 & 0.  \\
    0.334 & 0.334 & 0.334 & 0.   & 0.  \\
    0.5   & 0.5   & 0.    & 0.   & 0.  \\
\end{matrix}\right]
$$

<div style="page-break-after:always;"></div>
## Reduce 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 4                                                                                         # 输入张量 HWC
wIn     = 5
cIn     = 3
data    = np.ones((cIn, hIn, wIn),dtype=np.float32)                                                 # 输入张量

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替换部分
    reduce = network.add_reduce(inputTensor, trt.ReduceOperation.SUM, 1<<0, False)
    print("reduce->", reduce.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(reduce.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (3, 4, 5)
$$
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
$$

+ 输出张量 (4, 5)，在最高“非batch”维（C 维）上进行了求和
$$
\left[\begin{matrix}
    3. & 3. & 3. & 3. & 3. \\
    3. & 3. & 3. & 3. & 3. \\
    3. & 3. & 3. & 3. & 3. \\
    3. & 3. & 3. & 3. & 3.
\end{matrix}\right]
$$

---
### op
```python
    reduce = network.add_reduce(inputTensor, trt.ReduceOperation.SUM, 1<<0, False)
    reduce.op = trt.ReduceOperation.PROD                                                            # 规约运算种类，可覆盖函数 add_reduce 的参数
    print("reduce->", reduce.get_output(0).shape)
```

+ 输出张量 (4, 5)，其中指定 op = trt.ReduceOperation.PROD，在最高“非batch”维（C 维）上进行了求积
$$
\left[\begin{matrix}
    1. & 1. & 1. & 1. & 1. \\
    1. & 1. & 1. & 1. & 1. \\
    1. & 1. & 1. & 1. & 1. \\
    1. & 1. & 1. & 1. & 1.
\end{matrix}\right]
$$

+ 注意，可用的计算方法
| trt.ReduceOperation | 函数     |
| :------------------ | :------- |
| PROD                | 求积     |
| AVG                 | 求平均值 |
| MAX                 | 取最大值 |
| MIN                 | 取最小值 |
| SUM                 | 求和     |

---
### axes
```python
    axesIndex = 0                                                                                   # 替换部分
    sm = network.add_softmax(inputTensor)
    sm.axes = 1 << axesIndex                                                                        # 规约的轴号，默认值为 1<<0
    print("sm->", sm.get_output(0).shape)
```

+ 输出张量 (4, 5)，其中指定 axes=1<<0，在最高“非batch”维（C 维）上进行了求和
$$
\left[\begin{matrix}
    3. & 3. & 3. & 3. & 3. \\
    3. & 3. & 3. & 3. & 3. \\
    3. & 3. & 3. & 3. & 3. \\
    3. & 3. & 3. & 3. & 3.
\end{matrix}\right]
$$

+ 输出张量 (3, 5)，其中指定 axes=1<<1，在次高“非batch”维（H 维）上进行了求和
$$
\left[\begin{matrix}
    4. & 4. & 4. & 4. & 4. \\
    4. & 4. & 4. & 4. & 4. \\
    4. & 4. & 4. & 4. & 4.
\end{matrix}\right]
$$

+ 输出张量 (3, 4)，其中指定 axes=1<<2，在季高“非batch”维（W 维）上进行了求和
$$
\left[\begin{matrix}
    5. & 5. & 5. & 5. \\
    5. & 5. & 5. & 5. \\
    5. & 5. & 5. & 5.
\end{matrix}\right]
$$

+ 输出张量 (5,)，其中指定 axes = 1<<0 + 1<<2，即同时在两个维度上进行 reduce
$$
\left[\begin{matrix}
    12. & 12. & 12. & 12. & 12. \\
\end{matrix}\right]
$$

---
### keep_dims
```python
    reduce = network.add_reduce(inputTensor, trt.ReduceOperation.SUM, 1<<0, False)                  # 替换部分
    reduce.keep_dims = True                                                                         # 是否保留规约轴的维度，默认值为 False
    print("reduce->", reduce.get_output(0).shape)
```

+ 输出张量 (1, 4, 5)，其中指定 keep_dims=True，保留了发生规约的通道维
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3. \\
        3. & 3. & 3. & 3. & 3.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (4, 5)，其中指定 keep_dims=False，删除了发生规约的通道维，结果与初始代码相同
$$
\left[\begin{matrix}
    3. & 3. & 3. & 3. & 3. \\
    3. & 3. & 3. & 3. & 3. \\
    3. & 3. & 3. & 3. & 3. \\
    3. & 3. & 3. & 3. & 3.
\end{matrix}\right]
$$

<div style="page-break-after:always;"></div>
## Resize 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 4                                                                                         # 输入张量 HWC
wIn     = 5
cIn     = 3
hOut    = 6                                                                                         # 输出张量 HWC
wOut    = 10
cOut    = 3
data    = np.arange(cIn*hIn*wIn,dtype=np.float32).reshape(cIn,hIn,wIn)                              # 输入张量

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替代部分
    rsz = network.add_resize(inputTensor)
    rsz.shape = (cOut, hOut, wOut)
    print("rsz->", rsz.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替代部分

    network.mark_output(rsz.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 3, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (3, 4, 5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         0. &  1. &  2. &  3. &  4. \\
         5. &  6. &  7. &  8. &  9. \\
        10. & 11. & 12. & 13. & 14. \\
        15. & 16. & 17. & 18. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 21. & 22. & 23. & 24. \\
        25. & 26. & 27. & 28. & 29. \\
        30. & 31. & 32. & 33. & 34. \\
        35. & 36. & 37. & 38. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 41. & 42. & 43. & 44. \\
        45. & 46. & 47. & 48. & 49. \\
        50. & 51. & 52. & 53. & 54. \\
        55. & 56. & 57. & 58. & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (3, 6, 10)，默认最邻近插值
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         0. &  0. &  1. &  1. &  2. &  2. &  3. &  3. &  4. &  4. \\
         0. &  0. &  1. &  1. &  2. &  2. &  3. &  3. &  4. &  4. \\
         5. &  5. &  6. &  6. &  7. &  7. &  8. &  8. &  9. &  9. \\
        10. & 10. & 11. & 11. & 12. & 12. & 13. & 13. & 14. & 14. \\
        10. & 10. & 11. & 11. & 12. & 12. & 13. & 13. & 14. & 14. \\
        15. & 15. & 16. & 16. & 17. & 17. & 18. & 18. & 19. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 20. & 21. & 21. & 22. & 22. & 23. & 23. & 24. & 24. \\
        20. & 20. & 21. & 21. & 22. & 22. & 23. & 23. & 24. & 24. \\
        25. & 25. & 26. & 26. & 27. & 27. & 28. & 28. & 29. & 29. \\
        30. & 30. & 31. & 31. & 32. & 32. & 33. & 33. & 34. & 34. \\
        30. & 30. & 31. & 31. & 32. & 32. & 33. & 33. & 34. & 34. \\
        35. & 35. & 36. & 36. & 37. & 37. & 38. & 38. & 39. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 40. & 41. & 41. & 42. & 42. & 43. & 43. & 44. & 44. \\
        40. & 40. & 41. & 41. & 42. & 42. & 43. & 43. & 44. & 44. \\
        45. & 45. & 46. & 46. & 47. & 47. & 48. & 48. & 49. & 49. \\
        50. & 50. & 51. & 51. & 52. & 52. & 53. & 53. & 54. & 54. \\
        50. & 50. & 51. & 51. & 52. & 52. & 53. & 53. & 54. & 54. \\
        55. & 55. & 56. & 56. & 57. & 57. & 58. & 58. & 59. & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### shape
```python
    rsz = network.add_resize(inputTensor)                                                           # 替换部分
    rsz.shape = (cOut, hOut, wOut)                                                                  # 输出张量形状
    print("rsz->", rsz.get_output(0).shape)                                                         # (cOut, hOut, wOut)
```

+ 输出张量 (3, 6, 10)，与初始代码相同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         0. &  0. &  1. &  1. &  2. &  2. &  3. &  3. &  4. &  4. \\
         0. &  0. &  1. &  1. &  2. &  2. &  3. &  3. &  4. &  4. \\
         5. &  5. &  6. &  6. &  7. &  7. &  8. &  8. &  9. &  9. \\
        10. & 10. & 11. & 11. & 12. & 12. & 13. & 13. & 14. & 14. \\
        10. & 10. & 11. & 11. & 12. & 12. & 13. & 13. & 14. & 14. \\
        15. & 15. & 16. & 16. & 17. & 17. & 18. & 18. & 19. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 20. & 21. & 21. & 22. & 22. & 23. & 23. & 24. & 24. \\
        20. & 20. & 21. & 21. & 22. & 22. & 23. & 23. & 24. & 24. \\
        25. & 25. & 26. & 26. & 27. & 27. & 28. & 28. & 29. & 29. \\
        30. & 30. & 31. & 31. & 32. & 32. & 33. & 33. & 34. & 34. \\
        30. & 30. & 31. & 31. & 32. & 32. & 33. & 33. & 34. & 34. \\
        35. & 35. & 36. & 36. & 37. & 37. & 38. & 38. & 39. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 40. & 41. & 41. & 42. & 42. & 43. & 43. & 44. & 44. \\
        40. & 40. & 41. & 41. & 42. & 42. & 43. & 43. & 44. & 44. \\
        45. & 45. & 46. & 46. & 47. & 47. & 48. & 48. & 49. & 49. \\
        50. & 50. & 51. & 51. & 52. & 52. & 53. & 53. & 54. & 54. \\
        50. & 50. & 51. & 51. & 52. & 52. & 53. & 53. & 54. & 54. \\
        55. & 55. & 56. & 56. & 57. & 57. & 58. & 58. & 59. & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### scales
```python
    rsz = network.add_resize(inputTensor)                                                           # 替换部分
    rsz.scales = (cOut/cIn, hOut/hIn, wOut/wIn)                                                     # 各维扩张比率，sizeNew = np.floor(sizeOld * factor) 
    print("rsz->", rsz.get_output(0).shape)                                                         # (cOut, hOut, wOut)
```

+ 输出张量 (3, 6, 10)，与初始代码相同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         0. &  0. &  1. &  1. &  2. &  2. &  3. &  3. &  4. &  4. \\
         0. &  0. &  1. &  1. &  2. &  2. &  3. &  3. &  4. &  4. \\
         5. &  5. &  6. &  6. &  7. &  7. &  8. &  8. &  9. &  9. \\
        10. & 10. & 11. & 11. & 12. & 12. & 13. & 13. & 14. & 14. \\
        10. & 10. & 11. & 11. & 12. & 12. & 13. & 13. & 14. & 14. \\
        15. & 15. & 16. & 16. & 17. & 17. & 18. & 18. & 19. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 20. & 21. & 21. & 22. & 22. & 23. & 23. & 24. & 24. \\
        20. & 20. & 21. & 21. & 22. & 22. & 23. & 23. & 24. & 24. \\
        25. & 25. & 26. & 26. & 27. & 27. & 28. & 28. & 29. & 29. \\
        30. & 30. & 31. & 31. & 32. & 32. & 33. & 33. & 34. & 34. \\
        30. & 30. & 31. & 31. & 32. & 32. & 33. & 33. & 34. & 34. \\
        35. & 35. & 36. & 36. & 37. & 37. & 38. & 38. & 39. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 40. & 41. & 41. & 42. & 42. & 43. & 43. & 44. & 44. \\
        40. & 40. & 41. & 41. & 42. & 42. & 43. & 43. & 44. & 44. \\
        45. & 45. & 46. & 46. & 47. & 47. & 48. & 48. & 49. & 49. \\
        50. & 50. & 51. & 51. & 52. & 52. & 53. & 53. & 54. & 54. \\
        50. & 50. & 51. & 51. & 52. & 52. & 53. & 53. & 54. & 54. \\
        55. & 55. & 56. & 56. & 57. & 57. & 58. & 58. & 59. & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### resize_mode
```python
    rsz = network.add_resize(inputTensor)                                                           # 替换部分
    rsz.shape = (cOut, hOut, wOut)
    rsz.resize_mode = trt.ResizeMode.LINEAR                                                         # 插值方法，默认值为 trt.ResizeMode.NEAREST
    print("rsz->", rsz.get_output(0).shape)
```

+ 输出张量 (3, 6, 10)，其中指定 resize_mode=LINEAR
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         0.    &  0.5   &  1.    &  1.5   &  2.    &  2.5   &  3.    &  3.5   &  4.    &  4.    \\
         3.333 &  3.833 &  4.333 &  4.833 &  5.333 &  5.833 &  6.333 &  6.833 &  7.333 &  7.333 \\
         6.667 &  7.167 &  7.667 &  8.167 &  8.667 &  9.167 &  9.667 & 10.167 & 10.667 & 10.667 \\
        10.    & 10.5   & 11.    & 11.5   & 12.    & 12.5   & 13.    & 13.5   & 14.    & 14.    \\
        13.333 & 13.833 & 14.333 & 14.833 & 15.333 & 15.833 & 16.333 & 16.833 & 17.333 & 17.333 \\
        15.    & 15.5   & 16.    & 16.5   & 17.    & 17.5   & 18.    & 18.5   & 19.    & 19.   
    \end{matrix}\right]\\
    \left[\begin{matrix}
        20.    & 20.5   & 21.    & 21.5   & 22.    & 22.5   & 23.    & 23.5   & 24.    & 24.    \\
        23.333 & 23.833 & 24.333 & 24.833 & 25.333 & 25.833 & 26.333 & 26.833 & 27.333 & 27.333 \\
        26.667 & 27.167 & 27.667 & 28.167 & 28.667 & 29.167 & 29.667 & 30.167 & 30.667 & 30.667 \\
        30.    & 30.5   & 31.    & 31.5   & 32.    & 32.5   & 33.    & 33.5   & 34.    & 34.    \\
        33.333 & 33.833 & 34.333 & 34.833 & 35.333 & 35.833 & 36.333 & 36.833 & 37.333 & 37.333 \\
        35.    & 35.5   & 36.    & 36.5   & 37.    & 37.5   & 38.    & 38.5   & 39.    & 39.   
    \end{matrix}\right]\\
    \left[\begin{matrix}
        40.    & 40.5   & 41.    & 41.5   & 42.    & 42.5   & 43.    & 43.5   & 44.    & 44.    \\
        43.333 & 43.833 & 44.333 & 44.833 & 45.333 & 45.833 & 46.333 & 46.833 & 47.333 & 47.333 \\
        46.667 & 47.167 & 47.667 & 48.167 & 48.667 & 49.167 & 49.667 & 50.167 & 50.667 & 50.667 \\
        50.    & 50.5   & 51.    & 51.5   & 52.    & 52.5   & 53.    & 53.5   & 54.    & 54.    \\
        53.333 & 53.833 & 54.333 & 54.833 & 55.333 & 55.833 & 56.333 & 56.833 & 57.333 & 57.333 \\
        55.    & 55.5   & 56.    & 56.5   & 57.    & 57.5   & 58.    & 58.5   & 59.    & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 注意，可用的插值方法
| trt.ResizeMode 名 | 说明       |
| :---------------- | :--------- |
| NEAREST           | 最邻近插值 |
| LINEAR            | 线性插值   |

---
### align_corners
```python
    rsz = network.add_resize(inputTensor)
    rsz.shape = (cOut, hOut, wOut)
    rsz.align_corners = True                                                                        # 是否角落对齐，默认值为 False
    print("rsz->", rsz.get_output(0).shape)                                                         # (cOut, hOut, wOut)
```

+ 输出张量 (3, 6, 10)，其中指定 align_corners=True
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         0. &  0. &  0. &  1. &  1. &  2. &  2. &  3. &  3. &  4. \\
         0. &  0. &  0. &  1. &  1. &  2. &  2. &  3. &  3. &  4. \\
         5. &  5. &  5. &  6. &  6. &  7. &  7. &  8. &  8. &  9. \\
         5. &  5. &  5. &  6. &  6. &  7. &  7. &  8. &  8. &  9. \\
        10. & 10. & 10. & 11. & 11. & 12. & 12. & 13. & 13. & 14. \\
        15. & 15. & 15. & 16. & 16. & 17. & 17. & 18. & 18. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 20. & 20. & 21. & 21. & 22. & 22. & 23. & 23. & 24. \\
        20. & 20. & 20. & 21. & 21. & 22. & 22. & 23. & 23. & 24. \\
        25. & 25. & 25. & 26. & 26. & 27. & 27. & 28. & 28. & 29. \\
        25. & 25. & 25. & 26. & 26. & 27. & 27. & 28. & 28. & 29. \\
        30. & 30. & 30. & 31. & 31. & 32. & 32. & 33. & 33. & 34. \\
        35. & 35. & 35. & 36. & 36. & 37. & 37. & 38. & 38. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 40. & 40. & 41. & 41. & 42. & 42. & 43. & 43. & 44. \\
        40. & 40. & 40. & 41. & 41. & 42. & 42. & 43. & 43. & 44. \\
        45. & 45. & 45. & 46. & 46. & 47. & 47. & 48. & 48. & 49. \\
        45. & 45. & 45. & 46. & 46. & 47. & 47. & 48. & 48. & 49. \\
        50. & 50. & 50. & 51. & 51. & 52. & 52. & 53. & 53. & 54. \\
        55. & 55. & 55. & 56. & 56. & 57. & 57. & 58. & 58. & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 注意：是否对齐角落的示意图如下，顺便补充一个 Torch 中角落不对齐的情况。
  TensorRT （或 Torch）中 align_corners = True：原始图像四个角栅格的中心与新图像的四个角栅格的中心对齐
  TensorRT 中 align_corners = False：原始图像和新图像缩放至相等边长，然后将原始图像左上角栅格的中心与新图像左上角栅格的中心对齐，不做进一步的缩放和调整
  Torch 中 align_corners = False：原始图像的四个角 （而不是栅格的中心点） 与新图像的四个角对齐
<div align="center" >
<img src="./ResizeLayer-oldGrid.png" alt="ResizeLayer-oldGrid" style="zoom:60%;" />
<img src="./ResizeLayer-newGrid.png" alt="ResizeLayer-newGrid" style="zoom:60%;" />
</div>

<div align="center" >
<img src="./ResizeLayer-TRTAlignTrue.png" alt="ResizeLayer-TRTAlignTrue" style="zoom:60%;" />
<img src="./ResizeLayer-TRTAlignFalse.png" alt="ResizeLayer-TRTAlignFalse" style="zoom:60%;" />
<img src="./ResizeLayer-TorchAlignFlase.png" alt="ResizeLayer-TorchAlignFlase" style="zoom:60%;" />
</div>

---
### 静态 set_input
```python
# 调整部分参数
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))      # 使用显式 batch 模式
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (1, cIn, hIn, wIn))
```
```python
    rsz = network.add_resize(inputTensor)
    constantLayer = network.add_constant((4,), np.array([1,cOut,hOut,wOut],dtype=np.int32))
    rsz.set_input(1, constantLayer.get_output(0))
    print("rsz->", rsz.get_output(0).shape)
```

+ 输出张量 (3, 6, 10)，与初始代码相同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         0. &  0. &  1. &  1. &  2. &  2. &  3. &  3. &  4. &  4. \\
         0. &  0. &  1. &  1. &  2. &  2. &  3. &  3. &  4. &  4. \\
         5. &  5. &  6. &  6. &  7. &  7. &  8. &  8. &  9. &  9. \\
        10. & 10. & 11. & 11. & 12. & 12. & 13. & 13. & 14. & 14. \\
        10. & 10. & 11. & 11. & 12. & 12. & 13. & 13. & 14. & 14. \\
        15. & 15. & 16. & 16. & 17. & 17. & 18. & 18. & 19. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 20. & 21. & 21. & 22. & 22. & 23. & 23. & 24. & 24. \\
        20. & 20. & 21. & 21. & 22. & 22. & 23. & 23. & 24. & 24. \\
        25. & 25. & 26. & 26. & 27. & 27. & 28. & 28. & 29. & 29. \\
        30. & 30. & 31. & 31. & 32. & 32. & 33. & 33. & 34. & 34. \\
        30. & 30. & 31. & 31. & 32. & 32. & 33. & 33. & 34. & 34. \\
        35. & 35. & 36. & 36. & 37. & 37. & 38. & 38. & 39. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 40. & 41. & 41. & 42. & 42. & 43. & 43. & 44. & 44. \\
        40. & 40. & 41. & 41. & 42. & 42. & 43. & 43. & 44. & 44. \\
        45. & 45. & 46. & 46. & 47. & 47. & 48. & 48. & 49. & 49. \\
        50. & 50. & 51. & 51. & 52. & 52. & 53. & 53. & 54. & 54. \\
        50. & 50. & 51. & 51. & 52. & 52. & 53. & 53. & 54. & 54. \\
        55. & 55. & 56. & 56. & 57. & 57. & 58. & 58. & 59. & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### 动态 set_input
+ 见 shuffle 层“动态 set_input”

<div style="page-break-after:always;"></div>
## RNN 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 4                                                                                         # 输入张量 HWC
wIn     = 7
cIn     = 3
lenH    = 5                                                                                         # 隐藏层元素宽度
data    = np.ones(cIn*hIn*wIn,dtype=np.float32).reshape(cIn,hIn,wIn)                                # 输入张量
weight  = np.ones((wIn+lenH,lenH),dtype=np.float32)                                                 # RNN 变换阵（TensorFlow 格式）
bias    = np.zeros(lenH, dtype=np.float32)                                                          # RNN 偏置

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替换部分
    rnnV2 = network.add_rnn_v2(inputTensor, 1, lenH, hIn, trt.RNNOperation.RELU)                    # 1 层 ReLU 型 RNN，隐藏层元素宽 lenH，序列长度 hIn，单词编码宽度 wIn，batchSize 为 cIn
    wX, wH = np.split(weight, [wIn])
    rnnV2.set_weights_for_gate(0, trt.RNNGateType.INPUT, True, wX.transpose())                      # 0 层 INPUT 门，输入元 X 变换阵，wX.shape=(lenH,wIn)
    rnnV2.set_weights_for_gate(0, trt.RNNGateType.INPUT, False, wH.transpose())                     # 0 层 INPUT 门，隐藏元 H 变换阵，wH.shape=(lenH,lenH)
    rnnV2.set_bias_for_gate(0, trt.RNNGateType.INPUT, True, bias)                                   # 0 层 INPUT 门，输入元 X 偏置，bX.shape=(lenH,)
    rnnV2.set_bias_for_gate(0, trt.RNNGateType.INPUT, False, np.zeros(lenH,dtype=np.float32))       # 0 层 INPUT 门，隐藏元 H 偏置，bH.shape=(lenH,)
    print("rnnV2->", rnnV2.get_output(0).shape, rnnV2.get_output(1).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(rnnV2.get_output(0))
    network.mark_output(rnnV2.get_output(1))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
    out2_h  = np.empty(engine.get_binding_shape(2),dtype = trt.nptype(engine.get_binding_dtype(2)))
    out2_d  = cuda.mem_alloc(out2_h.nbytes)
            
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d),int(out2_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    cuda.memcpy_dtoh_async(out2_h, out2_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    print("out2_h:", out2_h.shape)
    print(out2_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (3, 4, 7)，3 个独立输入，每个输入 4 个单词，每个单词 7 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        1. & 1. & 1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 0 形状 (3, 4, 5)，3 个独立输出，每个输出 4 个隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \textcolor[rgb]{0,0.5,0}{7.} & \textcolor[rgb]{0,0.5,0}{7.} & \textcolor[rgb]{0,0.5,0}{7.} & \textcolor[rgb]{0,0.5,0}{7.} & \textcolor[rgb]{0,0.5,0}{7.} \\
        \textcolor[rgb]{0,0,1}{42.} & \textcolor[rgb]{0,0,1}{42.} & \textcolor[rgb]{0,0,1}{42.} & \textcolor[rgb]{0,0,1}{42.} & \textcolor[rgb]{0,0,1}{42.} \\
         217. &  217. &  217. &  217. &  217. \\
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
    \left[\begin{matrix}
           7. &    7. &    7. &    7. &    7. \\
          42. &   42. &   42. &   42. &   42. \\
         217. &  217. &  217. &  217. &  217. \\
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
    \left[\begin{matrix}
           7. &    7. &    7. &    7. &    7. \\
          42. &   42. &   42. &   42. &   42. \\
         217. &  217. &  217. &  217. &  217. \\
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (3, 1, 5)，3 个独立输出，每个输出 1 个最终隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
    \left[\begin{matrix}
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
    \left[\begin{matrix}
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：默认 RNN 是单向、隐藏状态全 0、对输入张量作了线性变换的
$$
\begin{aligned}
h_{1}&=\textbf{ReLU}\left(W_{i,X}\cdot x_{1}+W_{i,H}\cdot h_{0}+b_{i,X}+b_{i,H}\right)\\
&=\textbf{ReLU}
\left(
  \left[\begin{matrix}
   1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1
  \end{matrix}\right]
  \left[\begin{matrix}
   1\\1\\1\\1\\1\\1\\1
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1
  \end{matrix}\right]
  \left[\begin{matrix}
   0\\0\\0\\0\\0\\0\\0
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   0\\0\\0\\0\\0\\0\\0
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   0\\0\\0\\0\\0\\0\\0
  \end{matrix}\right]
\right)\\
&=\textbf{ReLU}\left(\left(7,7,7,7,7\right)^\mathrm{T}\right)\\
&=\left(\textcolor[rgb]{0,0.5,0}{7},\textcolor[rgb]{0,0.5,0}{7},\textcolor[rgb]{0,0.5,0}{7},\textcolor[rgb]{0,0.5,0}{7},\textcolor[rgb]{0,0.5,0}{7}
  \right)^\mathrm{T}\\
\\\hfill\\
h_{2}&=\textbf{ReLU}\left(W_{i,X}\cdot x_{2}+W_{i,H}\cdot h_{1}+b_{i,X}+b_{i,H}\right)\\
&=\textbf{ReLU}
\left(
  \left[\begin{matrix}
   1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1   
  \end{matrix}\right]
  \left[\begin{matrix}
   1\\1\\1\\1\\1\\1\\1
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1
  \end{matrix}\right]
  \left[\begin{matrix}
   7\\7\\7\\7\\7\\7\\7
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   0\\0\\0\\0\\0\\0\\0
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   0\\0\\0\\0\\0\\0\\0
  \end{matrix}\right]
\right)\\
&=\textbf{ReLU}\left(\left(42,42,42,42,42\right)^\mathrm{T}\right)\\
&=\left(
    \textcolor[rgb]{0,0,1}{42},\textcolor[rgb]{0,0,1}{42},\textcolor[rgb]{0,0,1}{42},\textcolor[rgb]{0,0,1}{42},\textcolor[rgb]{0,0,1}{42}
  \right)^\mathrm{T}
\end{aligned}
$$

---
### seq_lengths
```python
    rnnV2 = network.add_rnn_v2(inputTensor, 1, lenH, hIn, trt.RNNOperation.RELU)                    # 替换部分
    length = network.add_constant((3,),np.array([4,3,2],dtype=np.int32))                            # 长度张量为 1 维，指定一个 batch 中各输入张量的长度，不大于 hIn 即可
    rnnV2.seq_lengths = length.get_output(0)														# 长度张量交给 rnnV2 层
    wX, wH = np.split(weight, [wIn])
    rnnV2.set_weights_for_gate(0, trt.RNNGateType.INPUT, True, wX.transpose())
    rnnV2.set_weights_for_gate(0, trt.RNNGateType.INPUT, False, wH.transpose())
    rnnV2.set_bias_for_gate(0, trt.RNNGateType.INPUT, True, bias)
    rnnV2.set_bias_for_gate(0, trt.RNNGateType.INPUT, False, np.zeros(lenH,dtype=np.float32))
    print("rnnV2->", rnnV2.get_output(0).shape, rnnV2.get_output(1).shape)
```

+ 输出张量 0 形状 (3, 4, 5)，同一 batch 中 长度不足的独立输入计算结果为 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
           7. &    7. &    7. &    7. &    7. \\
          42. &   42. &   42. &   42. &   42. \\
         217. &  217. &  217. &  217. &  217. \\
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
    \left[\begin{matrix}
           7. &    7. &    7. &    7. &    7. \\
          42. &   42. &   42. &   42. &   42. \\
         217. &  217. &  217. &  217. &  217. \\
           0. &    0. &    0. &    0. &    0.
    \end{matrix}\right]
    \left[\begin{matrix}
           7. &    7. &    7. &    7. &    7. \\
          42. &   42. &   42. &   42. &   42. \\
           0. &    0. &    0. &    0. &    0. \\
           0. &    0. &    0. &    0. &    0.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (3, 1, 5)，3 个独立输出，记录每个独立输入的末状态
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
    \left[\begin{matrix}
        217.  & 217.  & 217.  & 217.  &  217.
    \end{matrix}\right]
    \left[\begin{matrix}
          42. &   42. &   42. &   42. &   42.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### num_layers & hidden_size & max_seq_length & data_length & op
```python
    rnnV2 = network.add_rnn_v2(inputTensor, 1, lenH, hIn, trt.RNNOperation.LSTM)                    # 替换部分
    rnnV2.op = trt.RNNOperation.RELU                                                                # RNN 类型
    print(rnnV2.num_layers,rnnV2.hidden_size,rnnV2.max_seq_length,rnnV2.data_length)                # 仅供输出，不能调节

    wX, wH = np.split(weight, [wIn])
    rnnV2.set_weights_for_gate(0, trt.RNNGateType.INPUT, True, wX.transpose())
    rnnV2.set_weights_for_gate(0, trt.RNNGateType.INPUT, False, wH.transpose())
    rnnV2.set_bias_for_gate(0, trt.RNNGateType.INPUT, True, bias)
    rnnV2.set_bias_for_gate(0, trt.RNNGateType.INPUT, False, np.zeros(lenH,dtype=np.float32))
    print("rnnV2->", rnnV2.get_output(0).shape, rnnV2.get_output(1).shape)
```

+ 输出张量 0 形状 (3, 4, 5)，与初始代码相同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \textcolor[rgb]{0,0.5,0}{7.} & \textcolor[rgb]{0,0.5,0}{7.} & \textcolor[rgb]{0,0.5,0}{7.} & \textcolor[rgb]{0,0.5,0}{7.} & \textcolor[rgb]{0,0.5,0}{7.} \\
        \textcolor[rgb]{0,0,1}{42.} & \textcolor[rgb]{0,0,1}{42.} & \textcolor[rgb]{0,0,1}{42.} & \textcolor[rgb]{0,0,1}{42.} & \textcolor[rgb]{0,0,1}{42.} \\
         217. &  217. &  217. &  217. &  217. \\
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
    \left[\begin{matrix}
           7. &    7. &    7. &    7. &    7. \\
          42. &   42. &   42. &   42. &   42. \\
         217. &  217. &  217. &  217. &  217. \\
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
    \left[\begin{matrix}
           7. &    7. &    7. &    7. &    7. \\
          42. &   42. &   42. &   42. &   42. \\
         217. &  217. &  217. &  217. &  217. \\
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (3, 1, 5)，与初始代码相同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
    \left[\begin{matrix}
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
    \left[\begin{matrix}
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 注意，可用的 RNN 模式
| trt.RNNOperation 名 | 说明               |
| :------------------ | :----------------- |
| RELU                | 单门 ReLU 激活 RNN |
| TANH                | 单门 tanh 激活 RNN |
| LSTM                | 4 门 LSTM          |
| GRU                 | 3 门 GRU           |

---
### input mode
```python
# 调整部分参数
hIn     = 4
#wIn     = 7
cIn     = 3
lenH    = 5
wIn     = lenH                                                                                      # 输入宽度强制等于隐藏层宽度
weight  = np.ones((lenH,lenH),dtype=np.float32)                                                     # 只剩 wH
bias    = np.zeros(lenH, dtype=np.float32)
```
```python
    rnnV2 = network.add_rnn_v2(inputTensor, 1, lenH, hIn, trt.RNNOperation.RELU)                    # 替换部分
    rnnV2.input_mode = trt.RNNInputMode.SKIP                                                        # 是否对输入张量线性变换，默认值为 trt.RNNInputMode.LINEAR

    rnnV2.set_weights_for_gate(0, trt.RNNGateType.INPUT, False, weight.transpose())                 # 只需要 wH
    rnnV2.set_bias_for_gate(0, trt.RNNGateType.INPUT, True, bias)                                   # 两个 bias 必须都设置，就算全都是 0
    rnnV2.set_bias_for_gate(0, trt.RNNGateType.INPUT, False, np.zeros(lenH,dtype=np.float32))
    print("rnnV2->", rnnV2.get_output(0).shape, rnnV2.get_output(1).shape)
```

+ 输出张量 0 形状 (3, 4, 5)，3 个独立输出，每个输出 4 个隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \textcolor[rgb]{0,0.5,0}{1.} & \textcolor[rgb]{0,0.5,0}{1.} & \textcolor[rgb]{0,0.5,0}{1.} & \textcolor[rgb]{0,0.5,0}{1.} & \textcolor[rgb]{0,0.5,0}{1.} \\
        \textcolor[rgb]{0,0,1}{6.} & \textcolor[rgb]{0,0,1}{6.} & \textcolor[rgb]{0,0,1}{6.} & \textcolor[rgb]{0,0,1}{6.} & \textcolor[rgb]{0,0,1}{6.} \\
         31. &  31. &  31. &  31. &  31. \\
        156. & 156. & 156. & 156. & 156.
    \end{matrix}\right]
    \left[\begin{matrix}
          1. &   1. &   1. &   1. &   1. \\
          6. &   6. &   6. &   6. &   6. \\
         31. &  31. &  31. &  31. &  31. \\
        156. & 156. & 156. & 156. & 156.
    \end{matrix}\right]
    \left[\begin{matrix}
          1. &   1. &   1. &   1. &   1. \\
          6. &   6. &   6. &   6. &   6. \\
         31. &  31. &  31. &  31. &  31. \\
        156. & 156. & 156. & 156. & 156.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (3, 1, 5)，3 个独立输出，每个输出 1 个最终隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        156. & 156. & 156. & 156. & 156.
    \end{matrix}\right]
    \left[\begin{matrix}
        156. & 156. & 156. & 156. & 156.
    \end{matrix}\right]
    \left[\begin{matrix}
        156. & 156. & 156. & 156. & 156.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：
$$
\begin{aligned}
h_{1}&=\textbf{ReLU}\left(x_{1}+W_{i,H}\cdot h_{0}+b_{i,X}+b_{i,H}\right)\\
&=\textbf{ReLU}
\left(
  \left[\begin{matrix}
   1\\1\\1\\1\\1\\1\\1
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1
  \end{matrix}\right]
  \left[\begin{matrix}
   0\\0\\0\\0\\0\\0\\0
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   0\\0\\0\\0\\0\\0\\0
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   0\\0\\0\\0\\0\\0\\0
  \end{matrix}\right]
\right)\\
&=\textbf{ReLU}\left(\left(1,1,1,1,1\right)^\mathrm{T}\right)\\
&=\left(
    \textcolor[rgb]{0,0.5,0}{1},\textcolor[rgb]{0,0.5,0}{1},\textcolor[rgb]{0,0.5,0}{1},\textcolor[rgb]{0,0.5,0}{1},\textcolor[rgb]{0,0.5,0}{1},
  \right)^\mathrm{T}
\\\hfill\\
h_{2}&=\textbf{ReLU}\left(x_{2}+W_{i,H}\cdot h_{1}+b_{i,X}+b_{i,H}\right)\\
&=\textbf{ReLU}
\left(
  \left[\begin{matrix}
   1\\1\\1\\1\\1\\1\\1
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1
  \end{matrix}\right]
  \left[\begin{matrix}
   1\\1\\1\\1\\1\\1\\1
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   0\\0\\0\\0\\0\\0\\0
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   0\\0\\0\\0\\0\\0\\0
  \end{matrix}\right]
\right)\\
&=\textbf{ReLU}\left(\left(6,6,6,6,6\right)^\mathrm{T}\right)\\
&=\left(
    \textcolor[rgb]{0,0,1}{6},\textcolor[rgb]{0,0,1}{6},\textcolor[rgb]{0,0,1}{6},\textcolor[rgb]{0,0,1}{6},\textcolor[rgb]{0,0,1}{6}
  \right)^\mathrm{T}
\end{aligned}
$$

+ 注意，可用的输入张量处理方法
| trt.RNNInputMode 名 | 说明                                         |
| :------------------ | :------------------------------------------- |
| LINEAR              | 对输入张量 x 做线性变换                      |
| SKIP                | 不对输入张量 x 做线性变换（要求wIn == lenH） |

---
### direction
```python
# 调整部分参数
weightF = np.ones((wIn+lenH,lenH),dtype=np.float32)                                                 # 正向变换阵
weightB = np.ones((wIn+lenH,lenH),dtype=np.float32)                                                 # 反向变换阵
biasF   = np.zeros(lenH, dtype=np.float32)                                                          # 正向偏置
biasB   = np.zeros(lenH, dtype=np.float32)                                                          # 反向偏置
```
```python
    rnnV2 = network.add_rnn_v2(inputTensor, 1, lenH, hIn, trt.RNNOperation.RELU)                    # 替换部分
    rnnV2.direction     = trt.RNNDirection.BIDIRECTION                                              # RNN 方向，默认值 trt.RNNDirection.UNIDIRECTION
    
    wX, wH = np.split(weightF, [wIn])
    rnnV2.set_weights_for_gate(0, trt.RNNGateType.INPUT, True, wX.transpose())
    rnnV2.set_weights_for_gate(0, trt.RNNGateType.INPUT, False, wH.transpose())
    rnnV2.set_bias_for_gate(0, trt.RNNGateType.INPUT, True, biasF)
    rnnV2.set_bias_for_gate(0, trt.RNNGateType.INPUT, False, np.zeros(lenH,dtype=np.float32))
    wX, wH = np.split(weightB, [wIn])
    rnnV2.set_weights_for_gate(1, trt.RNNGateType.INPUT, True, wX.transpose())                      # 反向为第 1 层
    rnnV2.set_weights_for_gate(1, trt.RNNGateType.INPUT, False, wH.transpose())
    rnnV2.set_bias_for_gate(1, trt.RNNGateType.INPUT, True, biasB)
    rnnV2.set_bias_for_gate(1, trt.RNNGateType.INPUT, False, np.zeros(lenH,dtype=np.float32))
    print("rnnV2->", rnnV2.get_output(0).shape, rnnV2.get_output(1).shape)
```

+ 输出张量 0 形状 (3, 4, 10)，3 个独立输出，每个输出 4 个隐藏状态，每个隐藏状态 5 维坐标，2 个方向并排放置
$$
\left[\begin{matrix}
    \left[\begin{matrix}
           7. &    7. &    7. &    7. &    7. & 1092. & 1092. & 1092. & 1092. & 1092. \\
          42. &   42. &   42. &   42. &   42. &  217. &  217. &  217. &  217. &  217. \\
         217. &  217. &  217. &  217. &  217. &   42. &   42. &   42. &   42. &   42. \\
        1092. & 1092. & 1092. & 1092. & 1092. &    7. &    7. &    7. &    7. &    7.
    \end{matrix}\right]\\
    \left[\begin{matrix}
               7. &    7. &    7. &    7. &    7. & 1092. & 1092. & 1092. & 1092. & 1092. \\
          42. &   42. &   42. &   42. &   42. &  217. &  217. &  217. &  217. &  217. \\
         217. &  217. &  217. &  217. &  217. &   42. &   42. &   42. &   42. &   42. \\
        1092. & 1092. & 1092. & 1092. & 1092. &    7. &    7. &    7. &    7. &    7.
    \end{matrix}\right]\\
    \left[\begin{matrix}
               7. &    7. &    7. &    7. &    7. & 1092. & 1092. & 1092. & 1092. & 1092. \\
          42. &   42. &   42. &   42. &   42. &  217. &  217. &  217. &  217. &  217. \\
         217. &  217. &  217. &  217. &  217. &   42. &   42. &   42. &   42. &   42. \\
        1092. & 1092. & 1092. & 1092. & 1092. &    7. &    7. &    7. &    7. &    7.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (3, 2, 5)，3 个独立输出，每个输出 1 个最终隐藏状态，每个隐藏状态 5 维坐标，2 个方向分行放置
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1092. & 1092. & 1092. & 1092. & 1092. \\
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
    \left[\begin{matrix}
        1092. & 1092. & 1092. & 1092. & 1092. \\
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
    \left[\begin{matrix}
        1092. & 1092. & 1092. & 1092. & 1092. \\
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 注意，可用方向参数
| trt.RNNDirection 名 | 说明     |
| :------------------ | :------- |
| UNIDIRECTION        | 单向 RNN |
| BIDIRECTION         | 双向 RNN |

---
### hidden_state
```python
    rnnV2 = network.add_rnn_v2(inputTensor, 1, lenH, hIn, trt.RNNOperation.RELU)                    # 替换部分
    h0                 = network.add_constant((cIn,1,lenH),np.ones((cIn,1,lenH),dtype=np.float32))  # 初始隐藏状态
    rnnV2.hidden_state = h0.get_output(0)                                                           # 默认值为全 0
        
    wX, wH = np.split(weight, [wIn])
    rnnV2.set_weights_for_gate(0, trt.RNNGateType.INPUT, True, wX.transpose())
    rnnV2.set_weights_for_gate(0, trt.RNNGateType.INPUT, False, wH.transpose())
    rnnV2.set_bias_for_gate(0, trt.RNNGateType.INPUT, True, bias)
    rnnV2.set_bias_for_gate(0, trt.RNNGateType.INPUT, False, np.zeros(lenH,dtype=np.float32))
    print("rnnV2->", rnnV2.get_output(0).shape, rnnV2.get_output(1).shape)
```

+ 输出张量 0 形状 (3, 4, 5)，3 个独立输出，每个输出 4 个隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \textcolor[rgb]{0,0.5,0}{12.} & \textcolor[rgb]{0,0.5,0}{12.} & \textcolor[rgb]{0,0.5,0}{12.} & \textcolor[rgb]{0,0.5,0}{12.} & \textcolor[rgb]{0,0.5,0}{12.} \\
        \textcolor[rgb]{0,0,1}{67.} & \textcolor[rgb]{0,0,1}{12.} & \textcolor[rgb]{0,0,1}{12.} & \textcolor[rgb]{0,0,1}{12.} & \textcolor[rgb]{0,0,1}{12.} \\
         342. &  342. &  342. &  342. &  342. \\
        1717. & 1717. & 1717. & 1717. & 1717.
    \end{matrix}\right]
    \left[\begin{matrix}
          12. &   12. &   12. &   12. &   12. \\
          67. &   67. &   67. &   67. &   67. \\
         342. &  342. &  342. &  342. &  342. \\
        1717. & 1717. & 1717. & 1717. & 1717.
    \end{matrix}\right]
    \left[\begin{matrix}
          12. &   12. &   12. &   12. &   12. \\
          67. &   67. &   67. &   67. &   67. \\
         342. &  342. &  342. &  342. &  342. \\
        1717. & 1717. & 1717. & 1717. & 1717.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (3, 1, 5)，3 个独立输出，每个输出 1 个最终隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1717. & 1717. & 1717. & 1717. & 1717.
    \end{matrix}\right]
    \left[\begin{matrix}
        1717. & 1717. & 1717. & 1717. & 1717.
    \end{matrix}\right]
    \left[\begin{matrix}
        1717. & 1717. & 1717. & 1717. & 1717.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：
$$
\begin{aligned}
h_{1}&=\textbf{ReLU}\left(W_{i,X}\cdot x_{1}+W_{i,H}\cdot h_{0}+b_{i,X}+b_{i,H}\right)\\
&=\textbf{ReLU}
\left(
  \left[\begin{matrix}
   1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1
  \end{matrix}\right]
  \left[\begin{matrix}
   1\\1\\1\\1\\1\\1\\1
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1
  \end{matrix}\right]
  \left[\begin{matrix}
   1\\1\\1\\1\\1\\1\\1
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   0\\0\\0\\0\\0\\0\\0
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   0\\0\\0\\0\\0\\0\\0
  \end{matrix}\right]
\right)\\
&=\textbf{ReLU}\left(\left(12,12,12,12,12\right)^\mathrm{T}\right)\\
&=\left(
    \textcolor[rgb]{0,0.5,0}{12},\textcolor[rgb]{0,0.5,0}{12},\textcolor[rgb]{0,0.5,0}{12},\textcolor[rgb]{0,0.5,0}{12},\textcolor[rgb]{0,0.5,0}{12}
  \right)^\mathrm{T}
\\\hfill\\
h_{2}&=\textbf{ReLU}\left(W_{i,X}\cdot x_{2}+W_{i,H}\cdot h_{1}+b_{i,X}+b_{i,H}\right)\\
&=\textbf{ReLU}
\left(
  \left[\begin{matrix}
   1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1 & 1 & 1   
  \end{matrix}\right]
  \left[\begin{matrix}
   1\\1\\1\\1\\1\\1\\1
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1\\1 & 1 & 1 & 1 & 1
  \end{matrix}\right]
  \left[\begin{matrix}
   12\\12\\12\\12\\12\\12\\12
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   0\\0\\0\\0\\0\\0\\0
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   0\\0\\0\\0\\0\\0\\0
  \end{matrix}\right]
\right)\\
&=\textbf{ReLU}\left(\left(67,67,67,67,67\right)^\mathrm{T}\right)\\
&=\left(
    \textcolor[rgb]{0,0,1}{67},\textcolor[rgb]{0,0,1}{67},\textcolor[rgb]{0,0,1}{67},\textcolor[rgb]{0,0,1}{67},\textcolor[rgb]{0,0,1}{67}
  \right)^\mathrm{T}
\end{aligned}
$$

---
### cell state（单向 LSTM 的例子）
```python
# 调整部分参数
weight  = np.ones((wIn+lenH,lenH*4),dtype=np.float32)                                               # RNN 变换阵（TensorFlow 格式）
bias    = np.zeros(lenH*4, dtype=np.float32)                                                        # RNN 偏置（TensorFlow 格式）
```
```python
    rnnV2 = network.add_rnn_v2(inputTensor, 1, lenH, hIn, trt.RNNOperation.LSTM)                    # 替换部分
    rnnV2.direction    = trt.RNNDirection.UNIDIRECTION
    rnnV2.input_mode   = trt.RNNInputMode.LINEAR
    
    h0                 = network.add_constant((cIn,1,lenH),np.zeros((cIn,1,lenH),dtype=np.float32)) # 设置初始隐藏状态
    rnnV2.hidden_state = h0.get_output(0)
    c0                 = network.add_constant((cIn,1,lenH),np.zeros((cIn,1,lenH),dtype=np.float32)) # 设置初始细胞状态
    rnnV2.cell_state   = c0.get_output(0)                                                           # 默认值为全 0
        
    gateList = [trt.RNNGateType.INPUT, trt.RNNGateType.CELL, trt.RNNGateType.FORGET, trt.RNNGateType.OUTPUT]
    wX, wH = np.split(weight, [wIn])
    wX = [w.transpose().reshape(-1) for w in np.split(wX, 4, axis=1)]
    wH = [w.transpose().reshape(-1) for w in np.split(wH, 4, axis=1)]
    bX = np.split(bias, 4)
    for kind, wx, wh, bx in zip(gateList, wX, wH, bX):
        rnnV2.set_weights_for_gate(0, kind, True, wx)
        rnnV2.set_bias_for_gate(0, kind, True, bx)
        rnnV2.set_weights_for_gate(0, kind, False, wh)
        rnnV2.set_bias_for_gate(0, kind, False, np.zeros(lenH, dtype=np.float32))

    print("rnnV2->", rnnV2.get_output(0).shape, rnnV2.get_output(1).shape, rnnV2.get_output(2).shape)
```

+ 输出张量 0 形状 (3, 4, 5)，3 个独立输出，每个输出 4 个隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \textcolor[rgb]{0,0.5,0}{0.760517} & \textcolor[rgb]{0,0.5,0}{0.760517} & \textcolor[rgb]{0,0.5,0}{0.760517} & \textcolor[rgb]{0,0.5,0}{0.760517} & \textcolor[rgb]{0,0.5,0}{0.760517} \\
        \textcolor[rgb]{0,0,1}{0.963940} & \textcolor[rgb]{0,0,1}{0.963940} & \textcolor[rgb]{0,0,1}{0.963940} & \textcolor[rgb]{0,0,1}{0.963940} & \textcolor[rgb]{0,0,1}{0.963940} \\
        0.995037 & 0.995037 & 0.995037 & 0.995037 & 0.995037 \\
        0.999321 & 0.999321 & 0.999321 & 0.999321 & 0.999321 \\
    \end{matrix}\right]
    \left[\begin{matrix}
        0.760517 & 0.760517 & 0.760517 & 0.7605171& 0.7605171\\
        0.963940 & 0.963940 & 0.963940 & 0.963940 & 0.963940 \\
        0.995037 & 0.995037 & 0.995037 & 0.995037 & 0.995037 \\
        0.999321 & 0.999321 & 0.999321 & 0.999321 & 0.999321
    \end{matrix}\right]
    \left[\begin{matrix}
        0.760517 & 0.760517 & 0.760517 & 0.7605171& 0.7605171\\
        0.963940 & 0.963940 & 0.963940 & 0.963940 & 0.963940 \\
        0.995037 & 0.995037 & 0.995037 & 0.995037 & 0.995037 \\
        0.999321 & 0.999321 & 0.999321 & 0.999321 & 0.999321
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (3, 1, 5)，3 个独立输出，每个输出 1 个最终隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0.999321 & 0.999321 & 0.999321 & 0.999321 & 0.999321
    \end{matrix}\right]
    \left[\begin{matrix}
        0.999321 & 0.999321 & 0.999321 & 0.999321 & 0.999321
    \end{matrix}\right]
    \left[\begin{matrix}
        0.999321 & 0.999321 & 0.999321 & 0.999321 & 0.999321
    \end{matrix}\right]
\end{matrix}\right]
$$

+ *rnnV2.get_output(2)* 输出形状 (3, 1, 5)，3 个独立输出，每个输出 1 个最终细胞状态，每个细胞状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        3.998999 & 3.998999 & 3.998999 & 3.998999 & 3.998999
    \end{matrix}\right]
    \left[\begin{matrix}
        3.998999 & 3.998999 & 3.998999 & 3.998999 & 3.998999
    \end{matrix}\right]
    \left[\begin{matrix}
        3.998999 & 3.998999 & 3.998999 & 3.998999 & 3.998999
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：
$$
\begin{aligned}
I_{1}=F_{1}=O_{1}=\textbf{sigmoid}\left(W_{?,X}\cdot x_{1}+W_{?,H}\cdot h_{0}+b_{i,X}+b_{i,H}\right)&=
  \left(0.999088,0.999088,0.999088,0.999088,0.999088\right)^\mathrm{T}\\
C_{1}=               \textbf{tanh}\left(W_{C,X}\cdot x_{1}+W_{C,H}\cdot h_{0}+b_{C,X}+b_{C,H}\right)&=
  \left(0.999998,0.999998,0.999998,0.999998,0.999998\right)^\mathrm{T}\\
c_{1}=F_{1}\cdot c_{0}+I_{1}\cdot C_{1}&=\left(0.999087,0.999087,0.999087,0.999087,0.999087\right)^\mathrm{T}\\
h_{1}=O_{1}\cdot \textbf{tanh}\left(c_{1}\right)&=\left(
  \textcolor[rgb]{0,0.5,0}{0.760517},\textcolor[rgb]{0,0.5,0}{0.760517},\textcolor[rgb]{0,0.5,0}{0.760517},\textcolor[rgb]{0,0.5,0}{0.760517},\textcolor[rgb]{0,0.5,0}{0.760517}
                                                  \right)^\mathrm{T}\\
\hfill\\
I_{2}=F_{2}=O_{2}=\textbf{sigmoid}\left(W_{?,X}\cdot x_{2}+W_{?,H}\cdot h_{1}+b_{i,X}+b_{i,H}\right)&=
  \left(0.999979,0.999979,0.999979,0.999979,0.999979\right)^\mathrm{T}\\
C_{2}=               \textbf{tanh}\left(W_{C,X}\cdot x_{2}+W_{C,H}\cdot h_{1}+b_{C,X}+b_{C,H}\right)&=
  \left(0.999999,0.999999,0.999999,0.999999,0.999999\right)^\mathrm{T}\\
c_{2}=F_{2}\cdot c_{1}+I_{2}\cdot C_{2}&=\left(1.999046,1.999046,1.999046,1.999046,1.999046\right)^\mathrm{T}\\
h_{2}=O_{2}\cdot \textbf{tanh}\left(c_{2}\right)&=\left(
  \textcolor[rgb]{0,0,1}{0.963940},\textcolor[rgb]{0,0,1}{0.963940},\textcolor[rgb]{0,0,1}{0.963940},\textcolor[rgb]{0,0,1}{0.963940},\textcolor[rgb]{0,0,1}{0.963940}
                                                  \right)^\mathrm{T}\\
\end{aligned}
$$

---
### 双向 LSTM 的例子
```python
# 调整部分参数
weightF = np.ones((wIn+lenH,lenH*4),dtype=np.float32)                                               # 正向变换阵（TensorFlow 格式）
weightB = np.ones((wIn+lenH,lenH*4),dtype=np.float32)                                               # 反向变换阵（TensorFlow 格式）
biasF    = np.zeros(lenH*4, dtype=np.float32)                                                       # 正向偏置
biasB    = np.zeros(lenH*4, dtype=np.float32)                                                       # 反向偏置
```
```python
    rnnV2 = network.add_rnn_v2(inputTensor, 1, lenH, hIn, trt.RNNOperation.LSTM)
    rnnV2.direction    = trt.RNNDirection.BIDIRECTION
    rnnV2.input_mode   = trt.RNNInputMode.LINEAR
    
    h0                 = network.add_constant((cIn,2,lenH),np.zeros(cIn*2*lenH,dtype=np.float32))   # 初始隐藏状态和细胞状态有单向 LSTM 的 2 倍长
    rnnV2.hidden_state = h0.get_output(0)
    c0                 = network.add_constant((cIn,2,lenH),np.zeros(cIn*2*lenH,dtype=np.float32))
    rnnV2.cell_state   = c0.get_output(0)
        
    gateList = [trt.RNNGateType.INPUT, trt.RNNGateType.CELL, trt.RNNGateType.FORGET, trt.RNNGateType.OUTPUT]
    for layer in range(2):
        wX, wH = np.split([weightF,weightB][layer], [wIn])
        wX = [w.transpose().reshape(-1) for w in np.split(wX, 4, axis=1)]
        wH = [w.transpose().reshape(-1) for w in np.split(wH, 4, axis=1)]
        bX = np.split([biasF,biasB][layer], 4)
        for kind, wx, wh, bx in zip(gateList, wX, wH, bX):
            rnnV2.set_weights_for_gate(layer, kind, True, wx)
            rnnV2.set_bias_for_gate(layer, kind, True, bx)
            rnnV2.set_weights_for_gate(layer, kind, False, wh)
            rnnV2.set_bias_for_gate(layer, kind, False, np.zeros(lenH, dtype=np.float32))
        
    print("rnnV2->", rnnV2.get_output(0).shape, rnnV2.get_output(1).shape, rnnV2.get_output(2).shape)
```
+ 输出张量 0 形状 (3, 4, 10)，3 个独立输出，每个输出 4 个隐藏状态，每个隐藏状态 5 维坐标，2 个方向并排放置
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0.760517 & 0.760517 & 0.760517 & 0.760517 & 0.760517 & 0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322 \\
        0.963941 & 0.963941 & 0.963941 & 0.963941 & 0.963941 & 0.995038 & 0.995038 & 0.995038 & 0.995038 & 0.995038 \\
        0.995038 & 0.995038 & 0.995038 & 0.995038 & 0.995038 & 0.963941 & 0.963941 & 0.963941 & 0.963941 & 0.963941 \\
        0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.760517 & 0.760517 & 0.760517 & 0.760517 & 0.760517
    \end{matrix}\right]\\
    \left[\begin{matrix}
        0.760517 & 0.760517 & 0.760517 & 0.760517 & 0.760517 & 0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322 \\
        0.963941 & 0.963941 & 0.963941 & 0.963941 & 0.963941 & 0.995038 & 0.995038 & 0.995038 & 0.995038 & 0.995038 \\
        0.995038 & 0.995038 & 0.995038 & 0.995038 & 0.995038 & 0.963941 & 0.963941 & 0.963941 & 0.963941 & 0.963941 \\
        0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.760517 & 0.760517 & 0.760517 & 0.760517 & 0.760517
    \end{matrix}\right]\\
    \left[\begin{matrix}
        0.760517 & 0.760517 & 0.760517 & 0.760517 & 0.760517 & 0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322 \\
        0.963941 & 0.963941 & 0.963941 & 0.963941 & 0.963941 & 0.995038 & 0.995038 & 0.995038 & 0.995038 & 0.995038 \\
        0.995038 & 0.995038 & 0.995038 & 0.995038 & 0.995038 & 0.963941 & 0.963941 & 0.963941 & 0.963941 & 0.963941 \\
        0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.760517 & 0.760517 & 0.760517 & 0.760517 & 0.760517
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (3, 2, 5)，3 个独立输出，每个输出 1 个最终隐藏状态，每个隐藏状态 5 维坐标，2 个方向分行放置
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322 \\
        0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322
    \end{matrix}\right]\\
    \left[\begin{matrix}
        0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322 \\
        0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322
    \end{matrix}\right]\\
    \left[\begin{matrix}
        0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322 \\
        0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322
    \end{matrix}\right]
\end{matrix}\right]
$$

+ *rnnV2.get_output(2)* 输出形状 (3, 1, 5)，3 个独立输出，每个输出 1 个最终细胞状态，每个细胞状态 5 维坐标，2 个方向分行放置
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        3.998999 & 3.998999 & 3.998999 & 3.998999 & 3.998999 \\
        3.998999 & 3.998999 & 3.998999 & 3.998999 & 3.998999
    \end{matrix}\right]\\
    \left[\begin{matrix}
        3.998999 & 3.998999 & 3.998999 & 3.998999 & 3.998999 \\
        3.998999 & 3.998999 & 3.998999 & 3.998999 & 3.998999
    \end{matrix}\right]\\
    \left[\begin{matrix}
        3.998999 & 3.998999 & 3.998999 & 3.998999 & 3.998999 \\
        3.998999 & 3.998999 & 3.998999 & 3.998999 & 3.998999
    \end{matrix}\right]
\end{matrix}\right]
$$

--
### weight & bias（add_rnn）
```python
# 调整部分参数
bias    = np.zeros(lenH*2, dtype=np.float32)                                                        # RNN 偏置，bX 和 bH 连接在一起
```
```python
    shuffle = network.add_shuffle(inputTensor)                                                      # 替换部分，要先 shuffle 成 RNNv1 接受的形式
    shuffle.first_transpose = (1,0,2)
    print("shuffle->", shuffle.get_output(0).shape)                                                 # (hIn,cIn,wIn) 等价于 (sequenceLength,batchSize,embeddingLength)
        
    fakeWeight  = np.random.rand(wIn+lenH,lenH).astype(np.float32)
    fakeBias    = np.random.rand(lenH*2).astype(np.float32)
    rnn = network.add_rnn(shuffle.get_output(0), 1, lenH, hIn, trt.RNNOperation.RELU,\
                          trt.RNNInputMode.LINEAR, trt.RNNDirection.UNIDIRECTION, fakeWeight, fakeBias) # 先填入一些参数，后续再修改
    rnn.weights = weight.transpose()                                                                # RNN 变换阵，可覆盖函数 add_rnn 的参数
    rnn.bias    = bias                                                                              # RNN 偏置，可覆盖函数 add_rnn 的参数
    print("rnn->", rnn.get_output(0).shape, rnn.get_output(1).shape)
```

+ 输出张量 0 形状 (4, 3, 5)，3 个独立输出（H 维），每个输出 4 个隐藏状态（C 维），每个隐藏状态 5 维坐标（W 维），计算与初始代码相同，只是结果放置方式不同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
           7. &    7. &    7. &    7. &    7. \\
           7. &    7. &    7. &    7. &    7. \\
           7. &    7. &    7. &    7. &    7.
    \end{matrix}\right]
    \left[\begin{matrix}
          42. &   42. &   42. &   42. &   42. \\
          42. &   42. &   42. &   42. &   42. \\
          42. &   42. &   42. &   42. &   42.
    \end{matrix}\right]
    \left[\begin{matrix}
         217. &  217. &  217. &  217. &  217. \\
         217. &  217. &  217. &  217. &  217. \\
         217. &  217. &  217. &  217. &  217.
    \end{matrix}\right]
    \left[\begin{matrix}
        1092. & 1092. & 1092. & 1092. & 1092. \\
        1092. & 1092. & 1092. & 1092. & 1092. \\
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (1, 3, 5)，3 个独立输出，每个输出 1 个最终隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1092. & 1092. & 1092. & 1092. & 1092. \\
        1092. & 1092. & 1092. & 1092. & 1092. \\
        1092. & 1092. & 1092. & 1092. & 1092.
    \end{matrix}\right]
\end{matrix}\right]
$$

<div style="page-break-after:always;"></div>
## Scale 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 3                                                                                         # 输入张量 HWC
wIn     = 3
cIn     = 3
data    = np.arange(1,1+cIn*wIn*wIn,dtype=np.float32).reshape(cIn,hIn,wIn)                          # 输入张量

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替换部分
    scale = np.array([0.5],dtype=np.float32)
    shift = np.array([-7.0],dtype=np.float32)
    power = np.array([1.0],dtype=np.float32)
    sc = network.add_scale(inputTensor, trt.ScaleMode.UNIFORM, shift,scale,power)
    print("sc->", sc.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(sc.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (3, 3, 3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1. & 2. & 3. \\
        4. & 5. & 6. \\
        7. & 8. & 9.
    \end{matrix}\right]
    \left[\begin{matrix}
        10. & 11. & 12. \\
        13. & 14. & 15. \\
        16. & 17. & 18.
    \end{matrix}\right]
    \left[\begin{matrix}
        19. & 20. & 21. \\
        22. & 23. & 24. \\
        25. & 26. & 27.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (3, 3, 3)，所有元素都做了变换 $ y = \left( x \cdot scale + shift\right) ^{power} $
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        -6.5 & -6.  & -5.5 \\
        -5.  & -4.5 & -4.  \\
        -3.5 & -3.  & -2.5
    \end{matrix}\right]
    \left[\begin{matrix}
        -2.  & -1.5 & -1.  \\
        -0.5 &  0.  &  0.5 \\
         1.  &  1.5 &  2.
    \end{matrix}\right]
    \left[\begin{matrix}
        2.5  & 3.  & 3.5 \\
        4.   & 4.5 & 5.  \\
        5.5  & 6.  & 6.5
    \end{matrix}\right]
\end{matrix}\right]
$$

+ TensorRT7 之前，需要对参数 scale，shift，power 的 np 数组作拷贝，防止其被后续同名变量覆盖，因为 TensorRT 中这些参数的定义和使用是异步的。TensorRT7 之后该问题被修正，不再需要额外的拷贝工作，如前述样例代码那样写就行
```python
    bag = []
    scale = np.array([0.5],dtype=np.float32)
    shift = np.array([-7.0],dtype=np.float32)
    power = np.array([1.0],dtype=np.float32)
    bag  += [scale,shift,power]
    sc = network.add_scale(...)
```

---
### mode & scale & shift & power
```python
    one = np.array([1],dtype=np.float32)                                                            # 替换部分
    sc = network.add_scale(inputTensor, trt.ScaleMode.UNIFORM, one, one, one)
    sc.mode  = trt.ScaleMode.UNIFORM                                                                # 模式，可覆盖函数 add_scale 的参数
    sc.shift = np.array([-7.0],dtype=np.float32)                                                    # 加法参数，可覆盖函数 add_scale 的参数
    sc.scale = np.array([0.5],dtype=np.float32)                                                     # 乘法参数，可覆盖函数 add_scale 的参数
    sc.power = np.array([1.0],dtype=np.float32)                                                     # 指数参数，可覆盖函数 add_scale 的参数
    print("sc->", sc.get_output(0).shape)                                                           # (cIn, hIn, wIn)
```

+ 输出张量 (3, 3, 3)，与初始代码相同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        -6.5 & -6.  & -5.5 \\
        -5.  & -4.5 & -4.  \\
        -3.5 & -3.  & -2.5
    \end{matrix}\right]
    \left[\begin{matrix}
        -2.  & -1.5 & -1.  \\
        -0.5 &  0.  &  0.5 \\
         1.  &  1.5 &  2.
    \end{matrix}\right]
    \left[\begin{matrix}
        2.5  & 3.  & 3.5 \\
        4.   & 4.5 & 5.  \\
        5.5  & 6.  & 6.5
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 注意，可用的模式
| trt.ScaleMode 名 | 说明                        |
| :--------------- | :-------------------------- |
| ELEMENTWISE      | 每个元素使用一套参数        |
| UNIFORM          | 所有元素使用一套参数        |
| CHANNEL          | 同个 C 维的元素使用一套参数 |

---
### CHANNEL 和 ELEMENTWISE 级的 scale
```python
    shift = np.array([-2.5,-7.0,-11.5],dtype=np.float32)                                            # 替换部分，参数元素数等于通道数
    scale = np.full(3,0.5,dtype=np.float32)
    power = np.ones(3,dtype=np.float32)
    sc = network.add_scale(inputTensor, trt.ScaleMode.CHANNEL, shift, scale, power)
    print("sc->", sc.get_output(0).shape)
```

+ 输出张量 (3, 3, 3)，每个通道分别用自己的参数进行 scale
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        -2.  & -1.5 & -1.  \\
        -0.5 &  0.  &  0.5 \\
         1.  &  1.5 &  2.
    \end{matrix}\right]
    \left[\begin{matrix}
        -2.  & -1.5 & -1.  \\
        -0.5 &  0.  &  0.5 \\
         1.  &  1.5 &  2.
    \end{matrix}\right]
    \left[\begin{matrix}
        -2.  & -1.5 & -1.  \\
        -0.5 &  0.  &  0.5 \\
         1.  &  1.5 &  2.
    \end{matrix}\right]
\end{matrix}\right]
$$
```python
    shift = np.full([cIn,hIn,wIn],-7.0,dtype=np.float32)                                            # 替换部分，参数元素数等于输入张量元素数
    scale = np.full([cIn,hIn,wIn],0.5,dtype=np.float32)
    power = np.ones([cIn,hIn,wIn],dtype=np.float32)
    sc = network.add_scale(inputTensor, trt.ScaleMode.ELEMENTWISE, shift, scale, power)
    print("sc->", sc.get_output(0).shape)
```

+ 输出张量 (3, 3, 3)，与初始代码相同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        -6.5 & -6.  & -5.5 \\
        -5.  & -4.5 & -4.  \\
        -3.5 & -3.  & -2.5
    \end{matrix}\right]
    \left[\begin{matrix}
        -2.  & -1.5 & -1.  \\
        -0.5 &  0.  &  0.5 \\
         1.  &  1.5 &  2.
    \end{matrix}\right]
    \left[\begin{matrix}
        2.5  & 3.  & 3.5 \\
        4.   & 4.5 & 5.  \\
        5.5  & 6.  & 6.5
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### channel_axis（add_scale_nd）
```python
# 调整部分参数
pIn     = 3
data    = np.arange(1,1+pIn*cIn*wIn*wIn,dtype=np.float32).reshape(pIn,cIn,hIn,wIn)
#...
```
```python
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (pIn, cIn, hIn, wIn))        # 输入张量调整
    #...
    scale = np.array([0.5,0.5,0.5],dtype=np.float32)                                                # 替换部分
    shift = np.array([-7.0,-20.5,-34.0],dtype=np.float32)
    power = np.array([1.0,1.0,1.0],dtype=np.float32)
    sc = network.add_scale_nd(inputTensor, trt.ScaleMode.CHANNEL, shift, scale, power, 0)           # 必须在函数 add_scale_nd 中指定 Channel 维是哪根轴
    print("sc->", sc.get_output(0).shape, sc.channel_axis)                                          # channel_axis 仅供输出，不能调节
```

+ 输入张量 (3, 3, 3, 3)
$$
\left[\begin{matrix}
\left[\begin{matrix}
    \left[\begin{matrix}
         1. &  2. &  3. \\
         4. &  5. &  6. \\
         7. &  8. &  9.
    \end{matrix}\right]
    \left[\begin{matrix}
        10. & 11. & 12. \\
        13. & 14. & 15. \\
        16. & 17. & 18.
    \end{matrix}\right]
    \left[\begin{matrix}
        19. & 20. & 21. \\
        22. & 23. & 24. \\
        25. & 26. & 27.
    \end{matrix}\right]
    \end{matrix}\right]\\
    \left[\begin{matrix}
    \left[\begin{matrix}
        28. & 29. & 30. \\
        31. & 32. & 33. \\
        34. & 35. & 36.
    \end{matrix}\right]
    \left[\begin{matrix}
        37. & 38. & 39. \\
        40. & 41. & 42. \\
        43. & 44. & 45.
    \end{matrix}\right]
    \left[\begin{matrix}
        46. & 47. & 48. \\
        49. & 50. & 51. \\
        52. & 53. & 54.
    \end{matrix}\right]
    \end{matrix}\right]\\
    \left[\begin{matrix}
    \left[\begin{matrix}
        55. & 56. & 57. \\
        58. & 59. & 60. \\
        61. & 62. & 63.
    \end{matrix}\right]
    \left[\begin{matrix}
        64. & 65. & 66. \\
        67. & 68. & 69. \\
        70. & 71. & 72.
    \end{matrix}\right]
    \left[\begin{matrix}
        73. & 74. & 75. \\
        76. & 77. & 78. \\
        79. & 80. & 81.
    \end{matrix}\right]
\end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (3, 3, 3, 3)，其中指定 channel_axis = 0，在 P 维上进行 scale，图中每行三个通道共用一套参数
$$
\left[\begin{matrix}
\left[\begin{matrix}
    \left[\begin{matrix}
         -6.5 &  -6.  &  -5.5 \\
         -5.  &  -4.5 &  -4.  \\
         -3.5 &  -3.  &  -2.5
    \end{matrix}\right]
    \left[\begin{matrix}
         -2.  &  -1.5 &  -1.  \\
         -0.5 &   0.  &   0.5 \\
          1.0 &   1.5 &   2.
    \end{matrix}\right]
    \left[\begin{matrix}
          2.5 &   3.  &   3.5 \\
          4.  &   4.5 &   5.  \\
          5.5 &   6.  &   6.5 \\
    \end{matrix}\right]
    \end{matrix}\right]\\
    \left[\begin{matrix}
    \left[\begin{matrix}
         -6.5 &  -6.  &  -5.5 \\
         -5.  &  -4.5 &  -4.  \\
         -3.5 &  -3.  &  -2.5
    \end{matrix}\right]
    \left[\begin{matrix}
         -2.  &  -1.5 &  -1.  \\
         -0.5 &   0.  &   0.5 \\
          1.0 &   1.5 &   2.
    \end{matrix}\right]
    \left[\begin{matrix}
          2.5 &   3.  &   3.5 \\
          4.  &   4.5 &   5.  \\
          5.5 &   6.  &   6.5 \\
    \end{matrix}\right]
    \end{matrix}\right]\\
    \left[\begin{matrix}
    \left[\begin{matrix}
         -6.5 &  -6.  &  -5.5 \\
         -5.  &  -4.5 &  -4.  \\
         -3.5 &  -3.  &  -2.5
    \end{matrix}\right]
    \left[\begin{matrix}
         -2.  &  -1.5 &  -1.  \\
         -0.5 &   0.  &   0.5 \\
          1.0 &   1.5 &   2.
    \end{matrix}\right]
    \left[\begin{matrix}
          2.5 &   3.  &   3.5 \\
          4.  &   4.5 &   5.  \\
          5.5 &   6.  &   6.5 \\
    \end{matrix}\right]
\end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (3, 3, 3, 3)，其中指定 channel_axis = 1，在 C 维上进行 scale，图中每列三个 HW 矩阵共用一套参数
$$
\left[\begin{matrix}
\left[\begin{matrix}
    \left[\begin{matrix}
         -6.5 &  -6.  &  -5.5 \\
         -5.  &  -4.5 &  -4.  \\
         -3.5 &  -3.  &  -2.5
    \end{matrix}\right]
    \left[\begin{matrix}
        -15.5 & -15. &  -14.5 \\
        -14.  & -13.5 & -13.  \\
        -12.5 & -12. &  -11.5
    \end{matrix}\right]
    \left[\begin{matrix}
        -24.5 & -24. &  -23.5 \\
        -23.  & -22.5 & -22.  \\
        -21.5 & -21. &  -20.5
    \end{matrix}\right]
    \end{matrix}\right]\\
    \left[\begin{matrix}
    \left[\begin{matrix}
          7.  &   7.5 &   8.  \\
          8.5 &   9. &    9.5 \\
         10.  &  10.5 &  11. 
    \end{matrix}\right]
    \left[\begin{matrix}
         -2.  &  -1.5 &  -1.  \\
         -0.5 &   0. &    0.5 \\
          1.  &   1.5 &   2.
    \end{matrix}\right]
    \left[\begin{matrix}
        -11.  & -10.5 & -10.  \\
         -9.5 &  -9. &   -8.5 \\
         -8.  &  -7.5 &  -7.
    \end{matrix}\right]
    \end{matrix}\right]\\
    \left[\begin{matrix}
    \left[\begin{matrix}
         20.5 &  21. &   21.5 \\
         22.  &  22.5 &  23.  \\
         23.5 &  24. &   24.5
    \end{matrix}\right]
    \left[\begin{matrix}
         11.5 &  12. &   12.5 \\
         13.  &  13.5 &  14.  \\
         14.5 &  15. &   15.5
    \end{matrix}\right]
    \left[\begin{matrix}
          2.5 &   3. &    3.5 \\
          4.  &   4.5 &   5.  \\
          5.5 &   6. &    6.5
    \end{matrix}\right]
\end{matrix}\right]
\end{matrix}\right]
$$

<div style="page-break-after:always;"></div>
## Select 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 4                                                                                         # 输入张量 HWC
wIn     = 5
cIn     = 3
data1   = np.arange(cIn*hIn*wIn,dtype=np.float32).reshape(cIn,hIn,wIn)                              # 输入张量
data2   = -data1

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network(1<<0)                                                          # 使用显式 batch 维，创建网络时传入 trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH 
    inputTensor1 = network.add_input('inputTensor1', trt.DataType.FLOAT, (1, cIn, hIn, wIn))
    inputTensor2 = network.add_input('inputTensor2', trt.DataType.FLOAT, (1, cIn, hIn, wIn))
    print("inputTensor->", inputTensor1.shape)

    #-----------------------------------------------------------------------------------------------# 可替换部分
    cond_ = network.add_constant((1,cIn,hIn,wIn), (np.arange(cIn*hIn*wIn)%2).astype(np.int32))      # 条件张量，要改装 BOOL 型
    
    cond = network.add_identity(cond_.get_output(0))
    cond.set_output_type(0,trt.DataType.BOOL)

    select = network.add_select(cond.get_output(0), inputTensor1, inputTensor2)
    print("select->", select.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(select.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data1.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    in2_h   = np.ascontiguousarray(data2.reshape(-1))
    in2_d   = cuda.mem_alloc(in2_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    cuda.memcpy_htod_async(in2_d, in2_h, stream)
    context.execute_async(1, [int(in1_d), int(in2_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data1:", data1.shape)
    print(data1)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (3, 4, 5)，第一个张量元素全正，第二个张量的所有元素为第一个的相反数
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         0. &  1. &  2. &  3. &  4. \\
         5. &  6. &  7. &  8. &  9. \\
        10. & 11. & 12. & 13. & 14. \\
        15. & 16. & 17. & 18. & 19.
    \end{matrix}\right]
    \left[\begin{matrix}
        20. & 21. & 22. & 23. & 24. \\
        25. & 26. & 27. & 28. & 29. \\
        30. & 31. & 32. & 33. & 34. \\
        35. & 36. & 37. & 38. & 39.
    \end{matrix}\right]
    \left[\begin{matrix}
        40. & 41. & 42. & 43. & 44. \\
        45. & 46. & 47. & 48. & 49. \\
        50. & 51. & 52. & 53. & 54. \\
        55. & 56. & 57. & 58. & 59.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 3, 4, 5)，交替输出两个输入张量的值
$$
\left[\begin{matrix}
\left[\begin{matrix}
    \left[\begin{matrix}
         -0. &   1. &  -2. &   3. &  -4. \\
          5. &  -6. &   7. &  -8. &   9. \\
        -10. &  11. & -12. &  13. & -14. \\
         15. & -16. &  17. & -18. &  19.
    \end{matrix}\right]
    \left[\begin{matrix}
        -20. &  21. & -22. &  23. & -24. \\
         25. & -26. &  27. & -28. &  29. \\
        -30. &  31. & -32. &  33. & -34. \\
         35. & -36. &  37. & -38. &  39.
    \end{matrix}\right]
    \left[\begin{matrix}
        -40. &  41. & -42. &  43. & -44. \\
         45. & -46. &  47. & -48. &  49. \\
        -50. &  51. & -52. &  53. & -54. \\
         55. & -56. &  57. & -58. &  59.
    \end{matrix}\right]
\end{matrix}\right]
\end{matrix}\right]
$$

<div style="page-break-after:always;"></div>
## Shape 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 4                                                                                         # 输入张量 HWC
wIn     = 5
cIn     = 3
data    = np.ones([cIn*hIn*wIn],dtype=np.float32).reshape(cIn,hIn,wIn)

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))      # 使用显式 batch 模式
    #-----------------------------------------------------------------------------------------------# 可替换部分
    dataLayer = network.add_constant([1,cIn,hIn,wIn],data)
    
    shape = network.add_shape(dataLayer.get_output(0))
    print("shape->", shape.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(shape.get_output(0))                                                        # 常规方法获取形状张量
    res = network.mark_output_for_shapes(shape.get_output(0))                                       # “标记形状张量作为输出”的专用方法
    print("mark_output_for_shapes succeed?", res)
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    out1_h  = np.empty(engine.get_binding_shape(0),dtype = trt.nptype(engine.get_binding_dtype(0)))
    out1_d       = cuda.mem_alloc(out1_h.nbytes)

    context.execute_async(1, [int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
        
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    print("output shape from context: ", context.get_shape(0))                                      # 专用方法从 context 中获取形状张量的值，注意本网络没有输入张量
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (3, 4, 5)
$$
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
$$

+ 输出张量 (4,)，显式包含 batch 维。可以从 context 中等价的获取，output shape from context:  [1, 3, 4, 5]
$$
\left[\begin{matrix}
  1 & 3 & 4 & 5 \\
\end{matrix}\right]
$$

+ 注意，TensorRT 7 中两种方法均可使用，TensorRT 6 中只能使用专用方法，若在 TensorRT 6 中使用常规方法，则会得到如下报错：

> [TensorRT] ERROR: (Unnamed Layer* 1) [Shape]: ShapeLayer output tensor ((Unnamed Layer* 1) [Shape]_output) used as an execution tensor,  but must be used only as shape tensor.
build engine failed.

<div style="page-break-after:always;"></div>
## Shuffle 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 4                                                                                         # 输入张量 HWC
wIn     = 5
cIn     = 3
data    = np.arange(cIn,dtype=np.float32).reshape(cIn,1,1)*100 + np.arange(hIn).reshape(1,hIn,1)*10 + np.arange(wIn).reshape(1,1,wIn)# 输入张量
data    = data.astype(np.float32)

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替换部分
    sh = network.add_shuffle(inputTensor)
    print("sh->", sh.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(sh.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (3, 4, 5)，百位表示 C 维编号，十位表示 H 维编号，个位表示 W 维编号
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         0. &  1. &  2. &  3. &  4. \\
        10. & 11. & 12. & 13. & 14. \\
        20. & 21. & 22. & 23. & 24. \\
        30. & 31. & 32. & 33. & 34.
    \end{matrix}\right]
    \left[\begin{matrix}
        100. & 101. & 102. & 103. & 104. \\
        110. & 111. & 112. & 113. & 114. \\
        120. & 121. & 122. & 123. & 124. \\
        130. & 131. & 132. & 133. & 134.
    \end{matrix}\right]
    \left[\begin{matrix}
        200. & 201. & 202. & 203. & 204. \\
        210. & 211. & 212. & 213. & 214. \\
        220. & 221. & 222. & 223. & 224. \\
        230. & 231. & 232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (3, 4, 5)，无变化
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         0. &  1. &  2. &  3. &  4. \\
        10. & 11. & 12. & 13. & 14. \\
        20. & 21. & 22. & 23. & 24. \\
        30. & 31. & 32. & 33. & 34.
    \end{matrix}\right]
    \left[\begin{matrix}
        100. & 101. & 102. & 103. & 104. \\
        110. & 111. & 112. & 113. & 114. \\
        120. & 121. & 122. & 123. & 124. \\
        130. & 131. & 132. & 133. & 134.
    \end{matrix}\right]
    \left[\begin{matrix}
        200. & 201. & 202. & 203. & 204. \\
        210. & 211. & 212. & 213. & 214. \\
        220. & 221. & 222. & 223. & 224. \\
        230. & 231. & 232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### first_transpose
```python
    sh = network.add_shuffle(inputTensor)                                                           # 替换
    sh.first_transpose = (1, 0, 2)                                                                  # 首次转置，默认值 (0,1,2,...)
    print("sh->", sh.get_output(0).shape)
```

+ 输出张量 (4, 3, 5)，其中指定 first_transpose=(1, 0, 2)，相当于将第 0、1、2 维（原始顺序）分别放到第 1、0、2 维（指定顺序）的位置
$$
\left[\begin{matrix}
    \left[\begin{matrix}
          0. &   1. &   2. &   3. &   4. \\
        100. & 101. & 102. & 103. & 104. \\
        200. & 201. & 202. & 203. & 204.
    \end{matrix}\right]
    \left[\begin{matrix}
         10. &  11. &  12. &  13. &  14. \\
        110. & 111. & 112. & 113. & 114. \\
        210. & 211. & 212. & 213. & 214.
    \end{matrix}\right]
    \left[\begin{matrix}
         20. &  21. &  22. &  23. &  24. \\
        120. & 121. & 122. & 123. & 124. \\
        220. & 221. & 222. & 223. & 224.
    \end{matrix}\right]
    \left[\begin{matrix}
         30. &  31. &  32. &  33. &  34. \\
        130. & 131. & 132. & 133. & 134. \\
        230. & 231. & 232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### reshape_dims
```python
    sh = network.add_shuffle(inputTensor)                                                           # 替换部分
    sh.reshape_dims = (-1, 2, 15)                                                                   # 指定新形状，至多有一个元素为 -1，表示自动计算，默认值为 (cIn, hIn, wIn)
    print("sh->", sh.get_output(0).shape)
```

+ 输出张量 (2, 2, 15)，其中指定 reshape_dims=(-1, 2, 15)，保持原来元素顺序的条件下调整张量形状。可以使用至多一个 -1 表示自动计算
$$
\left[\begin{matrix}
    \left[\begin{matrix}
          0. & 1. & 2. & 3. & 4. & 10. & 11. & 12. & 13. & 14. & 20. & 21. & 22. & 23. & 24. \\
         30. & 31. & 32. & 33. & 34. & 100. & 101. & 102. & 103. & 104. & 110. & 111. & 112. & 113. & 114.
    \end{matrix}\right]\\
    \left[\begin{matrix}
        120. & 121. & 122. & 123. & 124. & 130. & 131. & 132. & 133. & 134. & 200. & 201. & 202. & 203. & 204. \\
        210. & 211. & 212. & 213. & 214. & 220. & 221. & 222. & 223. & 224. & 230. & 231. & 232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (3, 4, 5)，其中指定 reshape_dims=(0, 0, 0)，表示从 inputTensor 形状的相应位置上拷贝具体数值
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         0. &  1. &  2. &  3. &  4. \\
        10. & 11. & 12. & 13. & 14. \\
        20. & 21. & 22. & 23. & 24. \\
        30. & 31. & 32. & 33. & 34.
    \end{matrix}\right]
    \left[\begin{matrix}
        100. & 101. & 102. & 103. & 104. \\
        110. & 111. & 112. & 113. & 114. \\
        120. & 121. & 122. & 123. & 124. \\
        130. & 131. & 132. & 133. & 134.
    \end{matrix}\right]
    \left[\begin{matrix}
        200. & 201. & 202. & 203. & 204. \\
        210. & 211. & 212. & 213. & 214. \\
        220. & 221. & 222. & 223. & 224. \\
        230. & 231. & 232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### second_transpose
```python
    sh = network.add_shuffle(inputTensor)                                                           # 替换部分
    sh.second_transpose = (1, 0, 2)                                                                 # 末次转置，与首次转置作用相同，默认值 (0,1,2,...)
    print("sh->", sh.get_output(0).shape)
```

+ 输出张量 (4, 3, 5)，其中指定 second_transpose=(1, 0, 2)，与首次转置作用相同，但是发生在调整形状之后
$$
\left[\begin{matrix}
    \left[\begin{matrix}
          0. &   1. &   2. &   3. &   4. \\
        100. & 101. & 102. & 103. & 104. \\
        200. & 201. & 202. & 203. & 204.
    \end{matrix}\right]
    \left[\begin{matrix}
         10. &  11. &  12. &  13. &  14. \\
        110. & 111. & 112. & 113. & 114. \\
        210. & 211. & 212. & 213. & 214.
    \end{matrix}\right]
    \left[\begin{matrix}
         20. &  21. &  22. &  23. &  24. \\
        120. & 121. & 122. & 123. & 124. \\
        220. & 221. & 222. & 223. & 224.
    \end{matrix}\right]
    \left[\begin{matrix}
         30. &  31. &  32. &  33. &  34. \\
        130. & 131. & 132. & 133. & 134. \\
        230. & 231. & 232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### 组合使用的例子
```python
    sh = network.add_shuffle(inputTensor)                                                           # 替换部分
    sh.first_transpose  = (1, 0, 2)
    sh.reshape_dims     = (4, 5, 3)
    sh.second_transpose = (1, 0, 2)
    print("sh->", sh.get_output(0).shape)
```

+ 输出张量 (5, 4, 3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
          0. &   1. &   2. \\
         10. &  11. &  12. \\
         20. &  21. &  22. \\
         30. &  31. &  32.
    \end{matrix}\right]
    \left[\begin{matrix}
          3. &   4. & 100. \\
         13. &  14. & 110. \\
         23. &  24. & 120. \\
         33. &  34. & 130.
    \end{matrix}\right]
    \left[\begin{matrix}
        101. & 102. & 103. \\
        111. & 112. & 113. \\
        121. & 122. & 123. \\
        131. & 132. & 133.
    \end{matrix}\right]
    \left[\begin{matrix}
        104. & 200. & 201. \\
        114. & 210. & 211. \\
        124. & 220. & 221. \\
        134. & 230. & 231.
    \end{matrix}\right]
    \left[\begin{matrix}
        202. & 203. & 204. \\
        212. & 213. & 214. \\
        222. & 223. & 224. \\
        232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：
$$
\left[\begin{matrix}
    \left[\begin{matrix}
          0. &  1. &  2. &  3. &  4. \\
         10. & 11. & 12. & 13. & 14. \\
         20. & 21. & 22. & 23. & 24. \\
         30. & 31. & 32. & 33. & 34.
    \end{matrix}\right]
    \left[\begin{matrix}
        100. & 101. & 102. & 103. & 104. \\
        110. & 111. & 112. & 113. & 114. \\
        120. & 121. & 122. & 123. & 124. \\
        130. & 131. & 132. & 133. & 134.
    \end{matrix}\right]
    \left[\begin{matrix}
        200. & 201. & 202. & 203. & 204. \\
        210. & 211. & 212. & 213. & 214. \\
        220. & 221. & 222. & 223. & 224. \\
        230. & 231. & 232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right] \rightarrow \left( 3,4,5 \right) \\
\Downarrow \ first\_transpose \left( 1,0,2 \right) \\
\left[\begin{matrix}
    \left[\begin{matrix}
          0. &   1. &   2. &   3. &   4. \\
        100. & 101. & 102. & 103. & 104. \\
        200. & 201. & 202. & 203. & 204.
    \end{matrix}\right]
    \left[\begin{matrix}
         10. &  11. &  12. &  13. &  14. \\
        110. & 111. & 112. & 113. & 114. \\
        210. & 211. & 212. & 213. & 214.
    \end{matrix}\right]
    \left[\begin{matrix}
         20. &  21. &  22. &  23. &  24. \\
        120. & 121. & 122. & 123. & 124. \\
        220. & 221. & 222. & 223. & 224.
    \end{matrix}\right]
    \left[\begin{matrix}
         30. &  31. &  32. &  33. &  34. \\
        130. & 131. & 132. & 133. & 134. \\
        230. & 231. & 232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right] \rightarrow \left( 4,3,5 \right) \\
\Downarrow \ reshape \left( 4,5,3 \right) \\
\left[\begin{matrix}
    \left[\begin{matrix}
          0. &   1. &   2. \\
          3. &   4. & 100. \\
        101. & 102. & 103. \\
        104. & 200. & 201. \\
        202. & 203. & 204.
    \end{matrix}\right]
    \left[\begin{matrix}
         10. &  11. &  12. \\
         13. &  14. & 110. \\
        111. & 112. & 113. \\
        114. & 210. & 211. \\
        212. & 213. & 214.
    \end{matrix}\right]
    \left[\begin{matrix}
         20. &  21. &  22. \\
         23. &  24. & 120. \\
        121. & 122. & 123. \\
        124. & 220. & 221. \\
        222. & 223. & 224.
    \end{matrix}\right]
    \left[\begin{matrix}
         30. &  31. &  32. \\
         33. &  34. & 130. \\
        131. & 132. & 133. \\
        134. & 230. & 231. \\
        232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right] \rightarrow \left( 4,5,3 \right) \\
\Downarrow \ second\_transpose \left( 1,0,2 \right) \\
\left[\begin{matrix}
    \left[\begin{matrix}
          0. &   1. &   2. \\
         10. &  11. &  12. \\
         20. &  21. &  22. \\
         30. &  31. &  32.
    \end{matrix}\right]
    \left[\begin{matrix}
          3. &   4. & 100. \\
         13. &  14. & 110. \\
         23. &  24. & 120. \\
         33. &  34. & 130.
    \end{matrix}\right]
    \left[\begin{matrix}
        101. & 102. & 103. \\
        111. & 112. & 113. \\
        121. & 122. & 123. \\
        131. & 132. & 133.
    \end{matrix}\right]
    \left[\begin{matrix}
        104. & 200. & 201. \\
        114. & 210. & 211. \\
        124. & 220. & 221. \\
        134. & 230. & 231.
    \end{matrix}\right]
    \left[\begin{matrix}
        202. & 203. & 204. \\
        212. & 213. & 214. \\
        222. & 223. & 224. \\
        232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right] \rightarrow \left( 5,4,3 \right) \\
$$

---
### zero_is_place_holder
```python
# 调整部分参数
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))      # 使用显式 batch 模式
```
```python
    sh = network.add_shuffle(inputTensor)                                                           # 替换部分
    sh.reshape_dims = (0,0,0)
    sh.zero_is_place_holder = True                                                                  # True，效果跟隐式 Batch 模式相同
    print("sh->", sh.get_output(0).shape)
```

+ 输出张量 (3, 4, 5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         0. &  1. &  2. &  3. &  4. \\
        10. & 11. & 12. & 13. & 14. \\
        20. & 21. & 22. & 23. & 24. \\
        30. & 31. & 32. & 33. & 34.
    \end{matrix}\right]
    \left[\begin{matrix}
        100. & 101. & 102. & 103. & 104. \\
        110. & 111. & 112. & 113. & 114. \\
        120. & 121. & 122. & 123. & 124. \\
        130. & 131. & 132. & 133. & 134.
    \end{matrix}\right]
    \left[\begin{matrix}
        200. & 201. & 202. & 203. & 204. \\
        210. & 211. & 212. & 213. & 214. \\
        220. & 221. & 222. & 223. & 224. \\
        230. & 231. & 232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right]
$$

```python
# 调整部分参数
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))      # 使用显式 batch 模式
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (0, cIn, hIn, wIn))          # 输入张量总尺寸为 0
```
```python
    sh = network.add_shuffle(inputTensor)                                                           # 替换部分
    sh.reshape_dims = (0,0,0)
    sh.zero_is_place_holder = False
    print("sh->", sh.get_output(0).shape)
```

+ 输出张量 (0, 0, 0)，这种用法常用于本层输出张量广播后再用于其他层的情况

---
### 静态 set_input
```python
# 调整部分参数
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))      # 使用显式 batch 模式
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (1, cIn, hIn, wIn))
```
```python
    sh = network.add_shuffle(inputTensor)                                                           # 替换部分
    tensor1 = network.add_constant((4,), np.array([1,4,5,3],dtype=np.int32))
    #sh.set_input(0,inputTensor)                                                                    # 第 0 个张量就是被 shuffle 的张量
    sh.set_input(1,tensor1.get_output(0))                                                           # 第 1 个张量是 reshape 的形状
    print("sh->", sh.get_output(0).shape)
```

+ 输出张量 (1, 4, 5, 3)
$$
\left[\begin{matrix}
\left[\begin{matrix}
    \left[\begin{matrix}
          0. &   1. &   2. \\
          3. &   4. &  10. \\
         11. &  12. &  13. \\
         14. &  20. &  21. \\
         22. &  23. &  24.
    \end{matrix}\right]
    \left[\begin{matrix}
         30. &  31. &  32. \\
         33. &  34. & 100. \\
        101. & 102. & 103. \\
        104. & 110. & 111. \\
        112. & 113. & 114.
    \end{matrix}\right]
    \left[\begin{matrix}
        120. & 121. & 122. \\
        123. & 124. & 130. \\
        131. & 132. & 133. \\
        134. & 200. & 201. \\
        202. & 203. & 204.
    \end{matrix}\right]
    \left[\begin{matrix}
        210. & 211. & 212. \\
        213. & 214. & 220. \\
        221. & 222. & 223. \\
        224. & 230. & 231. \\
        232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right]
\end{matrix}\right]
$$

---
### 动态 set_input
```python
# buildEngine() 和 run() 大改：
def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))      # 使用显式 batch 模式
    tensor1 = network.add_input('dataTensor', trt.DataType.FLOAT, (1, cIn, hIn, wIn))
    tensor2 = network.add_input('shapeTensor', trt.DataType.INT32, (4,))                            # reshape 形状张量变为运行时输入
    print(tensor1.shape,tensor2.shape)

    #-----------------------------------------------------------------------------------------------#
    sh = network.add_shuffle(tensor1)
    sh.set_input(1,tensor2)
    print("sh->", sh.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------#

    network.mark_output(sh.get_output(0))
    
    profile = builder.create_optimization_profile()                                                 # 需要 profile
    profile.set_shape_input(tensor2.name, [1,1,1,1],[1,5,1,12],[1,5,1,12])

    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    return builder.build_engine(network, config)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    context.set_shape_input(1, [1,5,1,12])                                                          # 运行时指定形状张量的值
    print("context->", context.all_shape_inputs_specified,context.all_binding_shapes_specified)     # 确认形状张量已经输入
    stream  = cuda.Stream()
    
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    in2_h   = np.ascontiguousarray(np.array([1,1,1,1],dtype=np.int32).reshape(-1))                  # 输入形状张量可初始化为垃圾数据
    in2_d   = cuda.mem_alloc(in2_h.nbytes)
    out1_h  = np.empty(context.get_binding_shape(2), dtype = trt.nptype(engine.get_binding_dtype(2)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    cuda.memcpy_htod_async(in2_d, in2_h, stream)
    
    context.execute_async(1, [int(in1_d), int(in2_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
```

+ 输出张量 (1, 5, 1, 12)
$$
\left[\begin{matrix}
\left[\begin{matrix}
    \left[\begin{matrix}
          0. &   1. &   2. &   3. &   4. &  10. &  11. &  12. &  13. &  14. &  20. &  21. \\
    \end{matrix}\right]\\
    \left[\begin{matrix}
         22. &  23. &  24. &  30. &  31. &  32. &  33. &  34. & 100. & 101. & 102. & 103. \\
    \end{matrix}\right]\\
    \left[\begin{matrix}
        104. & 110. & 111. & 112. & 113. & 114. & 120. & 121. & 122. & 123. & 124. & 130. \\
    \end{matrix}\right]\\
    \left[\begin{matrix}
        131. & 132. & 133. & 134. & 200. & 201. & 202. & 203. & 204. & 210. & 211. & 212. \\
    \end{matrix}\right]\\
    \left[\begin{matrix}
        213. & 214. & 220. & 221. & 222. & 223. & 224. & 230. & 231. & 232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right]
\end{matrix}\right]
$$

+ 建立网络时需要 profile，否则报错 "shapeTensor: shape values missing for shape input" 和 "Network validation failed"


### 在 dynamic shape 模式中进行 reshape
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 8
    builder.max_workspace_size = 3 << 30
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))      # 使用显式 batch 模式
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (-1, -1, -1, -1))        

    orginalShape = network.add_shape(inputTensor)
    one  = network.add_constant([1,],np.array([1],dtype=np.int32))    
    newShape = network.add_concatenation([orginalShape.get_output(0),one.get_output(0)])
    newShape.axis = 0
    
    shuffleLayer = network.add_shuffle(inputTensor)
    shuffleLayer.set_input(1,newShape.get_output(0))
    
    restoreLayer = network.add_shuffle(shuffleLayer.get_output(0))
    restoreLayer.set_input(1,orginalShape.get_output(0))
        
    network.mark_output(shuffleLayer.get_output(0))
    network.mark_output(restoreLayer.get_output(0))
    
    profile = builder.create_optimization_profile()
    profile.set_shape('inputTensor', (1,1,1,1),(4,3,4,5),(8,6,8,10))
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    return builder.build_engine(network, config)

def run(dimIn):
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")    

    stream  = cuda.Stream()
    context = engine.create_execution_context()
    context.set_binding_shape(0,dimIn)

    data    = np.arange(np.prod(dimIn),dtype=np.float32).reshape(dimIn)
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(context.get_binding_shape(1), dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
    out2_h  = np.empty(context.get_binding_shape(2), dtype = trt.nptype(engine.get_binding_dtype(2)))
    out2_d  = cuda.mem_alloc(out2_h.nbytes)
            
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(dimIn[0], [int(in1_d), int(out1_d),int(out2_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    cuda.memcpy_dtoh_async(out2_h, out2_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    #print(data)
    print("out1_h:", out1_h.shape)
    #print(out1_h)
    print("out2_h:", out2_h.shape)
    #print(out2_h)
    
    print("test",dimIn,"finish!")
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    cuda.Device(0).make_context()
    run([1,1,1,1])
    run([4,3,4,5])
    run([8,6,8,10])
    cuda.Context.pop()
    print("finish!!")
```

+ 程序输出结果
```
data: (1, 1, 1, 1)
out1_h: (1, 1, 1, 1, 1)
out2_h: (1, 1, 1, 1)
test [1, 1, 1, 1] finish!
build engine sucessfully.

data: (4, 3, 4, 5)
out1_h: (4, 3, 4, 5, 1)
out2_h: (4, 3, 4, 5)
test [4, 3, 4, 5] finish!
build engine sucessfully.

data: (8, 6, 8, 10)
out1_h: (8, 6, 8, 10, 1)
out2_h: (8, 6, 8, 10)
test [8, 6, 8, 10] finish!
finish!!
```

<div style="page-break-after:always;"></div>
## Slice 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 4                                                                                         # 输入张量 HWC
wIn     = 5
cIn     = 3
data    = np.arange(cIn).reshape(cIn,1,1)*100 + np.arange(hIn).reshape(1,hIn,1)*10 + np.arange(wIn).reshape(1,1,wIn)# 输入张量
data    = data.astype(np.float32)

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替换部分
    slice = network.add_slice(inputTensor,(0,0,0),(2,3,4),(1,1,1))
    print("slice->", slice.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(slice.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (3, 4, 5)，百位表示 C 维编号，十位表示 H 维编号，个位表示 W 维编号
$$
\left[\begin{matrix}
    \left[\begin{matrix}
          0. &   1. &   2. &   3. &   4. \\
         10. &  11. &  12. &  13. &  14. \\
         20. &  21. &  22. &  23. &  24. \\
         30. &  31. &  32. &  33. &  34.
    \end{matrix}\right]
    \left[\begin{matrix}
        100. & 101. & 102. & 103. & 104. \\
        110. & 111. & 112. & 113. & 114. \\
        120. & 121. & 122. & 123. & 124. \\
        130. & 131. & 132. & 133. & 134.
    \end{matrix}\right]
    \left[\begin{matrix}
        200. & 201. & 202. & 203. & 204. \\
        210. & 211. & 212. & 213. & 214. \\
        220. & 221. & 222. & 223. & 224. \\
        230. & 231. & 232. & 233. & 234.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (2, 3, 4)，以 (0,0,0) 元素为起点，切出 (2,3,4) 形状的张量，各维上的步长为 (1,1,1)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
          0. &   1. &   2. &   3. \\
         10. &  11. &  12. &  13. \\
         20. &  21. &  22. &  23.
    \end{matrix}\right]
    \left[\begin{matrix}
        100. & 101. & 102. & 103. \\
        110. & 111. & 112. & 113. \\
        120. & 121. & 122. & 123.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### start & shape & stride
```python
    slice = network.add_slice(inputTensor,(0,0,0),(0,0,0),(0,0,0))                                  # 替换部分
    slice.start  = (0,0,1)                                                                          # 起点元素，可覆盖函数 add_slice 的参数
    slice.shape  = (3,2,2)                                                                          # 切出张量形状，可覆盖函数 add_slice 的参数
    slice.stride = (1,2,2)                                                                          # 各维上的步长，可覆盖函数 add_slice 的参数
    print("slice->", slice.get_output(0).shape)
```

+ 输出张量 (3, 2, 2)，以 (0,0,1) 元素为起点，切出 (3,2,2) 形状的张量，各维上的步长为 (1,2,2)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
          1. &   3. \\
         21. &  23.
    \end{matrix}\right]
    \left[\begin{matrix}
        101. & 103. \\
        121. & 123.
    \end{matrix}\right]
    \left[\begin{matrix}
        201. & 203. \\
        221. & 223.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### mode
```python
    slice = network.add_slice(inputTensor,(0,0,0),(2,3,4),(2,2,2))                                  # 替换部分
    slice.mode = trt.SliceMode.WRAP
    print("slice->", slice.get_output(0).shape)
```

+ 输出张量 (2, 3, 4)，以 (0,0,0) 元素为起点，超出边界的元素从起始元素继续切片
$$
\left[\begin{matrix}
    \left[\begin{matrix}
          0. &   2. &   4. &   1. \\
         20. &  22. &  24. &  21.
          0. &   2. &   4. &   1.
    \end{matrix}\right]
    \left[\begin{matrix}
        200. & 202. & 204. & 201. \\
        220. & 222. & 224. & 221. \\
        200. & 202. & 204. & 201.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 注意，可用的选项
|trt.SliceMode 名|说明|
|:---|:---|
|DEFAULT | 默认模式，超出边界就报错 |
|WRAP | 超出边界就从起始元素继续 |

---
### 静态 set_input
```python
    network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))      # 替换部分，使用显式 batch 模式
    #...
    slice = network.add_slice(inputTensor,(0,0,0),(0,0,0),(0,0,0))
    start = network.add_constant((3,), np.array([0,0,1],dtype=np.int32))                            # 输入的三个参数用张量来代替
    shape = network.add_constant((3,), np.array([3,2,2],dtype=np.int32))
    stride= network.add_constant((3,), np.array([1,2,2],dtype=np.int32))
    slice.set_input(0,inputTensor)
    slice.set_input(1,start.get_output(0))
    slice.set_input(2,shape.get_output(0))
    slice.set_input(3,stride.get_output(0))
    print("slice->", slice.get_output(0).shape)
```

+ 输入张量 (3, 2, 2)，以 (0,0,1) 元素为起点，切出 (3,2,2) 形状的张量，各维上的步长为 (1,2,2)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
          1. &   3. \\
         21. &  23.
    \end{matrix}\right]
    \left[\begin{matrix}
        101. & 103. \\
        121. & 123.
    \end{matrix}\right]
    \left[\begin{matrix}
        201. & 203. \\
        221. & 223.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### 动态 set_input
+ 见 shuffle 层“动态 set_input”

<div style="page-break-after:always;"></div>
## Soft Max 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 3                                                                                         # 输入张量 HWC
wIn     = 3
cIn     = 3
data    = np.arange(1,1+cIn*hIn*wIn,dtype=np.float32).reshape(cIn,hIn,wIn)                          # 输入张量

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替换部分
    sm = network.add_softmax(inputTensor)
    print("sm->", sm.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(sm.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (3, 3, 3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1. & 2. & 3. \\
        4. & 5. & 6. \\ 
        7. & 8. & 9.
    \end{matrix}\right]
    \left[\begin{matrix}
        10. & 11. & 12. \\
        13. & 14. & 15. \\
        16. & 17. & 18.
    \end{matrix}\right]
    \left[\begin{matrix}
        19. & 20. & 21. \\
        22. & 23. & 24. \\ 
        25. & 26. & 27.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (3, 3, 3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \textcolor[rgb]{0,0.5,0}{0.00000002} & 0.00000002 & 0.00000002 \\
        0.00000002 & 0.00000002 & 0.00000002 \\
        0.00000002 & 0.00000002 & 0.00000002
    \end{matrix}\right]
    \left[\begin{matrix}
        \textcolor[rgb]{0,0,1}{0.00012339} & 0.00012339 & 0.00012339 \\
        0.00012339 & 0.00012339 & 0.00012339 \\
        0.00012339 & 0.00012339 & 0.00012339
    \end{matrix}\right]
    \left[\begin{matrix}
        \textcolor[rgb]{1,0,0}{0.9998766} & 0.9998766 & 0.9998766 \\
        0.9998766 & 0.9998766 & 0.9998766 \\
        0.9998766 & 0.9998766 & 0.9998766
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：
$$
\frac{ \left[ e^{1},e^{10},e^{19} \right] }{ e^{1}+e^{10}+e^{19} }
=
\left[ \textcolor[rgb]{0,0.5,0}{1.523 \times 10^{-8}}, \textcolor[rgb]{0,0,1}{1.234 \times 10^{-4}}, \textcolor[rgb]{1,0,0}{9.998 \times 10^{-1}} \right]
$$

---
### axes
```python
    axesIndex = 0                                                                                   # 替换部分
    sm = network.add_softmax(inputTensor)
    sm.axes = 1 << axesIndex                                                                        # 运算的轴号，默认值为 1<<0
    print("sm->", sm.get_output(0).shape)
```

+ 输出张量 (3, 3, 3)，其中指定 axes=1<<0，在最高“非batch”维（C 维）上 softmax，各通道相同 HW 位置上元素之和为 1，与初始代码相同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0.00000002 & 0.00000002 & 0.00000002 \\
        0.00000002 & 0.00000002 & 0.00000002 \\
        0.00000002 & 0.00000002 & 0.00000002
    \end{matrix}\right]
    \left[\begin{matrix}
        0.00012339 & 0.00012339 & 0.00012339 \\
        0.00012339 & 0.00012339 & 0.00012339 \\
        0.00012339 & 0.00012339 & 0.00012339
    \end{matrix}\right]
    \left[\begin{matrix}
        0.9998766  & 0.9998766  & 0.9998766  \\
        0.9998766  & 0.9998766  & 0.9998766  \\
        0.9998766  & 0.9998766  & 0.9998766 ]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (3, 3, 3)，其中指定 axes=1<<1，在次高“非batch”维（H 维）上 softmax，单通道内同列元素之和为 1
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0.00235563 & 0.00235563 & 0.00235563 \\
        0.04731416 & 0.04731416 & 0.04731416 \\
        0.95033026 & 0.95033026 & 0.95033026
    \end{matrix}\right]
    \left[\begin{matrix}
        0.00235563 & 0.00235563 & 0.00235563 \\
        0.04731416 & 0.04731416 & 0.04731416 \\
        0.95033026 & 0.95033026 & 0.95033026
    \end{matrix}\right]
    \left[\begin{matrix}
        0.00235563 & 0.00235563 & 0.00235563 \\
        0.04731416 & 0.04731416 & 0.04731416 \\
        0.95033026 & 0.95033026 & 0.95033026]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (3, 3, 3)，其中指定 axes=1<<2，在季高“非batch”维（W 维）上 softmax，单通道内同行元素之和为 1
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0.09003057 & 0.24472848 & 0.66524094 \\
        0.09003057 & 0.24472848 & 0.66524094 \\
        0.09003057 & 0.24472848 & 0.66524094
    \end{matrix}\right]
    \left[\begin{matrix}
        0.09003057 & 0.24472848 & 0.66524094 \\
        0.09003057 & 0.24472848 & 0.66524094 \\
        0.09003057 & 0.24472848 & 0.66524094
    \end{matrix}\right]
    \left[\begin{matrix}
        0.09003057 & 0.24472848 & 0.66524094 \\
        0.09003057 & 0.24472848 & 0.66524094 \\
        0.09003057 & 0.24472848 & 0.66524094
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 不能同时指定两个及以上的 axes，如使用 axes = 1<<0 + 1<<1，报错信息
```shell
[TensorRT] ERROR: Parameter check failed at: ../builder/Layers.h::setAxes::380, condition: isSingleBit(axes)
```

<div style="page-break-after:always;"></div>
## Top K 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

np.random.seed(97)

hIn     = 4                                                                                         # 输入张量 HWC
wIn     = 5
cIn     = 3
data    = np.random.permutation(np.arange(cIn*hIn*wIn,dtype=np.float32)).reshape(cIn,hIn,wIn)       # 输入张量

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替换部分
    topk = network.add_topk(inputTensor, trt.TopKOperation.MAX, 3, 1<<0)
    print("topk->", topk.get_output(0).shape, topk.get_output(1).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(topk.get_output(0))
    network.mark_output(topk.get_output(1))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
    out2_h  = np.empty(engine.get_binding_shape(2),dtype = trt.nptype(engine.get_binding_dtype(2)))
    out2_d  = cuda.mem_alloc(out2_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d), int(out2_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    cuda.memcpy_dtoh_async(out2_h, out2_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    print("out2_h:", out2_h.shape)
    print(out2_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (3, 4, 5)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         9. & 52. &  2. & 27. & 49. \\
         0. & 59. & 22. &  6. & 11. \\
        45. & 33. &  8. & 31. & 37. \\
        23. & 21. &  1. & 55. & 17. \\
    \end{matrix}\right]
    \left[\begin{matrix}
        34. & 15. & 32. & 54. & 39. \\
        10. & 43. & 57. & 30. & 12. \\
        19. & 38. & 40. & 36. & 25. \\
         3. & 42. & 24. & 16. & 47. \\
    \end{matrix}\right]
    \left[\begin{matrix}
        13. & 14. & 58. & 46. & 50. \\
        48. & 44. & 29. & 20. & 18. \\
         4. &  5. & 56. & 28. &  7. \\
        53. & 51. & 41. & 35. & 26. \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 0 形状 (3, 4, 5)，对各 C 维中相同 HW 位置上的元素取降序前 3 名
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        34. & 52. & 58. & 54. & 50. \\
        48. & 59. & 57. & 30. & 18. \\
        45. & 38. & 56. & 36. & 37. \\
        53. & 51. & 41. & 55. & 47. \\
    \end{matrix}\right]
    \left[\begin{matrix}
        13. & 15. & 32. & 46. & 49. \\
        10. & 44. & 29. & 20. & 12. \\
        19. & 33. & 40. & 31. & 25. \\
        23. & 42. & 24. & 35. & 26. \\
    \end{matrix}\right]
    \left[\begin{matrix}
         9. & 14. &  2. & 27. & 39. \\
         0. & 43. & 22. &  6. & 11. \\
         4. &  5. &  8. & 28. &  7. \\
         3. & 21. &  1. & 16. & 17. \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (3, 4, 5)，输出张量 0 中各元素在输入张量中的通道号，$output0 \left[ output1 \left[ c,h,w \right] ,h,w \right] = input \left[c,h,w \right]$
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1 & 0 & 2 & 1 & 2 \\
        2 & 0 & 1 & 1 & 2 \\
        0 & 1 & 2 & 1 & 0 \\
        2 & 2 & 2 & 0 & 1 \\
    \end{matrix}\right]
    \left[\begin{matrix}
        2 & 1 & 1 & 2 & 0 \\
        1 & 2 & 2 & 2 & 1 \\
        1 & 0 & 1 & 0 & 1 \\
        0 & 1 & 1 & 2 & 2 \\
    \end{matrix}\right]
    \left[\begin{matrix}
        0 & 2 & 0 & 0 & 1 \\
        0 & 1 & 0 & 0 & 0 \\
        2 & 2 & 0 & 2 & 2 \\
        1 & 0 & 0 & 1 & 0 \\
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### op
```python
    topk = network.add_topk(inputTensor, trt.TopKOperation.MAX, 3, 1<<0)                            # 替换部分
    topk.op = trt.TopKOperation.MIN                                                                 # topK 从最大值还是最小值取起，可覆盖函数 add_topk 的参数
    print("topk->", topk.get_output(0).shape, topk.get_output(1).shape)
```

+ 输出张量 0 形状 (3, 4, 5)，其中指定 op=MIN，对各通道中相同 HW 位置上的元素取升序前 3 名
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         9. & 14. &  2. & 27. & 39. \\
         0. & 43. & 22. &  6. & 11. \\
         4. &  5. &  8. & 28. &  7. \\
         3. & 21. &  1. & 16. & 17. \\
    \end{matrix}\right]
    \left[\begin{matrix}
        13. & 15. & 32. & 46. & 49. \\
        10. & 44. & 29. & 20. & 12. \\
        19. & 33. & 40. & 31. & 25. \\
        23. & 42. & 24. & 35. & 26. \\
    \end{matrix}\right]
    \left[\begin{matrix}
        34. & 52. & 58. & 54. & 50. \\
        48. & 59. & 57. & 30. & 18. \\
        45. & 38. & 56. & 36. & 37. \\
        53. & 51. & 41. & 55. & 47. \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (3, 4, 5)，其中指定 op=MIN
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        0 & 2 & 0 & 0 & 1 \\
        0 & 1 & 0 & 0 & 0 \\
        2 & 2 & 0 & 2 & 2 \\
        1 & 0 & 0 & 1 & 0 \\
    \end{matrix}\right]
    \left[\begin{matrix}
        2 & 1 & 1 & 2 & 0 \\
        1 & 2 & 2 & 2 & 1 \\
        1 & 0 & 1 & 0 & 1 \\
        0 & 1 & 1 & 2 & 2 \\
    \end{matrix}\right]
    \left[\begin{matrix}
        1 & 0 & 2 & 1 & 2 \\
        2 & 0 & 1 & 1 & 2 \\
        0 & 1 & 2 & 1 & 0 \\
        2 & 2 & 2 & 0 & 1 \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 注意，可用的选项
|trt.TopKOperation 名|说明|
|:---|:---|
|MAX | 从最大值开始取 |
|MIN | 从最小值开始取 |

---
### k
```python
    topk = network.add_topk(inputTensor, trt.TopKOperation.MAX, 3, 1<<0)                            # 替换部分
    topk.k = 2                                                                                      # 选取的元素数，可覆盖函数 add_topk 的参数，TRT7 中最大可取 3840（原 TRT 文档有误）
    print("topk->", topk.get_output(0).shape, topk.get_output(1).shape)
```

+ 输出张量 0 形状 (2, 4, 5)，其中指定 k=2，对各通道中相同 HW 位置上的元素取降序前 2 名
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        34. & 52. & 58. & 54. & 50. \\
        48. & 59. & 57. & 30. & 18. \\
        45. & 38. & 56. & 36. & 37. \\
        53. & 51. & 41. & 55. & 47. \\
    \end{matrix}\right]
    \left[\begin{matrix}
        13. & 15. & 32. & 46. & 49. \\
        10. & 44. & 29. & 20. & 12. \\
        19. & 33. & 40. & 31. & 25. \\
        23. & 42. & 24. & 35. & 26. \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (2, 4, 5)，其中指定 k=2
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1 & 0 & 2 & 1 & 2 \\
        2 & 0 & 1 & 1 & 2 \\
        0 & 1 & 2 & 1 & 0 \\
        2 & 2 & 2 & 0 & 1 \\
    \end{matrix}\right]
    \left[\begin{matrix}
        2 & 1 & 1 & 2 & 0 \\
        1 & 2 & 2 & 2 & 1 \\
        1 & 0 & 1 & 0 & 1 \\
        0 & 1 & 1 & 2 & 2 \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 在 GTX1070 (sm=61) 上规定了最大 k 为 3840，超出后报错
```shell
[TensorRT] ERROR: Parameter check failed at: ../builder/Layers.cpp::TopKLayer::3528, condition: k > 0 && k <= MAX_TOPK_K
```

---
### axes
```python
    axesIndex = 0                                                                                   # 替换部分
    topk = network.add_softmax(inputTensor)
    topk.axes = 1 << axesIndex                                                                        # 规约的轴号，默认值为 1<<0
    print("topk->", topk.get_output(0).shape)
```

+ 输出张量 0 形状 (3, 4, 5)，其中指定 axes = 1 << 0，对各 C 维中相同 HW 位置上的元素取降序前 3 名，与初始代码相同
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        34. & 52. & 58. & 54. & 50. \\
        48. & 59. & 57. & 30. & 18. \\
        45. & 38. & 56. & 36. & 37. \\
        53. & 51. & 41. & 55. & 47. \\
    \end{matrix}\right]
    \left[\begin{matrix}
        13. & 15. & 32. & 46. & 49. \\
        10. & 44. & 29. & 20. & 12. \\
        19. & 33. & 40. & 31. & 25. \\
        23. & 42. & 24. & 35. & 26. \\
    \end{matrix}\right]
    \left[\begin{matrix}
         9. & 14. &  2. & 27. & 39. \\
         0. & 43. & 22. &  6. & 11. \\
         4. &  5. &  8. & 28. &  7. \\
         3. & 21. &  1. & 16. & 17. \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (3, 4, 5)，其中指定 axes = 1 << 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1 & 0 & 2 & 1 & 2 \\
        2 & 0 & 1 & 1 & 2 \\
        0 & 1 & 2 & 1 & 0 \\
        2 & 2 & 2 & 0 & 1 \\
    \end{matrix}\right]
    \left[\begin{matrix}
        2 & 1 & 1 & 2 & 0 \\
        1 & 2 & 2 & 2 & 1 \\
        1 & 0 & 1 & 0 & 1 \\
        0 & 1 & 1 & 2 & 2 \\
    \end{matrix}\right]
    \left[\begin{matrix}
        0 & 2 & 0 & 0 & 1 \\
        0 & 1 & 0 & 0 & 0 \\
        2 & 2 & 0 & 2 & 2 \\
        1 & 0 & 0 & 1 & 0 \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 0 形状 (3, 3, 5)，其中指定 axes = 1 << 1，对各 H 维中相同 CW 位置上的元素取降序前 3 名
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        45. & 59. & 22. & 55. & 49. \\
        23. & 52. &  8. & 31. & 37. \\
         9. & 33. &  2. & 27. & 17. \\
    \end{matrix}\right]
    \left[\begin{matrix}
        34. & 43. & 57. & 54. & 47. \\
        19. & 42. & 40. & 36. & 39. \\
        10. & 38. & 32. & 30. & 25. \\
    \end{matrix}\right]
    \left[\begin{matrix}
        53. & 51. & 58. & 46. & 50. \\
        48. & 44. & 56. & 35. & 26. \\
        13. & 14. & 41. & 28. & 18.]] \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (3, 3, 5)，其中指定 axes = 1 << 1
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        2 & 1 & 1 & 3 & 0 \\
        3 & 0 & 2 & 2 & 2 \\
        0 & 2 & 0 & 0 & 3 \\
    \end{matrix}\right]
    \left[\begin{matrix}
        0 & 1 & 1 & 0 & 3 \\
        2 & 3 & 2 & 2 & 0 \\
        1 & 2 & 0 & 1 & 2 \\
    \end{matrix}\right]
    \left[\begin{matrix}
        3 & 3 & 0 & 0 & 0 \\
        1 & 1 & 2 & 3 & 3 \\
        0 & 0 & 3 & 2 & 1 \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 0 形状 (3, 4, 3)，其中指定 axes = 1 << 2，对各 w 维中相同 Ch 位置上的元素取降序前 3 名
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        52. & 49. & 27. \\
        59. & 22. & 11. \\
        45. & 37. & 33. \\
        55. & 23. & 21. \\
    \end{matrix}\right]
    \left[\begin{matrix}
        54. & 39. & 34. \\
        57. & 43. & 30. \\
        40. & 38. & 36. \\
        47. & 42. & 24. \\
    \end{matrix}\right]
    \left[\begin{matrix}
        58. & 50. & 46. \\
        48. & 44. & 29. \\
        56. & 28. &  7. \\
        53. & 51. & 41. \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (3, 4, 3)，其中指定 axes = 1 << 2
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1 & 4 & 3 \\
        1 & 2 & 4 \\
        0 & 4 & 1 \\
        3 & 0 & 1 \\
    \end{matrix}\right]
    \left[\begin{matrix}
        3 & 4 & 0 \\
        2 & 1 & 3 \\
        2 & 1 & 3 \\
        4 & 1 & 2 \\
    \end{matrix}\right]
    \left[\begin{matrix}
        2 & 4 & 3 \\
        0 & 1 & 2 \\
        2 & 3 & 4 \\
        0 & 1 & 2 \\
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 不能同时指定两个及以上的 axes，如使用 axes = 1<<0 + 1<<1，报错信息
```shell
[TensorRT] ERROR: (Unnamed Layer* 0) [TopK]: reduceAxes must specify exactly one dimension
```

<div style="page-break-after:always;"></div>
## Unary 层

### 初始代码
```python
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 3                                                                                         # 输入张量 HWC
wIn     = 3
cIn     = 1
data    = np.arange(-4,5,dtype=np.float32).reshape(cIn,hIn,wIn)                                     # 输入张量

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替换部分
    unary = network.add_unary(inputTensor, trt.UnaryOperation.ABS)
    print("unary->", unary.get_output(0).shape)
    #-----------------------------------------------------------------------------------------------# 可替换部分

    network.mark_output(unary.get_output(0))
    return builder.build_cuda_engine(network)

def run():
    engine = buildEngine()
    if engine == None:
        print("build engine failed.\n")
        return
    print("build engine sucessfully.\n")
    
    context = engine.create_execution_context()
    stream  = cuda.Stream()
    in1_h   = np.ascontiguousarray(data.reshape(-1))
    in1_d   = cuda.mem_alloc(in1_h.nbytes)
    out1_h  = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    out1_d  = cuda.mem_alloc(out1_h.nbytes)
        
    cuda.memcpy_htod_async(in1_d, in1_h, stream)
    context.execute_async(1, [int(in1_d), int(out1_d)], stream.handle)
    cuda.memcpy_dtoh_async(out1_h, out1_d, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("out1_h:", out1_h.shape)
    print(out1_h)
    
if __name__ == '__main__':
    np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
    run()
```

+ 输入张量 (1, 3, 3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        -4. & -3. & -2. \\
        -1. &  0. &  1. \\
         2. &  3. &  4.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 (1, 3, 3)
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        4. & 3. & 2. \\
        1. & 0. & 1. \\
        2. & 3. & 4.
    \end{matrix}\right]
\end{matrix}\right]
$$

---
### op
```python
    unary = network.add_unary(inputTensor, trt.UnaryOperation.ABS)                                  # 替换部分
    unary.op = trt.UnaryOperation.NEG                                                               # 一元函数，可覆盖函数 add_unary 的参数
    print("unary->", unary.get_output(0).shape)
```

+ 输出张量 (1, 3, 3)，使用求相反数的一元函数
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        4.  &  3. &  2. \\
        1.  & -0. & -1. \\
        -2. & -3. & -4.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 注意，可用的一元函数
|trt.UnaryOperation 名|函数|
|:---|:---|
|NEG | $-x$ |
|NOT | $not \left( x \right)$ |
|ABS | $\left| x \right|$ |
|FLOOR | $\lfloor x \rfloor$ |
|CEIL | $\lceil x \rceil$ |
|RECIP | $\frac{1}{x}$ |
|SQRT | $ \sqrt{x} $ |
|EXP | $\exp \left( x \right)$ |
|LOG | $ \log \left( x \right) $（以 e 为底）|
|ERF | $ erf \left( x \right) = \int_{0}^{x} \exp\left(-t^{2}\right)dt$ |
|SIN | $\sin \left( x \right)$ |
|COS | $\cos \left( x \right)$ |
|TAN | $\tan \left( x \right)$ |
|ASIN | $\sin^{-1} \left( x \right)$ |
|ACOS | $\cos^{-1} \left( x \right)$ |
|ATAN | $\tan^{-1} \left( x \right)$ |
|SINH | $\sinh \left( x \right)$ |
|COSH | $\cosh \left( x \right)$ |
|<font color=#FF0000>没有TANH</font>| 作为 activation 函数|
|ASINH | $\sinh^{-1} \left( x \right)$ |
|ACOSH | $\cosh^{-1} \left( x \right)$ |
|ATANH | $\tanh^{-1} \left( x \right)$ |

---

