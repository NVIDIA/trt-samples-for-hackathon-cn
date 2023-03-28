# RNN Layer
+ Simple example
+ seq_lengths
+ num_layers & hidden_size & max_seq_length & data_length & op
+ input_mode
+ direction
+ hidden_state
+ cell_state（单层单向 LSTM 的例子）
+ 单层双向 LSTM 的例子
+ 多层数的 RNN 的例子

---
## Simple example
+ Refer to ReLURNN.py，实现一个简单的 ReLU RNN

+ Shape of input tensor 0: (1,3,4,7)，3 个独立输入，每个输入 4 个单词，每个单词 7 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1. & 1. & 1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. & 1. & 1.
        \end{matrix}\right] \\
        \left[\begin{matrix}
            1. & 1. & 1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. & 1. & 1.
        \end{matrix}\right] \\
        \left[\begin{matrix}
            1. & 1. & 1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. & 1. & 1. \\
            1. & 1. & 1. & 1. & 1. & 1. & 1.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output  0  tensor: (1,3,4,5)，3 个独立输出，每个包含 4 个隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            {\color{#007F00}{7.}} & {\color{#007F00}{7.}} & {\color{#007F00}{7.}} & {\color{#007F00}{7.}} & {\color{#007F00}{7.}} \\
            {\color{#0000FF}{42.}} & {\color{#0000FF}{42.}} & {\color{#0000FF}{42.}} & {\color{#0000FF}{42.}} & {\color{#0000FF}{42.}} \\
             217. &  217. &  217. &  217. &  217. \\
            1092. & 1092. & 1092. & 1092. & 1092.
        \end{matrix}\right] \\
        \left[\begin{matrix}
               7. &    7. &    7. &    7. &    7. \\
              42. &   42. &   42. &   42. &   42. \\
             217. &  217. &  217. &  217. &  217. \\
            1092. & 1092. & 1092. & 1092. & 1092.
        \end{matrix}\right] \\
        \left[\begin{matrix}
               7. &    7. &    7. &    7. &    7. \\
              42. &   42. &   42. &   42. &   42. \\
             217. &  217. &  217. &  217. &  217. \\
            1092. & 1092. & 1092. & 1092. & 1092.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output  1  tensor: (1,3,1,5)，3 个独立输出，每个包含 1 个最终隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1092. & 1092. & 1092. & 1092. & 1092.
        \end{matrix}\right] \\
        \left[\begin{matrix}
            1092. & 1092. & 1092. & 1092. & 1092.
        \end{matrix}\right] \\
        \left[\begin{matrix}
            1092. & 1092. & 1092. & 1092. & 1092.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程：默认使用单向 RNN，隐藏状态全 0，并且要对输入张量作线性变换
$$
\begin{aligned}
h_{1}&=\textbf{ReLU}\left(W_{i,X}\cdot x_{1}+W_{i,H}\cdot h_{0}+b_{i,X}+b_{i,H}\right)\\
&=\textbf{ReLU}
\left(
  \left[\begin{matrix}
   1 & 1 & 1 & 1 & 1 & 1 & 1 \\
   1 & 1 & 1 & 1 & 1 & 1 & 1 \\
   1 & 1 & 1 & 1 & 1 & 1 & 1 \\
   1 & 1 & 1 & 1 & 1 & 1 & 1 \\
   1 & 1 & 1 & 1 & 1 & 1 & 1
  \end{matrix}\right]
  \left[\begin{matrix}
   1 \\ 1 \\ 1 \\ 1 \\ 1 \\ 1 \\ 1
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   1 & 1 & 1 & 1 & 1 \\
   1 & 1 & 1 & 1 & 1 \\
   1 & 1 & 1 & 1 & 1 \\
   1 & 1 & 1 & 1 & 1 \\
   1 & 1 & 1 & 1 & 1
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
&=\left({\color{#007F00}{7}},{\color{#007F00}{7}},{\color{#007F00}{7}},{\color{#007F00}{7}},{\color{#007F00}{7}}
  \right)^\mathrm{T}\\
\\\hfill\\
h_{2}&=\textbf{ReLU}\left(W_{i,X}\cdot x_{2}+W_{i,H}\cdot h_{1}+b_{i,X}+b_{i,H}\right)\\
&=\textbf{ReLU}
\left(
  \left[\begin{matrix}
   1 & 1 & 1 & 1 & 1 & 1 & 1 \\
   1 & 1 & 1 & 1 & 1 & 1 & 1 \\
   1 & 1 & 1 & 1 & 1 & 1 & 1 \\
   1 & 1 & 1 & 1 & 1 & 1 & 1 \\
   1 & 1 & 1 & 1 & 1 & 1 & 1
  \end{matrix}\right]
  \left[\begin{matrix}
   1 \\ 1 \\ 1 \\ 1 \\ 1 \\ 1 \\ 1
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   1 & 1 & 1 & 1 & 1 \\
   1 & 1 & 1 & 1 & 1 \\
   1 & 1 & 1 & 1 & 1 \\
   1 & 1 & 1 & 1 & 1 \\
   1 & 1 & 1 & 1 & 1
  \end{matrix}\right]
  \left[\begin{matrix}
   7 \\ 7 \\ 7 \\ 7 \\ 7 \\ 7 \\ 7
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0
  \end{matrix}\right]
  +
  \left[\begin{matrix}
   0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0
  \end{matrix}\right]
\right)\\
&=\textbf{ReLU}\left(\left(42,42,42,42,42\right)^\mathrm{T}\right)\\
&=\left(
    {\color{#0000FF}{42}},
    {\color{#0000FF}{42}},
    {\color{#0000FF}{42}},
    {\color{#0000FF}{42}},
    {\color{#0000FF}{42}}
  \right)^\mathrm{T}
\end{aligned}
$$

+ 注意设置 config.set_memory_pool_limit，否则可能报错：
```
[TensorRT] ERROR: 10: [optimizer.cpp::computeCosts::1855] Error Code 10: Internal Error (Could not find any implementation for node (Unnamed Layer* 0) [RNN].)
```

+ 收到警告，以后 RNNV2 层可能被废弃
```
DeprecationWarning: Use addLoop instead.
```

---

## seq_lengths
+ Refer to Seq_lengths.py，设置每个独立输入的真实长度

+ Shape of output  0  tensor: (1,3,4,5)，三个独立输入分别迭代 4 次、3 次和 2 次，长度不足的部分计算结果为 0
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
               7. &    7. &    7. &    7. &    7. \\
              42. &   42. &   42. &   42. &   42. \\
             217. &  217. &  217. &  217. &  217. \\
            1092. & 1092. & 1092. & 1092. & 1092.
        \end{matrix}\right] \\
        \left[\begin{matrix}
               7. &    7. &    7. &    7. &    7. \\
              42. &   42. &   42. &   42. &   42. \\
             217. &  217. &  217. &  217. &  217. \\
               0. &    0. &    0. &    0. &    0.
        \end{matrix}\right] \\
        \left[\begin{matrix}
               7. &    7. &    7. &    7. &    7. \\
              42. &   42. &   42. &   42. &   42. \\
               0. &    0. &    0. &    0. &    0. \\
               0. &    0. &    0. &    0. &    0.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output  1  tensor: (1,3,1,5)，3 个独立输出，记录每个独立输入的末状态
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1092. & 1092. & 1092. & 1092. & 1092.
        \end{matrix}\right] \\
        \left[\begin{matrix}
            217.  & 217.  & 217.  & 217.  &  217.
        \end{matrix}\right] \\
        \left[\begin{matrix}
              42. &   42. &   42. &   42. &   42.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---

## num_layers & hidden_size & max_seq_length & data_length & op
+ Refer to Num_layers+Hidden_size+Max_seq_length+Data_length+Op.py，构建 RNNv2 层后打印其层数、隐藏层维度数、最大序列长度、真实数据长度、RNN 类型，注意不可修改

+ 结果与初始范例代码相同，多了 RNN 层的相关信息的输出：
```
num_layers=1
hidden_size=5
max_seq_length=4
data_length=7
```

+ 可用的 RNN 模式
| trt.RNNOperation 名 | 说明               |
| :-----------------: | :----------------- |
|        RELU         | 单门 ReLU 激活 RNN |
|        TANH         | 单门 tanh 激活 RNN |
|        LSTM         | 4 门 LSTM          |
|         GRU         | 3 门 GRU           |

---

## input_mode
+ Refer to InputMode.py，设置是否需要对输入张量进行预线性变换

+ Shape of output  0  tensor: (1,3,4,5)，3 个独立输出，每个包含 4 个隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            {\color{#007F00}{1.}} & {\color{#007F00}{1.}} & {\color{#007F00}{1.}} & {\color{#007F00}{1.}} & {\color{#007F00}{1.}} \\
            {\color{#0000FF}{6.}} & {\color{#0000FF}{6.}} & {\color{#0000FF}{6.}} & {\color{#0000FF}{6.}} & {\color{#0000FF}{6.}} \\
             31. &  31. &  31. &  31. &  31. \\
            156. & 156. & 156. & 156. & 156.
        \end{matrix}\right] \\
        \left[\begin{matrix}
              1. &   1. &   1. &   1. &   1. \\
              6. &   6. &   6. &   6. &   6. \\
             31. &  31. &  31. &  31. &  31. \\
            156. & 156. & 156. & 156. & 156.
        \end{matrix}\right] \\
        \left[\begin{matrix}
              1. &   1. &   1. &   1. &   1. \\
              6. &   6. &   6. &   6. &   6. \\
             31. &  31. &  31. &  31. &  31. \\
            156. & 156. & 156. & 156. & 156.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output  1  tensor: (1,3,1,5)，3 个独立输出，每个包含 1 个最终隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            156. & 156. & 156. & 156. & 156.
        \end{matrix}\right] \\
        \left[\begin{matrix}
            156. & 156. & 156. & 156. & 156.
        \end{matrix}\right] \\
        \left[\begin{matrix}
            156. & 156. & 156. & 156. & 156.
        \end{matrix}\right]
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
    {\color{#007F00}{1}},{\color{#007F00}{1}},{\color{#007F00}{1}},{\color{#007F00}{1}},{\color{#007F00}{1}},
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
    {\color{#0000FF}{6}},{\color{#0000FF}{6}},{\color{#0000FF}{6}},{\color{#0000FF}{6}},{\color{#0000FF}{6}}
  \right)^\mathrm{T}
\end{aligned}
$$

+ 注意，可用的输入张量处理方法
| trt.RNNInputMode 名 | 说明                                            |
| :-----------------: | :---------------------------------------------- |
|       LINEAR        | 对输入张量 x 做线性变换                         |
|        SKIP         | 不对输入张量 x 做线性变换（要求wIn == nHidden） |

+ 在 SKIP 模式（不做输入张量线性变换）的情况下，仍然需要设置 biasX，否则报错：
```
[TensorRT] ERROR: Missing weights/bias: (0, INPUT, 1)
[TensorRT] ERROR: 4: [network.cpp::validate::2639] Error Code 4: Internal Error (Layer (Unnamed Layer* 0) [RNN] failed validation)
```

---

## direction
+ Refer to Direction.py，设置 RNN 循环方向

+ Shape of output  0  tensor: (1,3,4,10)，3 个独立输出，每个包含 4 个隐藏状态，每个隐藏状态 5 维坐标，2 个方向在同一行并排放置
$$
\left[\begin{matrix}
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
\end{matrix}\right]
$$

+ Shape of output  1  tensor: (1,3,2,5)，3 个独立输出，每个包含 1 个最终隐藏状态，每个隐藏状态 5 维坐标，2 个方向分行放置
$$
\left[\begin{matrix}
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
\end{matrix}\right]
$$

+ 可用方向参数
| trt.RNNDirection 名 | 说明     |
| :------------------ | :------- |
| UNIDIRECTION        | 单向 RNN |
| BIDIRECTION         | 双向 RNN |

---

## hidden_state
+ Refer to Hidden_state.py，设置 RNN 初始状态张量

+ Shape of output  0  tensor: (1,3,4,5)，3 个独立输出，每个包含 4 个隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            {\color{#007F00}{12.}} & {\color{#007F00}{12.}} & {\color{#007F00}{12.}} & {\color{#007F00}{12.}} & {\color{#007F00}{12.}} \\
            {\color{#0000FF}{67.}} & {\color{#0000FF}{12.}} & {\color{#0000FF}{12.}} & {\color{#0000FF}{12.}} & {\color{#0000FF}{12.}} \\
             1.   &  342. &  342. &  342. &  342. \\
            1.    & 1717. & 1717. & 1717. & 1717.
        \end{matrix}\right] \\
        \left[\begin{matrix}
              1.  &   12. &   12. &   12. &   12. \\
              2.  &   67. &   67. &   67. &   67. \\
             1.   &  342. &  342. &  342. &  342. \\
            1.    & 1717. & 1717. & 1717. & 1717.
        \end{matrix}\right] \\
        \left[\begin{matrix}
              1.  &   12. &   12. &   12. &   12. \\
              2.  &   67. &   67. &   67. &   67. \\
             1.   &  342. &  342. &  342. &  342. \\
            1.    & 1717. & 1717. & 1717. & 1717.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output  1  tensor: (1,3,1,5)，3 个独立输出，每个包含 1 个最终隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1.    & 1717. & 1717. & 1717. & 1717.
        \end{matrix}\right] \\
        \left[\begin{matrix}
            1.    & 1717. & 1717. & 1717. & 1717.
        \end{matrix}\right] \\
        \left[\begin{matrix}
            1.    & 1717. & 1717. & 1717. & 1717.
        \end{matrix}\right]
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
    {\color{#007F00}{12}},{\color{#007F00}{12}},{\color{#007F00}{12}},{\color{#007F00}{12}},{\color{#007F00}{12}}
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
    {\color{#0000FF}{67}},{\color{#0000FF}{67}},{\color{#0000FF}{67}},{\color{#0000FF}{67}},{\color{#0000FF}{67}}
  \right)^\mathrm{T}
\end{aligned}
$$

---

## cell_state（单层单向 LSTM 的例子）
+ Refer to Cell_state.py，单层单向 LSTM 的例子

+ Shape of output  0  tensor: (1,3,4,5)，3 个独立输出，每个包含 4 个隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            {\color{#007F00}{0.760517}} & {\color{#007F00}{0.760517}} & {\color{#007F00}{0.760517}} & {\color{#007F00}{0.760517}} & {\color{#007F00}{0.760517}} \\
            {\color{#0000FF}{0.963940}} & {\color{#0000FF}{0.963940}} & {\color{#0000FF}{0.963940}} & {\color{#0000FF}{0.963940}} & {\color{#0000FF}{0.963940}} \\
            0.995037 & 0.995037 & 0.995037 & 0.995037 & 0.995037 \\
            0.999321 & 0.999321 & 0.999321 & 0.999321 & 0.999321 \\
        \end{matrix}\right] \\
        \left[\begin{matrix}
            0.760517 & 0.760517 & 0.760517 & 0.7605171& 0.7605171\\
            0.963940 & 0.963940 & 0.963940 & 0.963940 & 0.963940 \\
            0.995037 & 0.995037 & 0.995037 & 0.995037 & 0.995037 \\
            0.999321 & 0.999321 & 0.999321 & 0.999321 & 0.999321
        \end{matrix}\right] \\
        \left[\begin{matrix}
            0.760517 & 0.760517 & 0.760517 & 0.7605171& 0.7605171\\
            0.963940 & 0.963940 & 0.963940 & 0.963940 & 0.963940 \\
            0.995037 & 0.995037 & 0.995037 & 0.995037 & 0.995037 \\
            0.999321 & 0.999321 & 0.999321 & 0.999321 & 0.999321
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output  1  tensor: (1,3,1,5)，3 个独立输出，每个包含 1 个最终隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            0.999321 & 0.999321 & 0.999321 & 0.999321 & 0.999321
        \end{matrix}\right] \\
        \left[\begin{matrix}
            0.999321 & 0.999321 & 0.999321 & 0.999321 & 0.999321
        \end{matrix}\right] \\
        \left[\begin{matrix}
            0.999321 & 0.999321 & 0.999321 & 0.999321 & 0.999321
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output  2  tensor: (1,3,1,5)，3 个独立输出，每个包含 1 个最终细胞状态，每个细胞状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            3.998999 & 3.998999 & 3.998999 & 3.998999 & 3.998999
        \end{matrix}\right] \\
        \left[\begin{matrix}
            3.998999 & 3.998999 & 3.998999 & 3.998999 & 3.998999
        \end{matrix}\right] \\
        \left[\begin{matrix}
            3.998999 & 3.998999 & 3.998999 & 3.998999 & 3.998999
        \end{matrix}\right]
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
  {\color{#007F00}{0.760517}},{\color{#007F00}{0.760517}},{\color{#007F00}{0.760517}},{\color{#007F00}{0.760517}},{\color{#007F00}{0.760517}}
                                                  \right)^\mathrm{T}\\
\hfill\\
I_{2}=F_{2}=O_{2}=\textbf{sigmoid}\left(W_{?,X}\cdot x_{2}+W_{?,H}\cdot h_{1}+b_{i,X}+b_{i,H}\right)&=
  \left(0.999979,0.999979,0.999979,0.999979,0.999979\right)^\mathrm{T}\\
C_{2}=               \textbf{tanh}\left(W_{C,X}\cdot x_{2}+W_{C,H}\cdot h_{1}+b_{C,X}+b_{C,H}\right)&=
  \left(0.999999,0.999999,0.999999,0.999999,0.999999\right)^\mathrm{T}\\
c_{2}=F_{2}\cdot c_{1}+I_{2}\cdot C_{2}&=\left(1.999046,1.999046,1.999046,1.999046,1.999046\right)^\mathrm{T}\\
h_{2}=O_{2}\cdot \textbf{tanh}\left(c_{2}\right)&=\left(
  {\color{#0000FF}{0.963940}},{\color{#0000FF}{0.963940}},{\color{#0000FF}{0.963940}},{\color{#0000FF}{0.963940}},{\color{#0000FF}{0.963940}}
                                                  \right)^\mathrm{T}\\
\end{aligned}
$$

---

## 单层双向 LSTM 的例子
+ Refer to BidirectionLSTM.py
+ Shape of output  0  tensor: (1,3,4,10)，3 个独立输出，每个包含 4 个隐藏状态，每个隐藏状态 5 维坐标，2 个方向在同一行并排放置
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            0.760517 & 0.760517 & 0.760517 & 0.760517 & 0.760517 & 0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322 \\
            0.963941 & 0.963941 & 0.963941 & 0.963941 & 0.963941 & 0.995038 & 0.995038 & 0.995038 & 0.995038 & 0.995038 \\
            0.995038 & 0.995038 & 0.995038 & 0.995038 & 0.995038 & 0.963941 & 0.963941 & 0.963941 & 0.963941 & 0.963941 \\
            0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.760517 & 0.760517 & 0.760517 & 0.760517 & 0.760517
        \end{matrix}\right] \\
        \left[\begin{matrix}
            0.760517 & 0.760517 & 0.760517 & 0.760517 & 0.760517 & 0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322 \\
            0.963941 & 0.963941 & 0.963941 & 0.963941 & 0.963941 & 0.995038 & 0.995038 & 0.995038 & 0.995038 & 0.995038 \\
            0.995038 & 0.995038 & 0.995038 & 0.995038 & 0.995038 & 0.963941 & 0.963941 & 0.963941 & 0.963941 & 0.963941 \\
            0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.760517 & 0.760517 & 0.760517 & 0.760517 & 0.760517
        \end{matrix}\right] \\
        \left[\begin{matrix}
            0.760517 & 0.760517 & 0.760517 & 0.760517 & 0.760517 & 0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322 \\
            0.963941 & 0.963941 & 0.963941 & 0.963941 & 0.963941 & 0.995038 & 0.995038 & 0.995038 & 0.995038 & 0.995038 \\
            0.995038 & 0.995038 & 0.995038 & 0.995038 & 0.995038 & 0.963941 & 0.963941 & 0.963941 & 0.963941 & 0.963941 \\
            0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.760517 & 0.760517 & 0.760517 & 0.760517 & 0.760517
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output  1  tensor: (1,3,2,5)，3 个独立输出，每个包含 1 个最终隐藏状态，每个隐藏状态 5 维坐标，2 个方向分行放置
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322 \\
            0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322
        \end{matrix}\right] \\
        \left[\begin{matrix}
            0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322 \\
            0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322
        \end{matrix}\right] \\
        \left[\begin{matrix}
            0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322 \\
            0.999322 & 0.999322 & 0.999322 & 0.999322 & 0.999322
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output  2  tensor: (1,3,1,5)，3 个独立输出，每个包含 1 个最终细胞状态，每个细胞状态 5 维坐标，2 个方向分行放置
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

---

## 多层数的 RNN 的例子
+ Refer to DoubleRNN.py，两层 RNN 组合单元

+ Shape of output  0  tensor: (1,3,4,5)，3 个独立输出，每个包含 4 个隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
               1.  &    35. &    35. &    35. &    35. \\
              1.   &   385. &   385. &   385. &   385. \\
             1.    &  3010. &  3010. &  3010. &  3010. \\
            1.     & 20510. & 20510. & 20510. & 20510.
        \end{matrix}\right] \\
        \left[\begin{matrix}
               1.  &    35. &    35. &    35. &    35. \\
              1.   &   385. &   385. &   385. &   385. \\
             1.    &  3010. &  3010. &  3010. &  3010. \\
            1.     & 20510. & 20510. & 20510. & 20510.
        \end{matrix}\right] \\
        \left[\begin{matrix}
               1.  &    35. &    35. &    35. &    35. \\
              1.   &   385. &   385. &   385. &   385. \\
             1.    &  3010. &  3010. &  3010. &  3010. \\
            1.     & 20510. & 20510. & 20510. & 20510.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output  1  tensor: (1,3,2,5)，3 个独立输出，每个包含各层的最终隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             1.    &  1092. &  1092. &  1092. &  1092. \\
            1.     & 20510. & 20510. & 20510. & 20510.
        \end{matrix}\right] \\
        \left[\begin{matrix}
             1.    &  1092. &  1092. &  1092. &  1092. \\
            1.     & 20510. & 20510. & 20510. & 20510.
        \end{matrix}\right] \\
        \left[\begin{matrix}
             1.    &  1092. &  1092. &  1092. &  1092. \\
            1.     & 20510. & 20510. & 20510. & 20510.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$
