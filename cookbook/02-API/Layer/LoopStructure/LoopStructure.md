# Loop Structure

+ Simple example (For-loop + two kinds of output)
+ For-loop + runtime-iteration-count
+ While-loop + two kinds of output
+ While-loop + a bug using inplace assignment
+ iterator layer
+ Simple ReLU RNN
+ Static unidirectional LSTM
+ Static bidirectional LSTM [TODO]
+ Dynamic unidirectional LSTM

---

## Simple example (For-loop + two kinds of output)

+ Refer to For+Output.py
+ Computation process, similar to the python code below:

```python
temp = inputT0
loopOutput0 = inputT0
loopOutput1 = np.zeros([t]+inputT0.shape)
for i in range(t):
    loopOutput0 += loopOutput0
    loopOutput1[t] = loopOutput0
return loopOutput0, loopOutput1
```

+ LAST_VALUE 和 CONCATENATE 两种输出，可以只使用其中一个或两者同时使用（需要标记为两个不同的 loopOutput 层）

+ 无 iterator 的循环不能将结果 CONCATENATE 到其他维度，也不能使用 REVERSE 输出，否则报错：
```
[TRT] [E] 10: [optimizer.cpp::computeCosts::2011] Error Code 10: Internal Error (Could not find any implementation for node {ForeignNode[(Unnamed Layer* 1) [Recurrence]...(Unnamed Layer* 4) [LoopOutput]]}.)
```

+ 使用 LAST_VALUE 输出时，add_loop_output 的Input tensor只能是 rLayer.get_output(0)，如果使用 _H0.get_output(0) 则会报错：
```
[TRT] [E] 4: [scopedOpValidation.cpp::reportIllegalLastValue::89] Error Code 4: Internal Error ((Unnamed Layer* 4) [LoopOutput]: input of LoopOutputLayer with LoopOutput::kLAST_VALUE must be output from an IRecurrenceLayer)
```

+ 如果无限循环，则会收到报错：
```
terminate called after throwing an instance of 'nvinfer1::CudaRuntimeError'
  what():  an illegal memory access was encountered
Aborted (core dumped)
```

+ Loop 仅支持 float32 和 float16

---

## For-loop + runtime-iteration-count
+ Refer to For+Set_shape_input.py
+ Input tensor和输出张量与初始范例代码相同

---

## While-loop + two kinds of output
+ Refer to While+Output.py
+ Shape of input tensor 0: (1,3,4,5)，结果与初始范例代码相同

+ 输出张量 0（loopOutput0）形状 (1,3,4,5)，循环最终的结果
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            6. & 6. & 6. & 6. & 6. \\
            6. & 6. & 6. & 6. & 6. \\
            6. & 6. & 6. & 6. & 6. \\
            6. & 6. & 6. & 6. & 6.
        \end{matrix}\right]
        \left[\begin{matrix}
            6. & 6. & 6. & 6. & 6. \\
            6. & 6. & 6. & 6. & 6. \\
            6. & 6. & 6. & 6. & 6. \\
            6. & 6. & 6. & 6. & 6.
        \end{matrix}\right]
        \left[\begin{matrix}
            6. & 6. & 6. & 6. & 6. \\
            6. & 6. & 6. & 6. & 6. \\
            6. & 6. & 6. & 6. & 6. \\
            6. & 6. & 6. & 6. & 6.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 在 loopOutput1 中传入 rLayer，输出张量 1（loopOutput1）形状 (7,1,3,4,5)，保留“第 0 到第 4 次迭代的结果”
$$
\left[\begin{matrix}
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
    \end{matrix}\right] \\
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                2. & 2. & 2. & 2. & 2. \\
                2. & 2. & 2. & 2. & 2. \\
                2. & 2. & 2. & 2. & 2. \\
                2. & 2. & 2. & 2. & 2.
            \end{matrix}\right]
            \left[\begin{matrix}
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2.
            \end{matrix}\right]
            \left[\begin{matrix}
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2.
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right] \\
	\mathbf{ \cdots } \\
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                5. & 5. & 5. & 5. & 5. \\
                5. & 5. & 5. & 5. & 5. \\
                5. & 5. & 5. & 5. & 5. \\
                5. & 5. & 5. & 5. & 5.
            \end{matrix}\right]
            \left[\begin{matrix}
                5. & 5. & 5. & 5. & 5. \\
                5. & 5. & 5. & 5. & 5. \\
                5. & 5. & 5. & 5. & 5. \\
                5. & 5. & 5. & 5. & 5.
            \end{matrix}\right]
            \left[\begin{matrix}
                5. & 5. & 5. & 5. & 5. \\
                5. & 5. & 5. & 5. & 5. \\
                5. & 5. & 5. & 5. & 5. \\
                5. & 5. & 5. & 5. & 5.
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right] \\
    \left[\begin{matrix}
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
    \end{matrix}\right] \\
    \left[\begin{matrix}
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
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 在 loopOutput1 中传入 _H0，输出张量 1（loopOutput1）形状 (7,1,3,4,5)，保留“第 1 到第 5 次迭代的结果”
+ **不推荐使用**，在循环判断他条件依赖循环体张量的时候可能有错误（见下一个范例）
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                2. & 2. & 2. & 2. & 2. \\
                2. & 2. & 2. & 2. & 2. \\
                2. & 2. & 2. & 2. & 2. \\
                2. & 2. & 2. & 2. & 2.
            \end{matrix}\right]
            \left[\begin{matrix}
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2.
            \end{matrix}\right]
            \left[\begin{matrix}
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2.
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right] \\
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                 3. & 3. & 3. & 3. & 3. \\
                 3. & 3. & 3. & 3. & 3. \\
                 3. & 3. & 3. & 3. & 3. \\
                 3. & 3. & 3. & 3. & 3.
            \end{matrix}\right]
            \left[\begin{matrix}
                 3. & 3. & 3. & 3. & 3. \\
                 3. & 3. & 3. & 3. & 3. \\
                 3. & 3. & 3. & 3. & 3. \\
                 3. & 3. & 3. & 3. & 3.
            \end{matrix}\right]
            \left[\begin{matrix}
                 3. & 3. & 3. & 3. & 3. \\
                 3. & 3. & 3. & 3. & 3. \\
                 3. & 3. & 3. & 3. & 3. \\
                 3. & 3. & 3. & 3. & 3.
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right] \\
	\mathbf{ \cdots } \\
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                6. & 6. & 6. & 6. & 6. \\
                6. & 6. & 6. & 6. & 6. \\
                6. & 6. & 6. & 6. & 6. \\
                6. & 6. & 6. & 6. & 6.
            \end{matrix}\right]
            \left[\begin{matrix}
                6. & 6. & 6. & 6. & 6. \\
                6. & 6. & 6. & 6. & 6. \\
                6. & 6. & 6. & 6. & 6. \\
                6. & 6. & 6. & 6. & 6.
            \end{matrix}\right]
            \left[\begin{matrix}
                6. & 6. & 6. & 6. & 6. \\
                6. & 6. & 6. & 6. & 6. \\
                6. & 6. & 6. & 6. & 6. \\
                6. & 6. & 6. & 6. & 6.
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right] \\
    \left[\begin{matrix}
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
    \end{matrix}\right] \\
    \left[\begin{matrix}
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
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 计算过程，loopOutput1 传入 rLayer 时类似以下代码
```python
temp = inputT0
loopOutput1 = np.zeros([t]+inputT0.shape)
while (temp.reshape(-1)[0] < 6)
    loopOutput1[t] = temp
    temp += 1
loopOutput0 = temp
return loopOutput0, loopOutput1
```

+ 计算过程，loopOutput1 传入 _H0 时类似以下代码
```python
temp = inputT0
loopOutput1 = np.zeros([t]+inputT0.shape)
do
    temp += 1
    loopOutput1[t] = temp
while (temp.reshape(-1)[0] < 6)
loopOutput0 = temp
return loopOutput0, loopOutput1
```

+ 可用的循环类型
| tensorrt.TripLimit 名 |           说明           |
| :-------------------: | :----------------------: |
|         COUNT         | for 型循环，给定循环次数 |
|         WHILE         |       while 型循环       |

+ 这段范例代码要求 TensorRT>=8，TensorRT7 中运行该段代码会收到报错：
```
[TensorRT] ERROR: ../builder/myelin/codeGenerator.cpp (114) - Myelin Error in addNodeToMyelinGraph: 0 ((Unnamed Layer* 2) [Reduce] outside operation not supported within a loop body.)
```

---

## While-loop + a bug using inplace assignment

+ 输出张量 0（loopOutput0）形状 (1,3,4,5)，循环最终的结果
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            64. & 64. & 64. & 64. & 64. \\
            64. & 64. & 64. & 64. & 64. \\
            64. & 64. & 64. & 64. & 64. \\
            64. & 64. & 64. & 64. & 64.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Case 1 和 Case 2，在 loopOutput1 中传入 rLayer，输出张量 1（loopOutput1）形状 (7,1,3,4,5)，符合预期
$$
\left[\begin{matrix}
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
    \end{matrix}\right] \\
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                2. & 2. & 2. & 2. & 2. \\
                2. & 2. & 2. & 2. & 2. \\
                2. & 2. & 2. & 2. & 2. \\
                2. & 2. & 2. & 2. & 2.
            \end{matrix}\right]
            \left[\begin{matrix}
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2.
            \end{matrix}\right]
            \left[\begin{matrix}
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2.
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right] \\
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                4. & 4. & 4. & 4. & 4. \\
                4. & 4. & 4. & 4. & 4. \\
                4. & 4. & 4. & 4. & 4. \\
                4. & 4. & 4. & 4. & 4.
            \end{matrix}\right]
            \left[\begin{matrix}
                4. & 4. & 4. & 4. & 4. \\
                4. & 4. & 4. & 4. & 4. \\
                4. & 4. & 4. & 4. & 4. \\
                4. & 4. & 4. & 4. & 4.
            \end{matrix}\right]
            \left[\begin{matrix}
                4. & 4. & 4. & 4. & 4. \\
                4. & 4. & 4. & 4. & 4. \\
                4. & 4. & 4. & 4. & 4. \\
                4. & 4. & 4. & 4. & 4.
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right] \\
	\mathbf{ \cdots } \\
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                32. & 32. & 32. & 32. & 32. \\
                32. & 32. & 32. & 32. & 32. \\
                32. & 32. & 32. & 32. & 32. \\
                32. & 32. & 32. & 32. & 32.
            \end{matrix}\right]
            \left[\begin{matrix}
                32. & 32. & 32. & 32. & 32. \\
                32. & 32. & 32. & 32. & 32. \\
                32. & 32. & 32. & 32. & 32. \\
                32. & 32. & 32. & 32. & 32.
            \end{matrix}\right]
            \left[\begin{matrix}
                32. & 32. & 32. & 32. & 32. \\
                32. & 32. & 32. & 32. & 32. \\
                32. & 32. & 32. & 32. & 32. \\
                32. & 32. & 32. & 32. & 32.
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right] \\
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0.
            \end{matrix}\right]
            \left[\begin{matrix}
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0.
            \end{matrix}\right]
            \left[\begin{matrix}
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0.
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Case 1，在 loopOutput1 中传入 _H0，输出张量 1（loopOutput1）形状 (7,1,3,4,5)
+ 结果是跳步保存的，原因是循环体是原地计算（in-place），在检验判断条件时会再把循环体再算一遍
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                 4. & 4. & 4. & 4. & 4. \\
                 4. & 4. & 4. & 4. & 4. \\
                 4. & 4. & 4. & 4. & 4. \\
                 4. & 4. & 4. & 4. & 4.
            \end{matrix}\right]
            \left[\begin{matrix}
                 4. & 4. & 4. & 4. & 4. \\
                 4. & 4. & 4. & 4. & 4. \\
                 4. & 4. & 4. & 4. & 4. \\
                 4. & 4. & 4. & 4. & 4.
            \end{matrix}\right]
            \left[\begin{matrix}
                 4. & 4. & 4. & 4. & 4. \\
                 4. & 4. & 4. & 4. & 4. \\
                 4. & 4. & 4. & 4. & 4. \\
                 4. & 4. & 4. & 4. & 4.
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right] \\
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                16. & 16. & 16. & 16. & 16. \\
                16. & 16. & 16. & 16. & 16. \\
                16. & 16. & 16. & 16. & 16. \\
                16. & 16. & 16. & 16. & 16.
            \end{matrix}\right]
            \left[\begin{matrix}
                16. & 16. & 16. & 16. & 16. \\
                16. & 16. & 16. & 16. & 16. \\
                16. & 16. & 16. & 16. & 16. \\
                16. & 16. & 16. & 16. & 16.
            \end{matrix}\right]
            \left[\begin{matrix}
                16. & 16. & 16. & 16. & 16. \\
                16. & 16. & 16. & 16. & 16. \\
                16. & 16. & 16. & 16. & 16. \\
                16. & 16. & 16. & 16. & 16.
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right] \\
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                64. & 64. & 64. & 64. & 64. \\
                64. & 64. & 64. & 64. & 64. \\
                64. & 64. & 64. & 64. & 64. \\
                64. & 64. & 64. & 64. & 64.
            \end{matrix}\right]
            \left[\begin{matrix}
                64. & 64. & 64. & 64. & 64. \\
                64. & 64. & 64. & 64. & 64. \\
                64. & 64. & 64. & 64. & 64. \\
                64. & 64. & 64. & 64. & 64.
            \end{matrix}\right]
            \left[\begin{matrix}
                64. & 64. & 64. & 64. & 64. \\
                64. & 64. & 64. & 64. & 64. \\
                64. & 64. & 64. & 64. & 64. \\
                64. & 64. & 64. & 64. & 64.
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right] \\
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
    \end{matrix}\right] \\
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0.
            \end{matrix}\right]
            \left[\begin{matrix}
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0.
            \end{matrix}\right]
            \left[\begin{matrix}
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0.
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right] \\
	\mathbf{ \cdots }
\end{matrix}\right]
$$

+ Case 2，在 loopOutput1 中传入 _H0，输出张量 1（loopOutput1）形状 (7,1,3,4,5)
+ 结果是跳步保存的，且第一个元素参与了更多次运算，原因与 case 1 类似
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                 4. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2.
            \end{matrix}\right]
            \left[\begin{matrix}
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2.
            \end{matrix}\right]
            \left[\begin{matrix}
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2. \\
                 2. & 2. & 2. & 2. & 2.
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right] \\
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                32. &  8. &  8. &  8. &  8. \\
                 8. &  8. &  8. &  8. &  8. \\
                 8. &  8. &  8. &  8. &  8. \\
                 8. &  8. &  8. &  8. &  8.
            \end{matrix}\right]
            \left[\begin{matrix}
                 8. &  8. &  8. &  8. &  8. \\
                 8. &  8. &  8. &  8. &  8. \\
                 8. &  8. &  8. &  8. &  8. \\
                 8. &  8. &  8. &  8. &  8.
            \end{matrix}\right]
            \left[\begin{matrix}
                 8. &  8. &  8. &  8. &  8. \\
                 8. &  8. &  8. &  8. &  8. \\
                 8. &  8. &  8. &  8. &  8. \\
                 8. &  8. &  8. &  8. &  8.
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right] \\
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                256. &  32. &  32. &  32. &  32. \\
                 32. &  32. &  32. &  32. &  32. \\
                 32. &  32. &  32. &  32. &  32. \\
                 32. &  32. &  32. &  32. &  32.
            \end{matrix}\right]
            \left[\begin{matrix}
                 32. &  32. &  32. &  32. &  32. \\
                 32. &  32. &  32. &  32. &  32. \\
                 32. &  32. &  32. &  32. &  32. \\
                 32. &  32. &  32. &  32. &  32.
            \end{matrix}\right]
            \left[\begin{matrix}
                 32. &  32. &  32. &  32. &  32. \\
                 32. &  32. &  32. &  32. &  32. \\
                 32. &  32. &  32. &  32. &  32. \\
                 32. &  32. &  32. &  32. &  32.
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right] \\
    \left[\begin{matrix}
        \left[\begin{matrix}
            \left[\begin{matrix}
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0.
            \end{matrix}\right]
            \left[\begin{matrix}
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0.
            \end{matrix}\right]
            \left[\begin{matrix}
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0. \\
                 0. &  0. &  0. &  0. &  0.
            \end{matrix}\right]
        \end{matrix}\right]
    \end{matrix}\right] \\
	\mathbf{ \cdots }
\end{matrix}\right]
$$

---

## iterator layer
+ 使用 Iterator 层作为Input tensor的切片，参与 RNN 中间计算。使用 iterator 层的第一个例子，在 C 维上每次正向抛出 1 层 (1,nH,nW)，见 Iterator.py，

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
\end{matrix}\right]
$$

+ 输出张量 0（loopOutput0）形状 (1,4,5)，循环最终的结果
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        7. & 7. & 7. & 7. & 7. \\
        7. & 7. & 7. & 7. & 7. \\
        7. & 7. & 7. & 7. & 7. \\
        7. & 7. & 7. & 7. & 7.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1（loopOutput1） 形状 (3,1,4,5)，在初始值 1 的基础上先加 1 再加 2 再加 3
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2.
        \end{matrix}\right]
    \end{matrix}\right]
    \left[\begin{matrix}
        \left[\begin{matrix}
            4. & 4. & 4. & 4. & 4. \\
            4. & 4. & 4. & 4. & 4. \\
            4. & 4. & 4. & 4. & 4. \\
            4. & 4. & 4. & 4. & 4.
        \end{matrix}\right]
    \end{matrix}\right]
    \left[\begin{matrix}
        \left[\begin{matrix}
            7. & 7. & 7. & 7. & 7. \\
            7. & 7. & 7. & 7. & 7. \\
            7. & 7. & 7. & 7. & 7. \\
            7. & 7. & 7. & 7. & 7.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 使用 REVERSE 模式（将 CONCATENATE 换成 REVERSE）输出张量 1（loopOutput1） 形状 (3,1,4,5)，相当于将 CONCATENATE 的结果在最高维上倒序
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            7. & 7. & 7. & 7. & 7. \\
            7. & 7. & 7. & 7. & 7. \\
            7. & 7. & 7. & 7. & 7. \\
            7. & 7. & 7. & 7. & 7.
        \end{matrix}\right]
    \end{matrix}\right]
    \left[\begin{matrix}
        \left[\begin{matrix}
            4. & 4. & 4. & 4. & 4. \\
            4. & 4. & 4. & 4. & 4. \\
            4. & 4. & 4. & 4. & 4. \\
            4. & 4. & 4. & 4. & 4.
        \end{matrix}\right]
    \end{matrix}\right]
    \left[\begin{matrix}
        \left[\begin{matrix}
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 注意，可用的输出类型
| tensorrt.LoopOutput 名 |                     说明                     |
| :--------------------: | :------------------------------------------: |
|       LAST_VALUE       |              仅保留最后一个输出              |
|      CONCATENATE       |  保留指定长度的中间输出（从第一次循环向后）  |
|        REVERSE         | 保留执行长度的中间输出（从最后一次循环向前） |

+ 使用 iterator 层的第二个例子，在 H 维上每次正向抛出 1 层 (nC,1,nW)，见 Iterator2.py

+ 输出张量 0（loopOutput0）形状 (1,3,5)，循环最终的结果
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         1. &  5. &  5. &  5. &  5. \\
         2. &  9. &  9. &  9. &  9. \\
        1.  & 13. & 13. & 13. & 13.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1（loopOutput1） 形状 (4,1,3,5)，在初始值 1 的基础上分别依次加 1 或者加 2 或者加 3
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             2. &  2. &  2. &  2. &  2. \\
             3. &  3. &  3. &  3. &  3. \\
             4. &  4. &  4. &  4. &  4.
        \end{matrix}\right]
    \end{matrix}\right] \\
    \left[\begin{matrix}
        \left[\begin{matrix}
             3. &  3. &  3. &  3. &  3. \\
             5. &  5. &  5. &  5. &  5. \\
             7. &  7. &  7. &  7. &  7.
        \end{matrix}\right]
    \end{matrix}\right] \\
    \left[\begin{matrix}
        \left[\begin{matrix}
             4. &  4. &  4. &  4. &  4. \\
             7. &  7. &  7. &  7. &  7. \\
            10. & 10. & 10. & 10. & 10.
        \end{matrix}\right]
    \end{matrix}\right] \\
    \left[\begin{matrix}
        \left[\begin{matrix}
             5. &  5. &  5. &  5. &  5. \\
             9. &  9. &  9. &  9. &  9. \\
            13. & 13. & 13. & 13. & 13.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 使用 iterator 的第三个例子，在 W 维上每次正向抛出 1 层 (nC,nH,1)，见 Iterator3.py

+ 输出张量 0（loopOutput0）形状 (1,3,4)，循环最终的结果
$$
\left[\begin{matrix}
    \left[\begin{matrix}
         6. &  6. &  6. &  6. \\
        11. & 11. & 11. & 11. \\
        16. & 16. & 16. & 16.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1（loopOutput1） 形状 (5,1,3,4)，在初始值 1 的基础上分别依次加 1 或者加 2 或者加 3
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
             2. &  2. &  2. &  2. \\
             3. &  3. &  3. &  3. \\
             4. &  4. &  4. &  4.
        \end{matrix}\right]
    \end{matrix}\right] \\
    \left[\begin{matrix}
        \left[\begin{matrix}
             3. &  3. &  3. &  3. \\
             5. &  5. &  5. &  5. \\
             7. &  7. &  7. &  7.
        \end{matrix}\right]
    \end{matrix}\right] \\
    \left[\begin{matrix}
        \left[\begin{matrix}
             4. &  4. &  4. &  4. \\
             7. &  7. &  7. &  7. \\
            10. & 10. & 10. & 10.
        \end{matrix}\right]
    \end{matrix}\right] \\
    \left[\begin{matrix}
        \left[\begin{matrix}
             6. &  6. &  6. &  6. \\
            11. & 11. & 11. & 11. \\
            16. & 16. & 16. & 16.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 使用 iterator 的第四个例子，在 C 维上每次反向抛出 1 层 (1,nH,nW)，见 Iterator4.py

+ 输出张量 0（loopOutput0）形状 (1,4,5)，循环最终的结果
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1. & 7. & 7. & 7. & 7. \\
        2. & 7. & 7. & 7. & 7. \\
        3. & 7. & 7. & 7. & 7. \\
        4. & 7. & 7. & 7. & 7.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1（loopOutput1） 形状 (3,1,4,5)，在初始值 1 的基础上先加 3 再加 2 再加 1，REVERSE 输出不再展示
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            4. & 4. & 4. & 4. & 4. \\
            4. & 4. & 4. & 4. & 4. \\
            4. & 4. & 4. & 4. & 4. \\
            4. & 4. & 4. & 4. & 4.
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
    \left[\begin{matrix}
        \left[\begin{matrix}
            7. & 7. & 7. & 7. & 7. \\
            7. & 7. & 7. & 7. & 7. \\
            7. & 7. & 7. & 7. & 7. \\
            7. & 7. & 7. & 7. & 7.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 使用 iterator 的第五个例子，在 C 维上每次反向抛出 1 层 (1,nH,nW)，修改 CONCATENATE 输出的连接维度， 见 Iterator5.py

+ 结果在次高维上进行连接
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2. \\
            3. & 2. & 2. & 2. & 2. \\
            4. & 2. & 2. & 2. & 2.
        \end{matrix}\right]
        \left[\begin{matrix}
            1. & 4. & 4. & 4. & 4. \\
            2. & 4. & 4. & 4. & 4. \\
            3. & 4. & 4. & 4. & 4. \\
            4. & 4. & 4. & 4. & 4.
        \end{matrix}\right]
        \left[\begin{matrix}
            1. & 7. & 7. & 7. & 7. \\
            2. & 7. & 7. & 7. & 7. \\
            3. & 7. & 7. & 7. & 7. \\
            4. & 7. & 7. & 7. & 7.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1（loopOutput1） 形状 (1,3,4,5)，结果为输出张量 0 的倒序
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            7. & 7. & 7. & 7. & 7. \\
            7. & 7. & 7. & 7. & 7. \\
            7. & 7. & 7. & 7. & 7. \\
            7. & 7. & 7. & 7. & 7.
        \end{matrix}\right]
        \left[\begin{matrix}
            4. & 4. & 4. & 4. & 4. \\
            4. & 4. & 4. & 4. & 4. \\
            4. & 4. & 4. & 4. & 4. \\
            4. & 4. & 4. & 4. & 4.
        \end{matrix}\right]
        \left[\begin{matrix}
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2. \\
            2. & 2. & 2. & 2. & 2.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

---

## Simple ReLU RNN
+ Refer to ReLURNN.py，Structure of the network和输入输出数据与“RNNv2 层”保持一致，只是去掉了最高的一维

+ Shape of input tensor 0: (3,4,7)，3 个独立输入，每个输入 4 个单词，每个单词 7 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        2. & 1. & 1. & 1. & 1. & 1. & 1. \\
        3. & 1. & 1. & 1. & 1. & 1. & 1. \\
        4. & 1. & 1. & 1. & 1. & 1. & 1.
    \end{matrix}\right] \\
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        2. & 1. & 1. & 1. & 1. & 1. & 1. \\
        3. & 1. & 1. & 1. & 1. & 1. & 1. \\
        4. & 1. & 1. & 1. & 1. & 1. & 1.
    \end{matrix}\right] \\
    \left[\begin{matrix}
        1. & 1. & 1. & 1. & 1. & 1. & 1. \\
        2. & 1. & 1. & 1. & 1. & 1. & 1. \\
        3. & 1. & 1. & 1. & 1. & 1. & 1. \\
        4. & 1. & 1. & 1. & 1. & 1. & 1.
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output  0  tensor: (3,5)，3 个独立输出，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    1717. & 1717. & 1717. & 1717. & 1717. \\
    1717. & 1717. & 1717. & 1717. & 1717. \\
    1717. & 1717. & 1717. & 1717. & 1717.
\end{matrix}\right]
$$

+ Shape of output  1  tensor: (3,4,5)，3 个独立输出，每个包含 4 个隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        {\color{#007F00}{12.}} & {\color{#007F00}{12.}} & {\color{#007F00}{12.}} & {\color{#007F00}{12.}} & {\color{#007F00}{12.}} \\
        {\color{#0000FF}{67.}} & {\color{#0000FF}{12.}} & {\color{#0000FF}{12.}} & {\color{#0000FF}{12.}} & {\color{#0000FF}{12.}} \\
         342. &  342. &  342. &  342. &  342. \\
        1717. & 1717. & 1717. & 1717. & 1717.
    \end{matrix}\right] \\
    \left[\begin{matrix}
          12. &   12. &   12. &   12. &   12. \\
          67. &   67. &   67. &   67. &   67. \\
         342. &  342. &  342. &  342. &  342. \\
        1717. & 1717. & 1717. & 1717. & 1717.
    \end{matrix}\right] \\
    \left[\begin{matrix}
          12. &   12. &   12. &   12. &   12. \\
          67. &   67. &   67. &   67. &   67. \\
         342. &  342. &  342. &  342. &  342. \\
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
   1 \\ 1 \\ 1 \\ 1 \\ 1 \\ 1 \\ 1
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
&=\textbf{ReLU}\left(\left(12,12,12,12,12\right)^\mathrm{T}\right)\\
&=\left(
    {\color{#007F00}{12}},
    {\color{#007F00}{12}},
    {\color{#007F00}{12}},
    {\color{#007F00}{12}},
    {\color{#007F00}{12}}
  \right)^\mathrm{T}
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
   12 \\ 12 \\ 12 \\ 12 \\ 12 \\ 12 \\ 12
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
&=\textbf{ReLU}\left(\left(67,67,67,67,67\right)^\mathrm{T}\right)\\
&=\left(
    {\color{#0000FF}{67}},
    {\color{#0000FF}{67}},
    {\color{#0000FF}{67}},
    {\color{#0000FF}{67}},
    {\color{#0000FF}{67}}
  \right)^\mathrm{T}
\end{aligned}
$$

---

## Static unidirectional LSTM
+ Refer to StaticUnidirectionalLSTM.py，Structure of the network和输入输出数据与“RNNv2 层”保持一致，只是去掉了最高的一维

+ 采用单输入网络，初始隐藏状态（$h_{0}$）和初始细胞状态（$c_{0}$）被写死成常量，需要改成变量的情况可以参考后面“Dynamic unidirectional LSTM ”部分

+ Shape of input tensor 0: (3,4,7)，与“简单 ReLU RNN”的输入相同

+ Shape of output  0  tensor: (3,5)，为最终隐藏状态，3 个独立输出，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    0.99932283 & 0.99932283 & 0.99932283 & 0.99932283 & 0.99932283 \\
    0.99932283 & 0.99932283 & 0.99932283 & 0.99932283 & 0.99932283 \\
    0.99932283 & 0.99932283 & 0.99932283 & 0.99932283 & 0.99932283
\end{matrix}\right]
$$

+ Shape of output  1  tensor: (3,4,5)，为所有隐藏状态，3 个独立输出，每个包含 4 个隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        {\color{#007F00}{0.76158684}} & {\color{#007F00}{0.76158684}} & {\color{#007F00}{0.76158684}} & {\color{#007F00}{0.76158684}} & {\color{#007F00}{0.76158684}} \\
        {\color{#0000FF}{0.96400476}} & {\color{#0000FF}{0.96400476}} & {\color{#0000FF}{0.96400476}} & {\color{#0000FF}{0.96400476}} & {\color{#0000FF}{0.96400476}} \\
        0.99504673 & 0.99504673 & 0.99504673 & 0.99504673 & 0.99504673 \\
        0.99932283 & 0.99932283 & 0.99932283 & 0.99932283 & 0.99932283
    \end{matrix}\right] \\
    \left[\begin{matrix}
        0.76158684 & 0.76158684 & 0.76158684 & 0.76158684 & 0.76158684 \\
        0.96400476 & 0.96400476 & 0.96400476 & 0.96400476 & 0.96400476 \\
        0.99504673 & 0.99504673 & 0.99504673 & 0.99504673 & 0.99504673 \\
        0.99932283 & 0.99932283 & 0.99932283 & 0.99932283 & 0.99932283
    \end{matrix}\right] \\
    \left[\begin{matrix}
        0.76158684 & 0.76158684 & 0.76158684 & 0.76158684 & 0.76158684 \\
        0.96400476 & 0.96400476 & 0.96400476 & 0.96400476 & 0.96400476 \\
        0.99504673 & 0.99504673 & 0.99504673 & 0.99504673 & 0.99504673 \\
        0.99932283 & 0.99932283 & 0.99932283 & 0.99932283 & 0.99932283
    \end{matrix}\right]
\end{matrix}\right]
$$

+ Shape of output  2  tensor: (3,5)，为最终细胞状态，3 个独立输出，每个细胞状态 5 维坐标
$$
\left[\begin{matrix}
    3.999906 & 3.999906 & 3.999906 & 3.999906 & 3.999906 \\
    3.999906 & 3.999906 & 3.999906 & 3.999906 & 3.999906 \\
    3.999906 & 3.999906 & 3.999906 & 3.999906 & 3.999906
\end{matrix}\right]
$$

+ 计算过程：这里只用了一个 bias，$b_{*} = b_{*,X} + b_{?,H}$
$$
\begin{aligned}
I_{1} = F_{1} = O_{1} = \textbf{sigmoid} \left( W_{*,X} \cdot x_{1} + W_{*,H} \cdot h_{0} + b_{*} \right) &=
    \left( 0.99999386,0.99999386,0.99999386,0.99999386,0.99999386 \right) ^\mathrm{T} \\
C_{1}=\textbf{tanh} \left( W_{C,X}\cdot x_{1}+W_{C,H}\cdot h_{0}+b_{C} \right) &=
    \left( 0.99999999,0.99999999,0.99999999,0.99999999,0.99999999 \right) ^\mathrm{T} \\
c_{1} = F_{1} \cdot c_{0} + I_{1} \cdot C_{1} &=
    \left( 0.99999386,0.99999386,0.99999386,0.99999386,0.99999386 \right) ^\mathrm{T} \\
h_{1} = O_{1} \cdot \textbf{tanh} \left( c_{1} \right) &=
    \left(
        {\color{#007F00}{0.76158690}},
        {\color{#007F00}{0.76158690}},
        {\color{#007F00}{0.76158690}},
        {\color{#007F00}{0.76158690}},
        {\color{#007F00}{0.76158690}}
    \right) ^\mathrm{T} \\
\hfill \\
I_{2} = F_{2} = O_{2} = \textbf{sigmoid} \left( W_{*,X} \cdot x_{2} + W_{*,H} \cdot h_{1} + b_{*} \right) &=
    \left( 0.99997976,0.99997976,0.99997976,0.99997976,0.99997976 \right) ^\mathrm{T} \\
C_{2} = \textbf{tanh} \left( W_{C,X} \cdot x_{2} + W_{C,H} \cdot h_{1} + b_{C} \right) &=
    \left( 0.99999999,0.99999999,0.99999999,0.99999999,0.99999999 \right) ^\mathrm{T} \\
c_{2} = F_{2} \cdot c_{1} + I_{2} \cdot C_{2} &=
    \left( 1.99995338,1.99995338,1.99995338,1.99995338,1.99995338 \right) ^\mathrm{T} \\
h_{2} = O_{2} \cdot \textbf{tanh} \left( c_{2} \right) &=
    \left(
        {\color{#0000FF}{0.96400477}},
        {\color{#0000FF}{0.96400477}},
        {\color{#0000FF}{0.96400477}},
        {\color{#0000FF}{0.96400477}},
        {\color{#0000FF}{0.96400477}}
    \right) ^\mathrm{T} \\
\end{aligned}
$$

---

## Static bidirectional LSTM [TODO]
+ Refer to StaticBidirectionalLSTM.py，Structure of the network和输入输出数据与“RNNv2 层”保持一致，只是去掉了最高的一维

+ 思路是使用两个迭代器，一个正抛一个反抛，最后把计算结果 concatenate 在一起

---

## Dynamic unidirectional LSTM 
+ Refer to DynamicUnidirectionalLSTM.py，Structure of the network和输入输出数据与“RNNv2 层”保持一致，只是去掉了最高的一维

+ 采用三输入网络，输入数据（$x$）、初始隐藏状态（$h_{0}$）和初始细胞状态（$c_{0}$）均独立输入

+ 输入输出张量形状和结果与“Static unidirectional LSTM”完全相同