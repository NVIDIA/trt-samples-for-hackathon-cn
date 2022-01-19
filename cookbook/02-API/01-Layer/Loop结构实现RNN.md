# Loop 结构实现 RNN
+ 简单 ReLU RNN
+ 单层单向 LSTM
+ 单层双向 LSTM [TODO]

---
### 简单 ReLU RNN
+ 网络结构和输入输出数据与“RNN 层”保持一致
```python
import numpy as np
from cuda import cuda
import tensorrt as trt

nIn,cIn,hIn,wIn = 1,3,4,7                                                                           # 输入张量 NCHW
lenH            = 5                                                                                 # 隐藏层宽度
data    = np.ones([nIn,cIn,hIn,wIn],dtype=np.float32)                                               # 输入数据
weightX = np.ones((lenH,wIn),dtype=np.float32)                                                      # 权重矩阵 (X->H)
weightH = np.ones((lenH,lenH),dtype=np.float32)                                                     # 权重矩阵 (H->H)
biasX   = np.zeros(lenH, dtype=np.float32)                                                          # 偏置 (X->H)
biasH   = np.zeros(lenH, dtype=np.float32)                                                          # 偏置 (H->H)

np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
cuda.cuInit(0)
cuda.cuDeviceGet(0)

logger  = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config  = builder.create_builder_config()
config.max_workspace_size = 1 << 30
inputT0 = network.add_input('inputT0', trt.DataType.FLOAT, (nIn,cIn,hIn,wIn))                       # 单输入示例代码
#---------------------------------------------------------------------------------------------------# 替换部分
'''
hidden0Layer = network.add_constant([cIn,1,lenH], np.ones([cIn,1,lenH],dtype=np.float32))           # 初始隐藏状态
rnnV2Layer = network.add_rnn_v2(inputT0, 1, lenH, hIn, trt.RNNOperation.RELU)                       # 使用RNNV2 层的实现，1 层 ReLU 型 RNN，隐藏层元素宽 lenH，序列长度 hIn，单词编码宽度 wIn，batchSize 为 cIn
rnnV2Layer.hidden_state = hidden0Layer.get_output(0)
rnnV2Layer.set_weights_for_gate(0, trt.RNNGateType.INPUT, True,  weightX)
rnnV2Layer.set_weights_for_gate(0, trt.RNNGateType.INPUT, False, weightH)
rnnV2Layer.set_bias_for_gate(0, trt.RNNGateType.INPUT, True,  biasX)
rnnV2Layer.set_bias_for_gate(0, trt.RNNGateType.INPUT, False, biasH)
outputT0 = rnnV2Layer.get_output(0)                                                                 # outputT0 位所有状态，outputT1 为末状态
outputT1 = rnnV2Layer.get_output(1)
'''
weightXLayer    = network.add_constant([1, wIn, lenH], weightX.transpose().reshape(-1))
weightHLayer    = network.add_constant([1, lenH,lenH], weightH.transpose().reshape(-1))
biasLayer       = network.add_constant([1, cIn, lenH], np.tile(biasX+biasH,(cIn,1)))
hidden0Layer    = network.add_constant([1, cIn, lenH], np.ones(cIn*lenH,dtype=np.float32))          # 初始隐藏状态，注意形状和 RNNV2层的不一样
lengthLayer     = network.add_constant((),np.array([hIn],dtype=np.int32))                           # 结果保留长度

loop = network.add_loop()
iteratorLayer = loop.add_iterator(inputT0, 2, False)                                                # 每次抛出 inputTensor 的 H 维的一层 (1,cIn,wIn)

limit = network.add_constant((),np.array([hIn],dtype=np.int32))
loop.add_trip_limit(limit.get_output(0),trt.TripLimit.COUNT)

rLayer = loop.add_recurrence(hidden0Layer.get_output(0))
_H0 = network.add_matrix_multiply(iteratorLayer.get_output(0), trt.MatrixOperation.NONE, weightXLayer.get_output(0), trt.MatrixOperation.NONE)
_H1 = network.add_matrix_multiply(rLayer.get_output(0), trt.MatrixOperation.NONE, weightHLayer.get_output(0), trt.MatrixOperation.NONE)
_H2 = network.add_elementwise(_H0.get_output(0), _H1.get_output(0), trt.ElementWiseOperation.SUM)
_H3 = network.add_elementwise(_H2.get_output(0), biasLayer.get_output(0), trt.ElementWiseOperation.SUM)
_H4 = network.add_activation(_H3.get_output(0),trt.ActivationType.RELU)
rLayer.set_input(1, _H4.get_output(0))

loopOutput0 = loop.add_loop_output(rLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)              # 形状 (1,3,5)，3 个独立输出，每个输出 1 个最终隐藏状态，每个隐藏状态 5 维坐标
loopOutput1 = loop.add_loop_output(_H4.get_output(0), trt.LoopOutput.CONCATENATE, 1)                # 形状 (1,4,3,5)，3 个独立输出，每个输出 4 个隐藏状态，每个隐藏状态 5 维坐标
loopOutput1.set_input(1,lengthLayer.get_output(0))
print("loop->", loopOutput0.get_output(0).shape, loopOutput1.get_output(0).shape)                   # 打印原始输出长度：loop-> (1, 3, 5) (1, 4, 3, 5)


shuffleLayer0 = network.add_shuffle(loopOutput0.get_output(0))                                      # 调整形状为 (3,1,5)，与 RNNv2 层的结果相同
shuffleLayer0.first_transpose = (1,0,2)
shuffleLayer0.reshape_dims    = (1,cIn,1,lenH)
shuffleLayer1 = network.add_shuffle(loopOutput1.get_output(0))                                      # 调整形状为 (1,3,4,5)，与 RNNv2 层的结果相同
shuffleLayer1.first_transpose = (0,2,1,3)
shuffleLayer1.reshape_dims    = (1,cIn,hIn,lenH)

outputT0 = shuffleLayer1.get_output(0)                                                              # outputT0 为末状态，outputT1 为所有状态
outputT1 = shuffleLayer0.get_output(0)
#---------------------------------------------------------------------------------------------------# 替换部分
network.mark_output(outputT0)
network.mark_output(outputT1)
engineString    = builder.build_serialized_network(network,config)
engine          = trt.Runtime(logger).deserialize_cuda_engine(engineString)
context         = engine.create_execution_context()
_, stream       = cuda.cuStreamCreate(0)

inputH0     = np.ascontiguousarray(data.reshape(-1))
outputH0    = np.empty(context.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
outputH1    = np.empty(context.get_binding_shape(2),dtype = trt.nptype(engine.get_binding_dtype(2)))
_,inputD0   = cuda.cuMemAllocAsync(inputH0.nbytes,stream)
_,outputD0  = cuda.cuMemAllocAsync(outputH0.nbytes,stream)
_,outputD1  = cuda.cuMemAllocAsync(outputH1.nbytes,stream)

cuda.cuMemcpyHtoDAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, stream)
context.execute_async_v2([int(inputD0), int(outputD0), int(outputD1)], stream)
cuda.cuMemcpyDtoHAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, stream)
cuda.cuMemcpyDtoHAsync(outputH1.ctypes.data, outputD1, outputH1.nbytes, stream)
cuda.cuStreamSynchronize(stream)

print("inputH0 :", data.shape)
print(data)
print("outputH0:", outputH0.shape)
print(outputH0)
print("outputH1:", outputH1.shape)
print(outputH1)

cuda.cuStreamDestroy(stream)
cuda.cuMemFree(inputD0)
cuda.cuMemFree(outputD0)
cuda.cuMemFree(outputD1)
```

+ 输入张量形状 (1,3,4,7)，3 个独立输入，每个输入 4 个单词，每个单词 7 维坐标
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

+ 输出张量 0 形状 (1,3,1,5)，3 个独立输出，每个包含 1 个最终隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            1717. & 1717. & 1717. & 1717. & 1717.
        \end{matrix}\right] \\
        \left[\begin{matrix}
            1717. & 1717. & 1717. & 1717. & 1717.
        \end{matrix}\right] \\
        \left[\begin{matrix}
            1717. & 1717. & 1717. & 1717. & 1717.
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (1,3,4,5)，3 个独立输出，每个包含 4 个隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            \textcolor[rgb]{0,0.5,0}{12.} & \textcolor[rgb]{0,0.5,0}{12.} & \textcolor[rgb]{0,0.5,0}{12.} & \textcolor[rgb]{0,0.5,0}{12.} & \textcolor[rgb]{0,0.5,0}{12.} \\
            \textcolor[rgb]{0,0,1}{67.} & \textcolor[rgb]{0,0,1}{12.} & \textcolor[rgb]{0,0,1}{12.} & \textcolor[rgb]{0,0,1}{12.} & \textcolor[rgb]{0,0,1}{12.} \\
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
    \textcolor[rgb]{0,0.5,0}{12},
    \textcolor[rgb]{0,0.5,0}{12},
    \textcolor[rgb]{0,0.5,0}{12},
    \textcolor[rgb]{0,0.5,0}{12},
    \textcolor[rgb]{0,0.5,0}{12}
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
    \textcolor[rgb]{0,0,1}{67},
    \textcolor[rgb]{0,0,1}{67},
    \textcolor[rgb]{0,0,1}{67},
    \textcolor[rgb]{0,0,1}{67},
    \textcolor[rgb]{0,0,1}{67}
  \right)^\mathrm{T}
\end{aligned}
$$

---
### 单层单向 LSTM
```python
import numpy as np
from cuda import cuda
import tensorrt as trt

nIn,cIn,hIn,wIn = 1,3,4,7                                                                           # 输入张量 NCHW
lenH            = 5
data        = np.ones([nIn,cIn,hIn,wIn],dtype=np.float32)                                           # 输入数据
weightAllX  = np.ones((lenH,wIn),dtype=np.float32)                                                  # 权重矩阵 (X->H)
weightAllH  = np.ones((lenH,lenH),dtype=np.float32)                                                 # 权重矩阵 (H->H)
biasAllX    = np.zeros(lenH, dtype=np.float32)                                                      # 偏置 (X->H)
biasAllH    = np.zeros(lenH, dtype=np.float32)                                                      # 偏置 (H->H)

np.set_printoptions(precision = 8, linewidth = 200, suppress = True)
cuda.cuInit(0)
cuda.cuDeviceGet(0)

logger  = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config  = builder.create_builder_config()
config.max_workspace_size = 1 << 30
inputT0 = network.add_input('inputT0', trt.DataType.FLOAT, (nIn,cIn,hIn,wIn))
#---------------------------------------------------------------------------------------------------
'''
hidden0Layer = network.add_constant((cIn,1,lenH),np.ones((cIn,1,lenH),dtype=np.float32))            # 初始隐藏状态
cell0Layer = network.add_constant((cIn,1,lenH),np.zeros((cIn,1,lenH),dtype=np.float32))             # 初始细胞状态
rnnV2Layer = network.add_rnn_v2(inputT0, 1, lenH, hIn, trt.RNNOperation.LSTM)
rnnV2Layer.hidden_state = hidden0Layer.get_output(0)
rnnV2Layer.cell_state = cell0Layer.get_output(0)

for kind in [trt.RNNGateType.INPUT, trt.RNNGateType.CELL, trt.RNNGateType.FORGET, trt.RNNGateType.OUTPUT]:
    rnnV2Layer.set_weights_for_gate(0, kind, True,  weightAllX)
    rnnV2Layer.set_weights_for_gate(0, kind, False, weightAllH)
    rnnV2Layer.set_bias_for_gate(0, kind, True,  biasAllX)
    rnnV2Layer.set_bias_for_gate(0, kind, False, biasAllH)

outputT0 = rnnV2Layer.get_output(0)                                                                 # outputT0 为所有状态，outputT1 为末隐藏状态，outputT2 为末细胞状态
outputT1 = rnnV2Layer.get_output(1)
outputT2 = rnnV2Layer.get_output(2)
'''

def gate(network, x, wx, hiddenStateLayer, wh, b, isSigmoid):
    _H0 = network.add_matrix_multiply(x, trt.MatrixOperation.NONE, wx, trt.MatrixOperation.NONE)
    _H1 = network.add_matrix_multiply(hiddenStateLayer, trt.MatrixOperation.NONE, wh, trt.MatrixOperation.NONE)
    _H2 = network.add_elementwise(_H0.get_output(0), _H1.get_output(0), trt.ElementWiseOperation.SUM)
    _H3 = network.add_elementwise(_H2.get_output(0), b, trt.ElementWiseOperation.SUM)
    _H4 = network.add_activation(_H3.get_output(0), [trt.ActivationType.TANH,trt.ActivationType.SIGMOID][int(isSigmoid)])
    return _H4

weightAllXLayer = network.add_constant([1, wIn, lenH], np.ascontiguousarray(weightAllX.transpose().reshape(-1)))
weightAllHLayer = network.add_constant([1,lenH, lenH], np.ascontiguousarray(weightAllH.transpose().reshape(-1)))
biasAllLayer = network.add_constant([1, cIn, lenH], np.ascontiguousarray(np.tile(biasAllX+biasAllH,(cIn,1))))
hidden0Layer = network.add_constant([1, cIn, lenH], np.ones(cIn*lenH,dtype=np.float32))                      # 初始隐藏状态
cell0Layer = network.add_constant([1, cIn, lenH], np.zeros(cIn*lenH,dtype=np.float32))                      # 初始细胞状态
length = network.add_constant((),np.array([hIn],dtype=np.int32))

loop = network.add_loop()
limit = network.add_constant((),np.array([hIn],dtype=np.int32))
loop.add_trip_limit(limit.get_output(0),trt.TripLimit.COUNT)

iteratorLayer = loop.add_iterator(inputT0, 2, False)                                                # 每次抛出 inputTensor 的 H 维的一层 (1,cIn,wIn)，反向 LSTM 要多一个反抛的迭代器
x = network.add_identity(iteratorLayer.get_output(0))                                               # x 要被多次使用，不能直接用 iteratorLayer.get_output(0)
hiddenStateLayer = loop.add_recurrence(hidden0Layer.get_output(0))                                  # 一个 loop 中有多个循环变量
cellStateLayer = loop.add_recurrence(cell0Layer.get_output(0))

gateI = gate(network, x.get_output(0), weightAllXLayer.get_output(0), hiddenStateLayer.get_output(0), weightAllHLayer.get_output(0), biasAllLayer.get_output(0), True)
gateC = gate(network, x.get_output(0), weightAllXLayer.get_output(0), hiddenStateLayer.get_output(0), weightAllHLayer.get_output(0), biasAllLayer.get_output(0), False)
gateF = gate(network, x.get_output(0), weightAllXLayer.get_output(0), hiddenStateLayer.get_output(0), weightAllHLayer.get_output(0), biasAllLayer.get_output(0), True)
gateO = gate(network, x.get_output(0), weightAllXLayer.get_output(0), hiddenStateLayer.get_output(0), weightAllHLayer.get_output(0), biasAllLayer.get_output(0), True)

_H5                 = network.add_elementwise(gateF.get_output(0), cellStateLayer.get_output(0), trt.ElementWiseOperation.PROD)
_H6                 = network.add_elementwise(gateI.get_output(0), gateC.get_output(0), trt.ElementWiseOperation.PROD)
newCellStateLayer   = network.add_elementwise(_H5.get_output(0), _H6.get_output(0), trt.ElementWiseOperation.SUM)
_H7                 = network.add_activation(newCellStateLayer.get_output(0), trt.ActivationType.TANH)
newHiddenStateLayer = network.add_elementwise(gateO.get_output(0), _H7.get_output(0), trt.ElementWiseOperation.PROD)

hiddenStateLayer.set_input(1, newHiddenStateLayer.get_output(0))
cellStateLayer.set_input(1, newCellStateLayer.get_output(0))

loopOutput0 = loop.add_loop_output(hiddenStateLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)               # 形状 (1,3,5)，3 个独立输出，每个输出 1 个最终隐藏状态，每个隐藏状态 5 维坐标
loopOutput1 = loop.add_loop_output(newHiddenStateLayer.get_output(0), trt.LoopOutput.CONCATENATE, 1)              # 形状 (1,4,3,5)，3 个独立输出，每个输出 4 个隐藏状态，每个隐藏状态 5 维坐标
loopOutput1.set_input(1,length.get_output(0))
loopOutput2 = loop.add_loop_output(cellStateLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)               # 形状 (1,3,5)，3 个独立输出，每个输出 1 个最终隐藏状态，每个隐藏状态 5 维坐标
print("loop->", loopOutput0.get_output(0).shape, loopOutput1.get_output(0).shape)

shuffleLayer0 = network.add_shuffle(loopOutput0.get_output(0))                                      # 调整形状为 (3,1,5)，与 RNNv2 层的结果相同，可以不要
shuffleLayer0.first_transpose = (1,0,2)
shuffleLayer0.reshape_dims    = (1,cIn,1,lenH)
shuffleLayer1 = network.add_shuffle(loopOutput1.get_output(0))                                      # 调整形状为 (1,3,4,5)，与 RNNv2 层的结果相同，可以不要
shuffleLayer1.first_transpose = (0,2,1,3)
shuffleLayer1.reshape_dims    = (1,cIn,hIn,lenH)
shuffleLayer2 = network.add_shuffle(loopOutput2.get_output(0))                                      # 调整形状为 (3,1,5)，与 RNNv2 层的结果相同，可以不要
shuffleLayer2.first_transpose = (1,0,2)
shuffleLayer2.reshape_dims    = (1,cIn,1,lenH)

outputT0 = shuffleLayer0.get_output(0)
outputT1 = shuffleLayer1.get_output(0)
outputT2 = shuffleLayer2.get_output(0)
#---------------------------------------------------------------------------------------------------
network.mark_output(outputT0)
network.mark_output(outputT1)
network.mark_output(outputT2)
engineString    = builder.build_serialized_network(network,config)
engine          = trt.Runtime(logger).deserialize_cuda_engine(engineString)
context         = engine.create_execution_context()
_, stream       = cuda.cuStreamCreate(0)

inputH0     = np.ascontiguousarray(data.reshape(-1))
outputH0    = np.empty(context.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
outputH1    = np.empty(context.get_binding_shape(2),dtype = trt.nptype(engine.get_binding_dtype(2)))
outputH2    = np.empty(context.get_binding_shape(3),dtype = trt.nptype(engine.get_binding_dtype(3)))
_,inputD0   = cuda.cuMemAllocAsync(inputH0.nbytes,stream)
_,outputD0  = cuda.cuMemAllocAsync(outputH0.nbytes,stream)
_,outputD1  = cuda.cuMemAllocAsync(outputH1.nbytes,stream)
_,outputD2  = cuda.cuMemAllocAsync(outputH2.nbytes,stream)

cuda.cuMemcpyHtoDAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, stream)
context.execute_async_v2([int(inputD0), int(outputD0), int(outputD1), int(outputD2)], stream)
cuda.cuMemcpyDtoHAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, stream)
cuda.cuMemcpyDtoHAsync(outputH1.ctypes.data, outputD1, outputH1.nbytes, stream)
cuda.cuMemcpyDtoHAsync(outputH2.ctypes.data, outputD2, outputH2.nbytes, stream)
cuda.cuStreamSynchronize(stream)

print("inputH0 :", data.shape)
print(data)
print("outputH0:", outputH0.shape)
print(outputH0)
print("outputH1:", outputH1.shape)
print(outputH1)
print("outputH2:", outputH2.shape)
print(outputH2)

cuda.cuStreamDestroy(stream)
cuda.cuMemFree(inputD0)
cuda.cuMemFree(outputD0)
cuda.cuMemFree(outputD1)
cuda.cuMemFree(outputD2)
```

+ 输入张量形状 (1,3,4,7)，与”简单 ReLU RNN”的输入相同

+ 输出张量 0 形状 (1,3,1,5)，3 个独立输出，每个包含 1 个最终隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            0.99932283 & 0.99932283 & 0.99932283 & 0.99932283 & 0.99932283
        \end{matrix}\right] \\
        \left[\begin{matrix}
            0.99932283 & 0.99932283 & 0.99932283 & 0.99932283 & 0.99932283
        \end{matrix}\right] \\
        \left[\begin{matrix}
            0.99932283 & 0.99932283 & 0.99932283 & 0.99932283 & 0.99932283
        \end{matrix}\right]
    \end{matrix}\right]
\end{matrix}\right]
$$

+ 输出张量 1 形状 (1,3,4,5)，3 个独立输出，每个包含 4 个隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            \textcolor[rgb]{0,0.5,0}{0.76158684} & \textcolor[rgb]{0,0.5,0}{0.76158684} & \textcolor[rgb]{0,0.5,0}{0.76158684} & \textcolor[rgb]{0,0.5,0}{0.76158684} & \textcolor[rgb]{0,0.5,0}{0.76158684} \\
            \textcolor[rgb]{0,0,1}{0.96400476} & \textcolor[rgb]{0,0,1}{0.96400476} & \textcolor[rgb]{0,0,1}{0.96400476} & \textcolor[rgb]{0,0,1}{0.96400476} & \textcolor[rgb]{0,0,1}{0.96400476} \\
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
\end{matrix}\right]
$$

+ 输出张量 2 形状 (1,3,1,5)，3 个独立输出，每个包含 1 个最终隐藏状态，每个隐藏状态 5 维坐标
$$
\left[\begin{matrix}
    \left[\begin{matrix}
        \left[\begin{matrix}
            3.999906 & 3.999906 & 3.999906 & 3.999906 & 3.999906
        \end{matrix}\right] \\
        \left[\begin{matrix}
            3.999906 & 3.999906 & 3.999906 & 3.999906 & 3.999906
        \end{matrix}\right] \\
        \left[\begin{matrix}
            3.999906 & 3.999906 & 3.999906 & 3.999906 & 3.999906
        \end{matrix}\right]
    \end{matrix}\right]
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
        \textcolor[rgb]{0,0.5,0}{0.76158690},
        \textcolor[rgb]{0,0.5,0}{0.76158690},
        \textcolor[rgb]{0,0.5,0}{0.76158690},
        \textcolor[rgb]{0,0.5,0}{0.76158690},
        \textcolor[rgb]{0,0.5,0}{0.76158690}
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
        \textcolor[rgb]{0,0,1}{0.96400477},
        \textcolor[rgb]{0,0,1}{0.96400477},
        \textcolor[rgb]{0,0,1}{0.96400477},
        \textcolor[rgb]{0,0,1}{0.96400477},
        \textcolor[rgb]{0,0,1}{0.96400477}
    \right) ^\mathrm{T} \\
\end{aligned}
$$

### 单层双向 LSTM [TODO]
+ 思路是使用两个迭代器，一个正抛一个反抛，最后把计算结果 concatenate 在一起

