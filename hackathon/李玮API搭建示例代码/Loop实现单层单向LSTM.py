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