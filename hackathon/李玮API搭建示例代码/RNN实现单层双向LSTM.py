import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

hIn     = 4                                                                                         # 输入张量 HWC
wIn     = 7
cIn     = 3
lenH    = 5                                                                                         # 隐藏层元素宽度
data    = np.ones(cIn*hIn*wIn,dtype=np.float32).reshape(cIn,hIn,wIn)                                # 输入张量
weightF = np.ones((wIn+lenH,lenH*4),dtype=np.float32)                                               # 正向变换阵（TensorFlow 格式）
weightB = np.ones((wIn+lenH,lenH*4),dtype=np.float32)                                               # 反向变换阵（TensorFlow 格式）
biasF    = np.zeros(lenH*4, dtype=np.float32)                                                       # 正向偏置
biasB    = np.zeros(lenH*4, dtype=np.float32)                                                       # 反向偏置

def buildEngine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    builder.max_batch_size = 1
    builder.max_workspace_size = 3 << 30
    network = builder.create_network()
    inputTensor = network.add_input('inputTensor', trt.DataType.FLOAT, (cIn, hIn, wIn))
    print("inputTensor->", inputTensor.shape)
        
    #-----------------------------------------------------------------------------------------------# 可替换部分
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