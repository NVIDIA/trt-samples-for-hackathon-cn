import os
import ctypes
from glob import glob
from time import time
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

useDataFromFile = True
ipnutDataFile   = './output.npz'
parameterFile   = './para.npz'
soFilePath      = './Attention.so'
nTime           = 30
nHead           = 4#8or4
nDimPerHead     = 64#64or32
nHDim           = nHead * nDimPerHead
seq = 15
np.random.seed(97)

npToTRT = {np.int8:trt.int8,np.float16:trt.float16,np.int32:trt.int32,np.float32:trt.float32}
npToPFT = {np.int8:trt.PluginFieldType.INT8,np.float16:trt.PluginFieldType.FLOAT16,
            np.int32:trt.PluginFieldType.INT32,np.float32:trt.PluginFieldType.FLOAT32}

def getAttentionPlugin(useFP16):
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == 'Attention':
            # p0 = trt.PluginField("useFP16", np.array([int(useFP16)],dtype=np.int32), trt.PluginFieldType.INT32)
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None

def buildEngine(logger,datatype):
    builder         = trt.Builder(logger)
    network         = builder.create_network(1<<0)
    config          = builder.create_builder_config()
    config.max_workspace_size = 6 << 30
    config.flags    = [0,1<<int(trt.BuilderFlag.FP16)][int(datatype == np.float16)]

    inputTensorList = []
    inputTensorList.append( network.add_input('q_x',                npToTRT[datatype], [-1,-1,nHDim]) )
    inputTensorList.append( network.add_input('pos_emb',            npToTRT[datatype], [-1,-1,nHDim]) )
    inputTensorList.append( network.add_input('mask',               npToTRT[datatype], [-1, 1, -1]) )
    inputTensorList.append( network.add_input('pos_bias_u',         npToTRT[datatype], [nHead,nDimPerHead]) )
    inputTensorList.append( network.add_input('pos_bias_v',         npToTRT[datatype], [nHead,nDimPerHead]) )
    inputTensorList.append( network.add_input('linear_qkv_weight',    npToTRT[datatype], [3, nHDim,nHDim]) )
    inputTensorList.append( network.add_input('linear_q_bias',      npToTRT[datatype], [nHDim]) )
    inputTensorList.append( network.add_input('linear_k_bias',      npToTRT[datatype], [nHDim]) )
    inputTensorList.append( network.add_input('linear_v_bias',      npToTRT[datatype], [nHDim]) )
    inputTensorList.append( network.add_input('linear_out_weight',  npToTRT[datatype], [nHDim,nHDim]) )
    inputTensorList.append( network.add_input('linear_out_bias',    npToTRT[datatype], [nHDim]) )
    inputTensorList.append( network.add_input('linear_pos_weight',  npToTRT[datatype], [nHDim,nHDim]) )

    profile = builder.create_optimization_profile()
    profile.set_shape('q_x',    [1,seq,nHDim],[16,seq,nHDim],[32,seq,nHDim])
    profile.set_shape('pos_emb',[1,seq,nHDim],[1,seq,nHDim],[1,seq,nHDim])
    profile.set_shape('mask',   [1,1,seq],  [16,1,seq],  [32,1,seq])    
    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2(inputTensorList, getAttentionPlugin(int(datatype == np.float16)))
    pluginLayer.get_output(0).dtype = [trt.float32,trt.float16][int(datatype == np.float16)]

    network.mark_output(pluginLayer.get_output(0))

    return builder.build_engine(network, config)

def run(datatype,batchSize,sequenceLength):
    testCase = "test<bs=%d,sl=%d,fp%s>"%(batchSize,sequenceLength,['32','16'][int(datatype == np.float16)])
    #print(testCase+"start!")

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    trtFile = 'engine-fp' + ['32','16'][int(datatype == np.float16)] +'.trt'
    # if os.path.isfile(trtFile):
    #     with open(trtFile, 'rb') as f:
    #         engineStr = f.read()
    #         engine = trt.Runtime(logger).deserialize_cuda_engine(engineStr)
    #     if engine == None:
    #         print("Failed loading engine!")
    #         return
    #     print("Succeeded loading engine!")
    # else:
    engine = buildEngine(logger,datatype)
    if engine == None:
        print("Failed building engine!")
        return
    print("succeeded building engine!")
    with open(trtFile, 'wb') as f:
        f.write( engine.serialize() )

    context = engine.create_execution_context()
    context.set_binding_shape( 0,[batchSize,sequenceLength,nHDim])
    context.set_binding_shape( 1,[1,sequenceLength,nHDim])
    context.set_binding_shape( 2,[batchSize,1,sequenceLength])
    print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    stream  = cuda.Stream()

    nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->",engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i))


    bufferH = []
    data = np.load(ipnutDataFile)
    para = np.load(parameterFile)
    if useDataFromFile:    

        bufferH.append(np.ascontiguousarray( data['856'].astype(datatype).reshape(-1) ))
        bufferH.append(np.ascontiguousarray( data['603'].astype(datatype).reshape(-1) ))
        bufferH.append(np.ascontiguousarray( data['613'].astype(datatype).reshape(-1) ))
        
        bufferH.append(np.ascontiguousarray( para['encoder.encoders.1.self_attn.pos_bias_u'].astype(datatype).reshape(-1) ))
        bufferH.append(np.ascontiguousarray( para['encoder.encoders.1.self_attn.pos_bias_v'].astype(datatype).reshape(-1) ))

        q_weight = para['encoder.encoders.1.self_attn.linear_q.weight']
        k_weight = para['encoder.encoders.1.self_attn.linear_k.weight']
        v_weight = para['encoder.encoders.1.self_attn.linear_v.weight']

        qkv_weight = np.stack([q_weight, k_weight, v_weight])

        bufferH.append(np.ascontiguousarray( qkv_weight.astype(datatype).reshape(-1) ))
        
        bufferH.append(np.ascontiguousarray( para['encoder.encoders.1.self_attn.linear_q.bias'].astype(datatype).reshape(-1) ))
        # bufferH.append(np.ascontiguousarray( para['encoder.encoders.1.self_attn.linear_k.weight'].astype(datatype).reshape(-1) ))
        bufferH.append(np.ascontiguousarray( para['encoder.encoders.1.self_attn.linear_k.bias'].astype(datatype).reshape(-1) ))
        # bufferH.append(np.ascontiguousarray( para['encoder.encoders.1.self_attn.linear_v.weight'].astype(datatype).reshape(-1) ))
        bufferH.append(np.ascontiguousarray( para['encoder.encoders.1.self_attn.linear_v.bias'].astype(datatype).reshape(-1) ))
        bufferH.append(np.ascontiguousarray( para['encoder.encoders.1.self_attn.linear_out.weight'].astype(datatype).reshape(-1) ))
        bufferH.append(np.ascontiguousarray( para['encoder.encoders.1.self_attn.linear_out.bias'].astype(datatype).reshape(-1) ))
        bufferH.append(np.ascontiguousarray( para['encoder.encoders.1.self_attn.linear_pos.weight'].astype(datatype).reshape(-1) ))

        print("test")
    else:
        bufferH.append(np.ascontiguousarray( np.random.rand(batchSize,sequenceLength,nHDim).astype(datatype).reshape(-1)*2-1 ))
        bufferH.append(np.ascontiguousarray( np.random.rand(batchSize,sequenceLength,nHDim).astype(datatype).reshape(-1)*2-1 ))
        bufferH.append(np.ascontiguousarray( np.random.rand(batchSize,sequenceLength,nHDim).astype(datatype).reshape(-1)*2-1 ))
        bufferH.append(np.ascontiguousarray( np.random.rand(batchSize,sequenceLength,nHDim).astype(datatype).reshape(-1)*2-1 ))
        bufferH.append(np.ascontiguousarray( np.ones([batchSize,1,sequenceLength])))
        bufferH.append(np.ascontiguousarray( np.random.rand(nHDim).astype(datatype).reshape(-1)*2-1 ))
        bufferH.append(np.ascontiguousarray( np.random.rand(nHDim).astype(datatype).reshape(-1)*2-1 ))
        bufferH.append(np.ascontiguousarray( np.random.rand(nHDim*nHDim).astype(datatype).reshape(-1)*2-1 ))
        bufferH.append(np.ascontiguousarray( np.random.rand(nHDim).astype(datatype).reshape(-1)*2-1 ))
        bufferH.append(np.ascontiguousarray( np.random.rand(nHDim*nHDim).astype(datatype).reshape(-1)*2-1 ))
        bufferH.append(np.ascontiguousarray( np.random.rand(nHDim).astype(datatype).reshape(-1)*2-1 ))
        bufferH.append(np.ascontiguousarray( np.random.rand(nHDim*nHDim).astype(datatype).reshape(-1)*2-1 ))
        bufferH.append(np.ascontiguousarray( np.random.rand(nHDim).astype(datatype).reshape(-1)*2-1 ))
        bufferH.append(np.ascontiguousarray( np.random.rand(nHDim*nHDim).astype(datatype).reshape(-1)*2-1 ))
        bufferH.append(np.ascontiguousarray( np.random.rand(nHDim).astype(datatype).reshape(-1)*2-1 ))
        bufferH.append(np.ascontiguousarray( np.random.rand(nHDim*nHDim).astype(datatype).reshape(-1)*2-1 ))

    bufferH.append(np.empty(context.get_binding_shape(12),dtype=trt.nptype(engine.get_binding_dtype(12))))

    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append( cuda.mem_alloc(bufferH[i].nbytes) )

    for i in range(nInput):
        cuda.memcpy_htod_async(bufferD[i], bufferH[i], stream)

    context.execute_async_v2(bufferD, stream.handle)
    stream.synchronize()
    
    time0 = time()
    for i in range(nTime):
        context.execute_async_v2(bufferD, stream.handle)
    stream.synchronize()
    time1 = time()
    print(testCase+"average %fms per inference\n"%((time1-time0)/nTime*1000))
    
    for i in range(nOutput):
        cuda.memcpy_dtoh_async(bufferH[nInput+i], bufferD[nInput+i], stream)
    stream.synchronize()
    
    print("res=")
    #print(bufferH[-1])
    temp1 = bufferH[-1]

    data = np.load(ipnutDataFile)
    temp4 = data["964"]
    tt = np.max(np.abs(temp1-temp4))
    print("test", tt)
    
    '''
    print("ref=")
    print(data['att'])
    '''
    #print(testCase+"finish!")
    
if __name__ == '__main__':
    #os.system("rm -f ./*.trt")
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
    #cuda.Device(0).make_context()
    
    run(np.float32,16,seq)
    '''    
    run(np.float32,16,256)
    run(np.float32,16,512)
    run(np.float32,32,128)
    run(np.float32,32,256)
    run(np.float32,32,512)
    '''
    '''
    run(np.float16,16,128)
    run(np.float16,16,256)
    run(np.float16,16,512)
    run(np.float16,32,128)
    run(np.float16,32,256)
    run(np.float16,32,512)
    '''
    #cuda.Context.pop()
    #print("test all finish!")

