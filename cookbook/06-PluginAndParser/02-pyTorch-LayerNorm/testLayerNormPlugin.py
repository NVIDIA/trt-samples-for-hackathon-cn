import os
import ctypes
import numpy as np
from time import time_ns
import tensorrt as trt
from cuda import cuda

useFile         = True
ipnutDataFile   = 'Decoder/decoderIO.npz'
soFilePath      = 'LayerNormPlugin/LayerNorm.so'
staticSL        = 1
nTime           = 30
epsilon         = 1e-6
embeddingSize = 640

np.random.seed(97)

npToTRT = {np.int8:trt.int8,np.float16:trt.float16,np.int32:trt.int32,np.float32:trt.float32}
npToPFT = {np.int8:trt.PluginFieldType.INT8,np.float16:trt.PluginFieldType.FLOAT16,
            np.int32:trt.PluginFieldType.INT32,np.float32:trt.PluginFieldType.FLOAT32}

def check(a, b, weak = False):
    if weak:
        return np.all( np.abs(a - b) < epsilon)
    else:
        return np.all( a == b )

def layerNormCPU(bufferH):
    _x,b,a = bufferH
    _0  = np.mean(_x,2)[:,:,np.newaxis]
    _1  = _x - _0
    _2  = _1 * _1
    _3  = np.mean(_2,2)[:,:,np.newaxis]
    _4  = np.array(1e-12,dtype=np.float32)
    _5  = _4.reshape(1,1,1)
    _6  = _3 + _5
    _7  = np.sqrt(_6)
    _8  = 1 / _7                # 1/sqrt(...)
    _9  = b
    _10 = _9.reshape(1,1,embeddingSize)
    _11 = _8 * _10              # b/sqrt(...)
    _12 = _0 * _11              # bμ/sqrt(...)
    _13 = a
    _14 = _13.reshape(1,1,embeddingSize)
    _15 = _14 - _12             # a-bμ/sqrt(...)
    _16 = _x * _11              # bx/sqrt(...)
    _17 = _15 + _16             # b(x-μ)/sqrt(...)+a
    _18 = _17.reshape(bufferH[0].shape[0],1,bufferH[0].shape[1],bufferH[0].shape[2])
    return _18

def testLayerNormCPU():
    print("test LayerNormCPU!")
    bufferH = []
    io = np.load(ipnutDataFile)
    bufferH.append( io['encoder1_inputs:0'] )
    bufferH.append( io['(Unnamed Layer* 9) [Constant]_output'] )
    bufferH.append( io['(Unnamed Layer* 13) [Constant]_output'] )

    temp1 = layerNormCPU(bufferH)
    print( 'outputCPU: %s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        str(temp1.shape),np.sum(abs(temp1)),np.var(temp1),np.max(temp1),np.min(temp1),np.sum(np.abs(np.diff(temp1.reshape(-1)))) ))
    #print(temp1)
    temp2 = io['seq2seq/encoder_1/layer_0/multi_head/conv1d/conv1d/ExpandDims:0']
    print( 'outputRef: %s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        str(temp2.shape),np.sum(abs(temp2)),np.var(temp2),np.max(temp2),np.min(temp2),np.sum(np.abs(np.diff(temp2.reshape(-1)))) ))
    #print(temp2)
    print("check result:")
    print(check( temp1, temp2, True ))
    #temp = temp1 - temp2
    #print("diff", temp.shape,np.sum(abs(temp)),np.var(temp),np.max(temp),np.min(temp),np.sum(np.abs(np.diff(temp.reshape(-1)))))
    #print(temp)
    print("test layerNormCPU finish!")

def getLayerNormPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == 'LayerNorm':
            return c.create_plugin(c.name, trt.PluginFieldCollection([
                trt.PluginField('prevScale', np.float32(1),    trt.PluginFieldType.FLOAT32),
                trt.PluginField('postScale', np.float32(1),    trt.PluginFieldType.FLOAT32)
            ]))
    return None

def buildEngine(logger,datatype):
    builder         = trt.Builder(logger)
    network         = builder.create_network(1<<0)
    config          = builder.create_builder_config()
    config.max_workspace_size = 6 << 30
    config.flags    = [0,1<<int(trt.BuilderFlag.FP16)][int(datatype == np.float16)]

    inputTensorList = []
    inputTensorList.append( network.add_input('inputT', npToTRT[datatype], [-1,-1,-1]) )
    inputTensorList.append( network.add_input('inputB', npToTRT[datatype], [-1]) )
    inputTensorList.append( network.add_input('inputA', npToTRT[datatype], [-1]) )

    profile = builder.create_optimization_profile()
    profile.set_shape('inputT',[1,staticSL,320],[16,staticSL,560],[64,staticSL,embeddingSize])
    profile.set_shape('inputB',[320],[560],[embeddingSize])
    profile.set_shape('inputA',[320],[560],[embeddingSize])

    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2(inputTensorList, getLayerNormPlugin())
    pluginLayer.get_output(0).dtype = [trt.float32,trt.float16][int(datatype == np.float16)]

    network.mark_output(pluginLayer.get_output(0))

    return builder.build_engine(network, config)

def run(datatype, nBatchSize):
    testCase = "test<bs=%d,fp%s>"%(nBatchSize,['32','16'][int(datatype == np.float16)])
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    trtFile = 'engine-fp' + ['32','16'][int(datatype == np.float16)] +'.trt'
    if os.path.isfile(trtFile):
        with open(trtFile, 'rb') as f:
            engineStr = f.read()
            engine = trt.Runtime(logger).deserialize_cuda_engine(engineStr)
        if engine == None:
            print("Failed loading engine!")
            return
        print("Succeeded loading engine!")
    else:
        engine = buildEngine(logger,datatype)
        if engine == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, 'wb') as f:
            f.write( engine.serialize() )

    context = engine.create_execution_context()
    context.set_binding_shape(0,[nBatchSize,staticSL,embeddingSize])
    context.set_binding_shape(1,[embeddingSize])
    context.set_binding_shape(2,[embeddingSize])

    print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    _, stream       = cuda.cuStreamCreate(0)

    nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->",engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i))

    bufferH = []
    if useFile:
        io = np.load(ipnutDataFile)
        # bufferH.append( io['encoder1_inputs:0'][:nBatchSize] )
        # bufferH.append( io['(Unnamed Layer* 9) [Constant]_output'] )
        # bufferH.append( io['(Unnamed Layer* 13) [Constant]_output'] )

        bufferH.append( io['seq2seq/decoder2/concat:0'])
        bufferH.append( io['(Unnamed Layer* 45) [Shuffle]_output']) 
        bufferH.append( io['(Unnamed Layer* 49) [Shuffle]_output'])

        # bufferH.append( io['Transpose__2414:0'] )
        # bufferH.append( np.ones(1280).astype(datatype) )
        # bufferH.append( np.zeros(1280).astype(datatype) )


    else:
        temp = np.tile(np.arange(560),(nBatchSize,staticSL,1))
        # temp[0][0][0] = 10
        temp = temp*0.001
        bufferH.append( temp.reshape(nBatchSize,staticSL,560).astype(datatype) * 0.1)
        bufferH.append( np.ones(560).astype(datatype) )
        bufferH.append( np.zeros(560).astype(datatype) )

        #bufferH.append( np.arange(nBatchSize*staticSL*560).reshape(nBatchSize,staticSL,560).astype(datatype) )
        #bufferH.append( np.random.rand(560).astype(datatype) )
        #bufferH.append( np.random.rand(560).astype(datatype) )        
        pass

    bufferH.append(np.empty(context.get_binding_shape(3),dtype=trt.nptype(engine.get_binding_dtype(3))))

    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append( cuda.mem_alloc(bufferH[i].nbytes) )

    for i in range(nInput):
        cuda.memcpy_htod_async(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)), stream)

    context.execute_async_v2(bufferD, stream.handle)
    stream.synchronize()

    for i in range(nOutput):
        cuda.memcpy_dtoh_async(bufferH[nInput+i], bufferD[nInput+i], stream)
    stream.synchronize()
    
    for i in range(nInput):
        temp = bufferH[i]
        print("inputH%d"%i, temp.shape,np.sum(abs(temp)),np.var(temp),np.max(temp),np.min(temp),np.sum(np.abs(np.diff(temp.reshape(-1)))))
        #print(temp)
    
    # for i in range(nOutput):
    #     temp = bufferH[nInput+i]
    #     print("outputH%d"%i, temp.shape,np.sum(abs(temp)),np.var(temp),np.max(temp),np.min(temp),np.sum(np.abs(np.diff(temp.reshape(-1)))))
    #     #print(temp)
    
    
    time0 = time_ns()
    for i in range(nTime):
        context.execute_async_v2(bufferD, stream.handle)
    stream.synchronize()
    time1 = time_ns()
    print(testCase+"average %fms per inference\n"%((time1-time0)/nTime/1000000))
    

    print("check result:")
    temp1 = bufferH[-1]
    temp2 = layerNormCPU(bufferH[:3])
    temp3 = io['seq2seq/decoder2/cif_concat/LayerNorm/batchnorm/add_1:0']
    max = np.max(np.abs(np.abs(temp1 - temp2)))

    f = open("outputfp16.txt", "w")
    for batch in temp1:
        for sequence in batch:
            for t in sequence:
                f.write(str(t) + "\t")

    print(check(temp1,temp2,True), "max diff=%f"%(np.max(np.abs(temp1 - temp2))) )
    #print(temp2)

if __name__ == '__main__':
    os.system("rm -f ./*.trt")
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)

    run(np.float32,2)

    print("test all finish!")

