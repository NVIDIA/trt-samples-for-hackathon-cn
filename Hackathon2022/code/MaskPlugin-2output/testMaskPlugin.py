import os
import ctypes
import numpy as np
#from time import time_ns
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

useFile         = False
ipnutDataFile   = '/workspace/data/encoder-16-64.npz'
soFilePath      = './Mask.so'
epsilon         = 1e-6
negInf          = -6e6

np.random.seed(97)

npToTRT = {np.int8:trt.int8,np.float16:trt.float16,np.int32:trt.int32,np.float32:trt.float32}
npToPFT = {np.int8:trt.PluginFieldType.INT8,np.float16:trt.PluginFieldType.FLOAT16,
            np.int32:trt.PluginFieldType.INT32,np.float32:trt.PluginFieldType.FLOAT32}

def check(a, b, weak = False):
    if weak:
        return np.all( np.abs(a - b) < epsilon)
    else:
        return np.all( a == b )

def maskCPU(bufferHList):
    input0 = bufferHList[0]
    input1 = bufferHList[1]    
    b = input0.shape[-0]
    t4 = input0.shape[1] // 4 - 1

    output0 = np.zeros([b,1,1,t4],dtype=np.float32) + negInf
    output1 = np.zeros([b,1,1,t4],dtype=np.float32)

    for i in range(b):
        output0[i,0,0,:input1[i]] = 0
        output1[i,0,0,:input1[i]] = 1

    return output0, output1

def getMaskPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == 'Mask':
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None

def run(nBS,nSL):
    testCase = "test<bs=%d,sl=%d>"%(nBS,nSL)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)
    builder         = trt.Builder(logger)
    network         = builder.create_network(1<<0)
    config          = builder.create_builder_config()
    config.max_workspace_size = 6 << 30
    config.flags    = 0 #1<<int(trt.BuilderFlag.FP16)

    inputTensorList = []
    inputTensorList.append( network.add_input('inputT0', trt.float32, [-1,-1,80]) )
    inputTensorList.append( network.add_input('inputT1', trt.int32, [-1]) )

    profile = builder.create_optimization_profile()
    profile.set_shape('inputT0',[1,16,80],[4,64,80],[16,256,80])
    profile.set_shape('inputT1',[1],[4],[16])
    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2(inputTensorList, getMaskPlugin())
    pluginLayer.get_output(0).dtype = trt.float32 #trt.float16

    network.mark_output(pluginLayer.get_output(0))
    network.mark_output(pluginLayer.get_output(1))
    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0,[nBS,nSL,80])
    context.set_binding_shape(1,[nBS])
    print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    stream  = cuda.Stream()

    nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->",engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i))

    bufferH = []
    bufferH.append( np.random.rand(nBS,nSL,80).astype(np.float32).reshape(nBS,nSL,80) )
    bufferH.append( np.arange(1,1+nBS).astype(np.int32) )
    bufferH.append(np.empty(context.get_binding_shape(2),dtype=trt.nptype(engine.get_binding_dtype(2))))
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
        print(temp)
    
    for i in range(nOutput):
        temp = bufferH[nInput+i]
        print("outputH%d"%i, temp.shape,np.sum(abs(temp)),np.var(temp),np.max(temp),np.min(temp),np.sum(np.abs(np.diff(temp.reshape(-1)))))
        print(temp)
    
    print("check result:")
    temp1 = bufferH[-1]
    temp2 = maskCPU(bufferH[:2])
    print(check(temp1,temp2,True), "max diff=%f"%(np.max(np.abs(temp1 - temp2))) )

if __name__ == '__main__':
    os.system("rm -f ./*.trt")
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
    #cuda.Device(0).make_context()

    run(16,16)

    #cuda.Context.pop()
    #print("test all finish!")

