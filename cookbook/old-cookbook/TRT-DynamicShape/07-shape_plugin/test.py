import os
import ctypes
from glob import glob
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

soFilePath = './test.so'
addend     = 0.5                                                                                    # could only change before engine built

def getShapePlugin(addend):
    for c in trt.get_plugin_registry().plugin_creator_list:
        if c.name == 'ShapePlugin':
            p1 = trt.PluginField("addend", np.array([addend],dtype=np.float32), trt.PluginFieldType.FLOAT32)
            fc = trt.PluginFieldCollection([p1,])
            return c.create_plugin(name = 'ShapePlugin', field_collection=fc)
    return None

def buildEngine(logger, dataType):
    plugin = getShapePlugin(addend)
    if plugin == None:
        print('Plugin not found')
        return

    builder = trt.Builder(logger)
    builder.max_batch_size = 4
    builder.max_workspace_size = 3 << 30
    builder.fp16_mode = (dataType == 'float16')
    builder.strict_type_constraints = True
    network = builder.create_network(1<<0)

    inputTensor = network.add_input('input', trt.DataType.FLOAT, (-1,-1,-1,-1))

    pluginLayer = network.add_plugin_v2(inputs = [inputTensor,], plugin = plugin)

    network.mark_output(pluginLayer.get_output(0))
    network.get_output(0).dtype = trt.DataType.INT32
    network.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.LINEAR)                       # need shuffle if using CHW4 
    profile = builder.create_optimization_profile();
    profile.set_shape(inputTensor.name, (1,1,1,1),(1,3,4,5),(4,3,9,12));

    config = builder.create_builder_config();
    config.add_optimization_profile(profile);
    return builder.build_engine(network, config)

def run(dimIn, dataType):
    print("test->", dimIn, dataType);
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)            

    trtFile = "./engine-"+dataType+".trt"
    if os.path.isfile(trtFile):
        with open(trtFile, 'rb') as f:
            engineStr = f.read()
            engine = trt.Runtime(logger).deserialize_cuda_engine(engineStr)
        if engine == None:
            print("Failed loading engine!")
            return
        print("succeeded loading engine!")
    else:
        engine = buildEngine(logger, dataType)
        if engine == None:
            print("Failed building engine!")
            return None
        print("succeeded building engine!")
        with open(trtFile, 'wb') as f:
            f.write( engine.serialize() )

    context = engine.create_execution_context()
    context.set_binding_shape(0,dimIn)
    stream  = cuda.Stream()
    print("EngineBinding0->", engine.get_binding_shape(0));
    print("EngineBinding1->",engine.get_binding_shape(1));
    print("ContextBinding0->",context.get_binding_shape(0));
    print("ContextBinding1->",context.get_binding_shape(1));

    data      = np.ones(np.prod(dimIn), dtype=np.float32)
    input1_h  = np.ascontiguousarray(data)
    input1_d  = cuda.mem_alloc(input1_h.nbytes)
    output1_h = np.empty(context.get_binding_shape(1), dtype = trt.nptype(engine.get_binding_dtype(1)))
    output1_d = cuda.mem_alloc(output1_h.nbytes)
        
    cuda.memcpy_htod_async(input1_d, input1_h, stream)
    context.execute_async(1, [int(input1_d), int(output1_d)], stream.handle)
    cuda.memcpy_dtoh_async(output1_h, output1_d, stream)
    stream.synchronize()
    
    print(output1_h.shape, engine.get_binding_dtype(1))
    print(output1_h)
    print("test " + dataType + " finish!")

if __name__ == '__main__':
    #[ os.remove(item) for item in glob("./*.trt") + glob("./*.cache")]
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
    run((1,3,4,5), "float32")
    run((1,3,1,1), "float32")
    run((4,3,9,12),"float32")
    print("test finish!")

