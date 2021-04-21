import os
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

if __name__ == '__main__':
    logger = trt.Logger(trt.Logger.INFO)
    if os.path.isfile('./engine.trt'):
        with open('./engine.trt', 'rb') as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine( f.read() )
            if engine == None:
                exit()
    else:
        builder                     = trt.Builder(logger)
        network                     = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))  #
        profile                     = builder.create_optimization_profile()         #
        config                      = builder.create_builder_config()               #
        config.max_workspace_size   = 1 << 30                                       #
        config.flags                = 0                                             #

        inputT0     = network.add_input('inputT0', trt.DataType.FLOAT, (-1, 4, 5))        
        profile.set_shape(inputT0.name, (1,4,5),(3,4,5),(5,4,5))                    #
        config.add_optimization_profile(profile)

        identityLayer   = network.add_identity(inputT0)

        network.mark_output(identityLayer.get_output(0))
        engine = builder.build_engine(network, config)                              #
        if engine == None:
            exit() 
        with open('./engine.trt', 'wb') as f:
            f.write( engine.serialize() )
    
    context         = engine.create_execution_context()
    context.set_binding_shape(0,(3,4,5))
    print("Bind0->", engine.get_binding_shape(0), context.get_binding_shape(0))
    print("Bind1->", engine.get_binding_shape(1), context.get_binding_shape(1))

    stream          = cuda.Stream()

    data        = np.arange(3*4*5,dtype=np.float32).reshape(3,4,5)
    inputH0     = np.ascontiguousarray(data.reshape(-1))
    inputD0     = cuda.mem_alloc(inputH0.nbytes)
    outputH0    = np.empty(context.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))    #
    outputD0    = cuda.mem_alloc(outputH0.nbytes)
        
    cuda.memcpy_htod_async(inputD0, inputH0, stream)
    context.execute_async_v2([int(inputD0), int(outputD0)], stream.handle)          #
    cuda.memcpy_dtoh_async(outputH0, outputD0, stream)
    stream.synchronize()
    
    print("inputH0:", data.shape,engine.get_binding_dtype(0))
    print(data)
    print("outputH0:", outputH0.shape,engine.get_binding_dtype(1))
    print(outputH0)


