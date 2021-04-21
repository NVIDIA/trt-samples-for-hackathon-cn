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
        builder.max_batch_size      = 1
        builder.max_workspace_size  = 3 << 30
        network                     = builder.create_network()

        inputTensor     = network.add_input('inputTensor', trt.DataType.FLOAT, (3, 4, 5))        
        identityLayer   = network.add_identity(inputTensor)

        network.mark_output(identityLayer.get_output(0))
        engine          = builder.build_cuda_engine(network)
        if engine == None:
            exit() 
        with open('./engine.trt', 'wb') as f:
            f.write( engine.serialize() )
    
    context         = engine.create_execution_context()
    stream          = cuda.Stream()

    data        = np.arange(3*4*5,dtype=np.float32).reshape(3,4,5)
    inputH0     = np.ascontiguousarray(data.reshape(-1))
    inputD0     = cuda.mem_alloc(inputH0.nbytes)
    outputH0    = np.empty(engine.get_binding_shape(1),dtype = trt.nptype(engine.get_binding_dtype(1)))
    outputD0    = cuda.mem_alloc(outputH0.nbytes)
        
    cuda.memcpy_htod_async(inputD0, inputH0, stream)
    context.execute_async(1, [int(inputD0), int(outputD0)], stream.handle)
    cuda.memcpy_dtoh_async(outputH0, outputD0, stream)
    stream.synchronize()
    
    print("data:", data.shape)
    print(data)
    print("outputH0:", outputH0.shape)
    print(outputH0)

