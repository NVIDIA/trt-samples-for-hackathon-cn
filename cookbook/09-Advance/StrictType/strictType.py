import numpy as np
from cuda import cudart
import tensorrt as trt

np.random.seed(97)
m, k, n = 3, 4, 5  # 输入张量 NCHW
data0 = np.tile(np.arange(1,1+k),[m,1]) * 1/10**(2*np.arange(1,1+m)-2)[:,np.newaxis]
data1 = np.tile(np.arange(k),[n,1]).T * 10**np.arange(n)[np.newaxis,:]

def run(useFP16):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    if useFP16:
        config.flags = config.flags | (1<<int(trt.BuilderFlag.STRICT_TYPES)) | (1<<int(trt.BuilderFlag.FP16))

    inputT0 = network.add_input('inputT0', trt.DataType.FLOAT, (m, k))

    constantLayer = network.add_constant([k,n], np.ascontiguousarray(data1.astype(np.float16 if useFP16 else np.float32)))
    matrixMultiplyLayer = network.add_matrix_multiply(inputT0, trt.MatrixOperation.NONE, constantLayer.get_output(0), trt.MatrixOperation.NONE)
    if useFP16:
        matrixMultiplyLayer.precision = trt.DataType.HALF
        matrixMultiplyLayer.get_output(0).dtype = trt.DataType.HALF

    network.mark_output(matrixMultiplyLayer.get_output(0))
    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    context = engine.create_execution_context()
    _, stream = cudart.cudaStreamCreate()

    inputH0 = np.ascontiguousarray(data0.reshape(-1))
    outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)
    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    #print("inputH0 :", data0.shape, data0.dtype)
    #print(data0)
    print("outputH0:", outputH0.shape, outputH0.dtype)
    print(outputH0)

    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputD0)

if __name__ == '__main__':
    np.set_printoptions(precision=3, linewidth=200, suppress=True)
    cudart.cudaDeviceSynchronize()

    run(False)  # 使用 FP32
    run(True)  # 使用 FP16
    
