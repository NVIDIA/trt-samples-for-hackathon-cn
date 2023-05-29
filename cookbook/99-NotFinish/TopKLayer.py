import numpy as np
import tensorrt as trt
from cuda import cudart

data1 = np.float32(1000).reshape(-1)
data2 = np.array([10, 1], dtype=np.float32)
data3 = np.ones([14], dtype=np.float32)

shape = [4, 5, 6]
data = np.zeros(shape).astype(np.float32)
data[0, 0, 1] = 1
data[0, 2, 3] = 2
data[0, 3, 4] = 3
data[1, 1, 0] = 4
data[1, 1, 1] = 5
data[1, 1, 2] = 6
data[1, 1, 3] = 7
data[1, 1, 4] = 8
data[1, 1, 5] = 9
data[2, 0, 1] = 10
data[2, 1, 1] = 11
data[2, 2, 1] = 12
data[2, 3, 1] = 13
data[2, 4, 1] = 13

np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.int32, shape)
profile.set_shape_input(inputT0.name, shape, shape, shape)
config.add_optimization_profile(profile)
#-------------------------------------------------------------------------------
# Case 1 [customer's original need]: use TopKLayer with data-dependent input tensor
#   result: [TRT] [E] 3: (Unnamed Layer* 2) [TopK]: only activation types allowed as input to this layer.
#           [TRT] [E] 4: [graphShapeAnalyzer.cpp::needTypeAndDimensions::2212] Error Code 4: Internal Error ((Unnamed Layer* 2) [TopK]: output shape can not be computed)
#           [TRT] [E] 3: (Unnamed Layer* 2) [TopK]: only activation types allowed as input to this layer.
#           [TRT] [E] 4: [network.cpp::validate::3005] Error Code 4: Internal Error (Layer (Unnamed Layer* 2) [TopK] failed validation)
if True:
    nonZeroLayer = network.add_non_zero(inputT0)

    identityLayer = network.add_identity(nonZeroLayer.get_output(0))
    identityLayer.get_output(0).dtype = trt.float32

    topKLayer = network.add_topk(identityLayer.get_output(0), trt.TopKOperation.MAX, 1, 1 << 0)
    #topKLayer.get_output(0).dtype = trt.float32
    #topKLayer.get_output(1).dtype = trt.int32
    
    #network.mark_output(topKLayer.get_output(0))
    network.mark_output(topKLayer.get_output(1))
#-------------------------------------------------------------------------------

"""
# Print the structure of the network
for i in range(network.num_layers):
    layer = network.get_layer(i)
    print("%4d->%s,in=%d,out=%d,%s" % (i, str(layer.type)[10:], layer.num_inputs, layer.num_outputs, layer.name))
    for j in range(layer.num_inputs):
        tensor = layer.get_input(j)
        print("\tInput  %2d:%s,%s,%s" % (j, tensor.shape, str(tensor.dtype)[9:], tensor.name))
    for j in range(layer.num_outputs):
        tensor = layer.get_output(j)
        print("\tOutput %2d:%s,%s,%s" % (j, tensor.shape, str(tensor.dtype)[9:], tensor.name))
"""

engineString = builder.build_serialized_network(network, config)
if engineString is None:
    print("Failed building serialized network")
    exit()
print("Succeeded building serialized network")

# Inference part
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()
print("Before inference")
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

bufferH = []
bufferH.append(data)
bufferH.append(np.empty([len(shape) * np.prod(shape)], dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[1]))))

bufferD = []
for i in range(nIO):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

for i in range(nIO):
    context.set_tensor_address(lTensorName[i], int(bufferD[i]))

context.execute_async_v3(0)

print("After inference")
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

for i in range(nInput, nIO):
    cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

shapeReal = context.get_tensor_shape(lTensorName[1])
bufferH[1] = bufferH[1][:np.prod(shapeReal)].reshape(shapeReal)

for i in range(nIO):
    print(lTensorName[i])
    print(bufferH[i])

for b in bufferD:
    cudart.cudaFree(b)

