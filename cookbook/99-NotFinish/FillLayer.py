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
# Case 0: Simple example of using NonZeroLayer
#   result: Succeeded
if False:
    nonZeroLayer = network.add_non_zero(inputT0)
    network.mark_output(nonZeroLayer.get_output(0))

# Case 1 [customer's original need]: use the shape of NonZeroLayer's output as the shape of following FillLayer
#   result: Segmentation fault (core dumped)
if True:
    nonZeroLayer = network.add_non_zero(inputT0)

    shapeLayer = network.add_shape(nonZeroLayer.get_output(0))

    constantLayer0 = network.add_constant([], trt.Weights(data1))
    constantLayer1 = network.add_constant([2], trt.Weights(data2))

    fillLayer = network.add_fill([1], trt.FillOperation.LINSPACE)
    fillLayer.set_input(0, shapeLayer.get_output(0))
    fillLayer.set_input(1, constantLayer0.get_output(0))
    fillLayer.set_input(2, constantLayer1.get_output(0))

    network.mark_output(fillLayer.get_output(0))

# Case 2: just use ShapeLayer to get the shape of NonZeroLayer
#   result: [TRT] [E] 2: [graphShapeAnalyzer.cpp::getSizeTensorLatticeValue::385] Error Code 2: Internal Error (Assertion x->tensor->producer failed. )
if False:
    nonZeroLayer = network.add_non_zero(inputT0)

    shapeLayer = network.add_shape(nonZeroLayer.get_output(0))

    network.mark_output(shapeLayer.get_output(0))

# Case 3: add other kind of layers following the NonZeroLayer, keeping data-dependent feature
#   result: Succeeded
#   comment: I tried IdentityLayer and also succeeded
if False:
    nonZeroLayer = network.add_non_zero(inputT0)

    reduceLayer = network.add_reduce(nonZeroLayer.get_output(0), trt.ReduceOperation.SUM, 1 << 0, False)  # shape of output tensor: [-1]
    reduceLayer.get_output(0).dtype = trt.int32

    network.mark_output(reduceLayer.get_output(0))

# Case 4: use output tensor's shape of the layers following the NonZeroLayer as the shape of a FillLayer, no keeping data-dependent feature
#   result: Segmentation fault (core dumped)
if False:
    nonZeroLayer = network.add_non_zero(inputT0)  # shape of output tensor: [3,-1]

    reduceLayer = network.add_reduce(nonZeroLayer.get_output(0), trt.ReduceOperation.SUM, 1 << 0, False)  # shape of output tensor: [-1]
    reduceLayer.get_output(0).dtype = trt.int32

    shapeLayer = network.add_shape(reduceLayer.get_output(0))

    constantLayer0 = network.add_constant([], trt.Weights(data1))
    constantLayer1 = network.add_constant([14], trt.Weights(data3))

    fillLayer = network.add_fill([1], trt.FillOperation.LINSPACE)
    fillLayer.set_input(0, shapeLayer.get_output(0))
    fillLayer.set_input(1, constantLayer0.get_output(0))
    fillLayer.set_input(2, constantLayer1.get_output(0))

    network.mark_output(fillLayer.get_output(0))
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
