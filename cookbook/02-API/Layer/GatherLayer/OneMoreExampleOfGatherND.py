from cuda import cudart
import numpy as np
import os
import tensorrt as trt
    
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

inputT0 = network.add_input("inputT0", trt.float32, [-1, -1, -1, -1])
profile.set_shape(inputT0.name, [1, 3, 4, 5], [3, 3, 4, 5], [6, 6, 8, 10])
indexT0 = network.add_input("indexT0", trt.int32, [-1, 1, 4])
profile.set_shape(indexT0.name, [1, 1, 4], [3, 1, 4], [6, 1, 4])
indexT1 = network.add_input("indexT1", trt.int32, [-1, 1, 4])
profile.set_shape(indexT1.name, [1, 1, 4], [3, 1, 4], [6, 1, 4])
indexT2 = network.add_input("indexT2", trt.int32, [-1, 1, 4])
profile.set_shape(indexT2.name, [1, 1, 4], [3, 1, 4], [6, 1, 4])
config.add_optimization_profile(profile)

indexL = network.add_concatenation([indexT0, indexT1, indexT2])
indexL.axis = 1

indexTL = network.add_shuffle(indexL.get_output(0))
indexTL.first_transpose = [0, 2, 1]

outputL = network.add_gather(inputT0, indexTL.get_output(0), 1)
outputL.mode = trt.GatherMode.ND
outputL.num_elementwise_dims = 1

network.mark_output(outputL.get_output(0))

engineString = builder.build_serialized_network(network, config)
if engineString == None:
    print("Failed building serialized engine!")
    return
print("Succeeded building serialized engine!")
with open(trtFile, "wb") as f:
    f.write(engineString)
    print("Succeeded saving .plan file!")

engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
if engine == None:
    print("Failed building engine!")
    return
print("Succeeded building engine!")

context = engine.create_execution_context()
context.set_binding_shape(0, [3, 3, 4, 5])
context.set_binding_shape(1, [3, 1, 4])
context.set_binding_shape(2, [3, 1, 4])
context.set_binding_shape(3, [3, 1, 4])
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput
for i in range(nInput):
    print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
for i in range(nInput, nInput + nOutput):
    print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

bufferH = []
bufferH.append(np.ascontiguousarray(np.arange(3 * 3 * 4 * 5, dtype=np.float32).reshape(3, 3, 4, 5)))
bufferH.append(np.ascontiguousarray(np.zeros([3, 1, 4], dtype=np.int32) + 0))
bufferH.append(np.ascontiguousarray(np.zeros([3, 1, 4], dtype=np.int32) + 1))
bufferH.append(np.ascontiguousarray(np.zeros([3, 1, 4], dtype=np.int32) + 2))
for i in range(nInput, nInput + nOutput):
    bufferH.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))
bufferD = []
for i in range(nInput + nOutput):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

context.execute_v2(bufferD)

for i in range(nInput, nInput + nOutput):
    cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

for i in range(nInput + nOutput):
    print(engine.get_binding_name(i))
    print(bufferH[i])

for b in bufferD:
    cudart.cudaFree(b)

