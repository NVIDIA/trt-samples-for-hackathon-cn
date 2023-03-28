import os
import ctypes
import numpy as np
from time import time_ns
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

soFilePath = "multinomial/RandomPlugin.so"
useFile = False
ipnutDataFile = "random_data.npz"

category_number = 192
npToTRT = {np.int8: trt.int8, np.float16: trt.float16, np.int32: trt.int32, np.float32: trt.float32}
npToPFT = {np.int8: trt.PluginFieldType.INT8, np.float16: trt.PluginFieldType.FLOAT16, np.int32: trt.PluginFieldType.INT32, np.float32: trt.PluginFieldType.FLOAT32}

def getRandomPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == "RandomPlugin":
            return c.create_plugin(c.name, trt.PluginFieldCollection([trt.PluginField("seed", np.int32(0), trt.PluginFieldType.INT32)]))
    return None

def buildEngine(logger, datatype):
    builder = trt.Builder(logger)
    network = builder.create_network(1 << 0)
    config = builder.create_builder_config()
    config.flags = [0, 1 << int(trt.BuilderFlag.FP16)][int(datatype == np.float16)]

    inputTensorList = []
    inputTensorList.append(network.add_input("inputT", npToTRT[datatype], [-1, -1]))

    profile = builder.create_optimization_profile()
    profile.set_shape("inputT", [1, category_number], [16, category_number], [64, category_number])

    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2(inputTensorList, getRandomPlugin())
    pluginLayer.get_output(0).dtype = trt.int32

    network.mark_output(pluginLayer.get_output(0))

    return builder.build_engine(network, config)

def run(datatype, nBatchSize):
    testCase = "test<bs=%d,fp%s>" % (nBatchSize, ["32", "16"][int(datatype == np.float16)])
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    trtFile = "engine-fp" + ["32", "16"][int(datatype == np.float16)] + ".plan"
    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            engineStr = f.read()
            engine = trt.Runtime(logger).deserialize_cuda_engine(engineStr)
        if engine == None:
            print("Failed loading engine!")
            return
        print("Succeeded loading engine!")
    else:
        engine = buildEngine(logger, datatype)
        if engine == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, "wb") as f:
            f.write(engine.serialize())

    context = engine.create_execution_context()
    context.set_binding_shape(0, [nBatchSize, category_number])

    print("Binding all? %s" % (["No", "Yes"][int(context.all_binding_shapes_specified)]))
    stream = cuda.Stream()

    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->", engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i))

    bufferH = []
    if useFile:
        io = np.load(ipnutDataFile)
        bufferH.append(io["input"][:nBatchSize])
    else:
        temp = np.random.randint(1, size=(nBatchSize, category_number)).astype(np.float32)
        for i in range(nBatchSize):
            for j in range(category_number):
                if j == 2 or j == 9 or j == 6:
                    temp[i][j] = 3
                else:
                    temp[i][j] = -1
        bufferH.append(temp)
        pass

    bufferH.append(np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1))))

    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cuda.mem_alloc(bufferH[i].nbytes))

    for i in range(nInput):
        cuda.memcpy_htod_async(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)), stream)

    context.execute_async_v2(bufferD, stream.handle)
    stream.synchronize()

    for i in range(nOutput):
        cuda.memcpy_dtoh_async(bufferH[nInput + i], bufferD[nInput + i], stream)
    stream.synchronize()

    for i in range(nInput):
        temp = bufferH[i]
        print("inputH%d" % i, temp.shape, np.sum(abs(temp)), np.var(temp), np.max(temp), np.min(temp), np.sum(np.abs(np.diff(temp.reshape(-1)))))

    print("check result:")
    temp1 = bufferH[-1]
    # temp2 = io["output"]
    # max = np.max(np.abs(np.abs(temp1 - temp2)))

    print("max is:", max)

if __name__ == "__main__":
    os.system("rm -f ./*.plan")
    np.set_printoptions(precision=3, linewidth=100, suppress=True)

    run(np.float32, 20)