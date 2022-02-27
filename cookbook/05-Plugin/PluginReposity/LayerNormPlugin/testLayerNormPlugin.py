import os
import ctypes
import numpy as np
from time import time_ns
import tensorrt as trt
from cuda import cudart

soFilePath = "./LayerNorm.so"
nTime = 30

nIn, cIn, hIn, wIn = 2, 3, 4, 5
npDataType = np.float32
globalEpsilon = 1e-5
np.random.seed(97)

def check(a, b, weak=False):
    if weak:
        return np.all(np.abs(a - b) < globalEpsilon)
    else:
        return np.all(a == b)

def layerNormCPU(bufferH, epsilon):
    _x = bufferH[0]
    _0 = np.mean(_x.reshape(_x.shape[0], -1), 1)[:, np.newaxis, np.newaxis, np.newaxis]
    _1 = _x - _0
    _2 = _1 * _1
    _3 = np.mean(_2.reshape(_x.shape[0], -1), 1)[:, np.newaxis, np.newaxis, np.newaxis]
    _4 = _3 + epsilon
    _5 = np.sqrt(_4)
    _6 = 1 / _5  # 1/sqrt(...)
    _7 = _1 * _6  # (x-μ)/sqrt(...)
    return [_7]

def getLayerNormPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == 'LayerNorm':
            p0 = trt.PluginField('epsilon', np.float32(globalEpsilon), trt.PluginFieldType.FLOAT32)
            return c.create_plugin(c.name, trt.PluginFieldCollection([p0]))
    return None

if __name__ == '__main__':
    os.system("rm -f ./*.plan")
    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    testCase = "fp%s" % ('16' if int(npDataType == np.float16) else '32')
    print("Test <%s>" % testCase)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    trtFile = "./model-" + testCase + ".plan"
    if os.path.isfile(trtFile):
        with open(trtFile, 'rb') as f:
            engineStr = f.read()
            engine = trt.Runtime(logger).deserialize_cuda_engine(engineStr)
        if engine == None:
            print("Failed loading engine!")
            exit()
        print("Succeeded loading engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << 0)
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        config.flags = 1 << int(trt.BuilderFlag.FP16) if int(npDataType == np.float16) else 0

        inputTensorList = []
        trtDataType = trt.float16 if int(npDataType == np.float16) else trt.float32
        inputTensorList.append(network.add_input('inputT', trtDataType, [-1, -1, -1, -1]))

        profile = builder.create_optimization_profile()
        profile.set_shape('inputT', [1, 1, 1, 1], [nIn, cIn, hIn, wIn], [nIn * 2, cIn * 2, hIn * 2, wIn * 2])
        config.add_optimization_profile(profile)

        pluginLayer = network.add_plugin_v2(inputTensorList, getLayerNormPlugin())
        pluginLayer.get_output(0).dtype = trtDataType

        network.mark_output(pluginLayer.get_output(0))
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            exit()
        print("Succeeded building engine!")
        with open(trtFile, 'wb') as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0, [nIn, cIn, hIn, wIn])

    print("Binding all? %s" % (["No", "Yes"][int(context.all_binding_shapes_specified)]))
    _, stream = cudart.cudaStreamCreate()

    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->", engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i))

    #data = np.random.rand(nIn,cIn,hIn,wIn).astype(np.float32)
    data = np.arange(nIn * cIn * hIn * wIn).reshape(nIn, cIn, hIn, wIn).astype(npDataType)

    bufferH = []
    bufferH.append(data)
    bufferH.append(np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1))))

    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpyAsync(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

    context.execute_async_v2(bufferD, stream)

    for i in range(nInput, nInput + nOutput):
        cudart.cudaMemcpyAsync(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    resCPU = layerNormCPU(bufferH, globalEpsilon)
    print("check result:", check(resCPU[0], bufferH[-1], True))

    print("Test <%s> finish!" % testCase)
