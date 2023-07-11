#!/usr/bin/python

import ctypes
from cuda import cudart
from datetime import datetime as dt
from glob import glob
import numpy as np
import os
import sys
import tensorrt as trt
from calibrator import DecoderCalibrator

basePath = "./"
onnxFile = sorted(glob(basePath + "decoderV*.onnx"))[-1]
trtFile = basePath + "decoder.plan"

isForSubmit = True
isConvertToStaticNetwork = False
isPrintNetwork = False
additionOutput = []

useInt8 = False
calibrationDataFile = "/workspace/data/calibration.npz"
int8CacheFile = basePath + "encoder-int8.cache"

strictTypeLayer = []

useTimeCache = True
timeCacheFile = "./decoder.cache"

nTestBS = 4
nTestSL = 64

logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')
for soFile in glob(basePath + "*.so"):
    ctypes.cdll.LoadLibrary(soFile)

timeCache = b""
if useTimeCache and os.path.isfile(timeCacheFile):
    with open(timeCacheFile, 'rb') as f:
        timeCache = f.read()
    if timeCache == None:
        print("Failed getting serialized timing cache!")
        exit()
    print("Succeeded getting serialized timing cache!")

if os.path.isfile(trtFile):
    print("Engine existed!")
    with open(trtFile, 'rb') as f:
        engineString = f.read()
else:
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    if useTimeCache:
        cache = config.create_timing_cache(timeCache)
        config.set_timing_cache(cache, False)

    if isForSubmit:
        config.max_workspace_size = 22 << 30
        config.flags = 1 << int(trt.BuilderFlag.FP16)
        config.flags = config.flags & ~(1 << int(trt.BuilderFlag.TF32))
        if useInt8:
            config.flags = config.flags | (1 << int(trt.BuilderFlag.INT8))
            config.int8_calibrator = DecoderCalibrator(calibrationDataFile, int8CacheFile, 16, 64)
    else:
        config.max_workspace_size = 6 << 30
        config.flags = config.flags & ~(1 << int(trt.BuilderFlag.TF32))
        config.flags = config.flags | (1 << int(trt.BuilderFlag.DEBUG))
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        if useInt8:
            config.flags = config.flags | (1 << int(trt.BuilderFlag.INT8))
            config.int8_calibrator = DecoderCalibrator(calibrationDataFile, int8CacheFile, 16, 64)

    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding ONNX file!")
        exit()
    print("Succeeded finding ONNX file!")
    with open(onnxFile, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing ONNX file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing ONNX file!")

    if isForSubmit:
        inputT0 = network.get_input(0)
        inputT0.shape = [-1, -1, 256]
        profile.set_shape(inputT0.name, (1, 16, 256), (64, 1024, 256), (64, 1024, 256))
        inputT1 = network.get_input(1)
        inputT1.shape = [-1]
        profile.set_shape(inputT1.name, (1, ), (64, ), (64, ))
        inputT2 = network.get_input(2)
        #inputT2.shape = [-1,10,-1]
        #profile.set_shape(inputT2.name, (1,10,32), (4,10,64), (16,10,128))
        inputT2.shape = [-1, 10, 64]
        profile.set_shape(inputT2.name, (1, 10, 64), (64, 10, 64), (64, 10, 64))
        inputT3 = network.get_input(3)
        inputT3.shape = [-1, 10]
        profile.set_shape(inputT3.name, (1, 10), (64, 10), (64, 10))
        inputT4 = network.get_input(4)
        inputT4.shape = [-1, 10]
        profile.set_shape(inputT4.name, (1, 10), (64, 10), (64, 10))
        config.add_optimization_profile(profile)

    else:
        if isConvertToStaticNetwork:
            inputT0 = network.get_input(0)
            inputT0.shape = [3, 17, 256]
            inputT1 = network.get_input(1)
            inputT1.shape = [3]
            inputT2 = network.get_input(2)
            inputT2.shape = [3, 10, 64]
            inputT3 = network.get_input(3)
            inputT3.shape = [3, 10]
            inputT4 = network.get_input(4)
            inputT4.shape = [3, 10]
        else:
            inputT0 = network.get_input(0)
            inputT0.shape = [-1, -1, 256]
            profile.set_shape(inputT0.name, (1, 16, 256), (4, 64, 256), (4, 64, 256))
            inputT1 = network.get_input(1)
            inputT1.shape = [-1]
            profile.set_shape(inputT1.name, (1, ), (4, ), (4, ))
            inputT2 = network.get_input(2)
            inputT2.shape = [-1, 10, 64]
            profile.set_shape(inputT2.name, (1, 10, 64), (4, 10, 64), (4, 10, 64))
            inputT3 = network.get_input(3)
            inputT3.shape = [-1, 10]
            profile.set_shape(inputT3.name, (1, 10), (4, 10), (4, 10))
            inputT4 = network.get_input(4)
            inputT4.shape = [-1, 10]
            profile.set_shape(inputT4.name, (1, 10), (4, 10), (4, 10))
            config.add_optimization_profile(profile)

    if isPrintNetwork:
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            print(i, "%s,in=%d,out=%d,%s" % (str(layer.type)[10:], layer.num_inputs, layer.num_outputs, layer.name))
            for j in range(layer.num_inputs):
                tensor = layer.get_input(j)
                if tensor == None:
                    print("\tInput  %2d:" % j, "None")
                else:
                    print("\tInput  %2d:%s,%s,%s" % (j, tensor.shape, str(tensor.dtype)[9:], tensor.name))
            for j in range(layer.num_outputs):
                tensor = layer.get_output(j)
                if tensor == None:
                    print("\tOutput %2d:" % j, "None")
                else:
                    print("\tOutput %2d:%s,%s,%s" % (j, tensor.shape, str(tensor.dtype)[9:], tensor.name))
        exit()

    if len(strictTypeLayer) > 0:
        for index in strictTypeLayer:
            layer = network.get_layer(i)
            layer.precision = trt.float32
            layer.get_output(0).dtype = trt.float32

    if len(additionOutput) > 0:
        for index in additionOutput:
            network.mark_output(network.get_layer(index).get_output(0))
            #network.mark_output(network.get_layer(index).get_input(0))

    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()

    if useTimeCache and not os.path.isfile(timeCacheFile):
        timeCache = config.get_timing_cache()
        timeCacheString = timeCache.serialize()
        with open(timeCacheFile, 'wb') as f:
            f.write(timeCacheString)
            print("Succeeded saving .cache file!")

    print("Succeeded building engine!")
    with open(trtFile, 'wb') as f:
        f.write(engineString)
'''
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

context = engine.create_execution_context()
context.set_binding_shape(0, [nTestBS,nTestSL,256])
context.set_binding_shape(1, [nTestBS])
context.set_binding_shape(2, [nTestBS,10,64])
context.set_binding_shape(3, [nTestBS,10])
context.set_binding_shape(4, [nTestBS,10])
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput
for i in range(nInput):
    print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
for i in range(nInput,nInput+nOutput):
    print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

dd = np.load("/workspace/data/decoder-%d-%d.npz"%(nTestBS,nTestSL))

bufferH = []
bufferH.append(np.ascontiguousarray(dd['encoder_out'][:nTestBS,:nTestSL].reshape(-1)))
bufferH.append(np.ascontiguousarray(dd['encoder_out_lens'][:nTestBS].reshape(-1)))
#bufferH.append(np.ascontiguousarray(np.array([2,3,4],dtype=np.int32).reshape(-1)))
bufferH.append(np.ascontiguousarray(dd['hyps_pad_sos_eos'][:nTestBS].reshape(-1)))
bufferH.append(np.ascontiguousarray(dd['hyps_lens_sos'][:nTestBS].reshape(-1)))
bufferH.append(np.ascontiguousarray(dd['ctc_score'][:nTestBS].reshape(-1)))

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

out={}
for i in range(nInput + nOutput):
    print(engine.get_binding_name(i))
    print(bufferH[i].reshape(context.get_binding_shape(i)))
    out[str(i)] = bufferH[i]

np.savez('decoderOut.npz',**out)

for b in bufferD:
    cudart.cudaFree(b)

print("Finished building Decoder engine!")
'''
