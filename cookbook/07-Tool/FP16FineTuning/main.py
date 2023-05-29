import ctypes
import os
from copy import deepcopy
from datetime import datetime as dt

import numpy as np
import tensorrt as trt
from cuda import cudart
from tqdm import tqdm

os.chdir("/w/gitlab/tensorrt-cookbook/08-Tool/FP16FineTuning")  # fro debug

# Customized variable ----------------------------------------------------------
onnxFile = "model.onnx"  # required
pluginFileList = []  # optional
ioDataFile = "IOData.npz"  # optional
targetAccuracy = 5  # optional
reportFile = "report.txt"

# Other cariable ---------------------------------------------------------------
planFile = "model.plan"
timingCacheFile = "model.timingCache"
bPrintNetwork = False
bUseOnnxruntime = False
bDrawPlot = False
excludeList = {"SHAPE", "PLUGIN", "PLUGIN_V2", "CONSTANT", "ASSERTION", "SHUFFLE", "IDENTITY", "CONCATENATION", "GATHER", "SLICE", "RESIZE", "UNARY", "CONDITION", "CONDITIONAL_INPUT", "CONDITIONAL_OUTPUT", "FILL", "NON_ZERO", "ONE_HOT"}
resultList = []

np.random.seed(31193)
np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

# preparation work -------------------------------------------------------------
logger = trt.Logger(trt.Logger.ERROR)

if not os.path.exists(onnxFile):
    print("Failed finding %s" % onnxFile)
    exit()

if len(pluginFileList) > 0:
    bUseOnnxruntime = False

    trt.init_libnvinfer_plugins(logger, '')
    for soFile in pluginFileList:
        if not os.path.exists(soFile):
            print("Failed finding %s" % soFile)
            exit()
        ctypes.cdll.LoadLibrary(soFile)

# parse ONNX file to get metadata of network -----------------------------------
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
with open(onnxFile, "rb") as model:
    if not parser.parse(model.read()):
        print("Failed parsing %s" % onnxFile)
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
    print("Succeeded parsing %s" % onnxFile)

layerList = []
for i in range(network.num_layers):
    layer = network.get_layer(i)
    layerList.append([layer.name, layer.type.name])
    if bPrintNetwork:
        print("%4d->%s,in=%d,out=%d,%s" % (i, str(layer.type)[10:], layer.num_inputs, layer.num_outputs, layer.name))
        for j in range(layer.num_inputs):
            tensor = layer.get_input(j)
            print("\tInput  %2d:%s,%s,%s" % (j, tensor.shape, str(tensor.dtype)[9:], tensor.name))
        for j in range(layer.num_outputs):
            tensor = layer.get_output(j)
            print("\tOutput %2d:%s,%s,%s" % (j, tensor.shape, str(tensor.dtype)[9:], tensor.name))

# preare data for inference ----------------------------------------------------
dataMap = {}
if ioDataFile != "":  # use IO data from file and create shape list from
    ioData = np.load(ioDataFile)
    for key, value in ioData.items():
        dataMap[key] = value

else:  # use random input data
    for i in range(network.num_inputs):
        inputTensor = network.get_input(i)
        shape = [x if x != -1 else 1 for x in inputTensor.shape]
        dataMap[inputTensor.name] = np.random.rand(np.prod(shape)).astype(trt.nptype(inputTensor.dtype)).reshape(shape)

# Main process -----------------------------------------------------------------
def run(bFP32, layerNameListInFP32=[]):
    #print("%s, Build engine of %s: %s" % (dt.now(), ("FP32" if bFP32 else "FP16"), [x[0] for x in layerNameListInFP32]))  # for debug
    command = "trtexec --onnx=%s --useSpinWait --noDataTransfers" % onnxFile
    command += " " + "--saveEngine=%s" % planFile
    #command += " " + "--verbose"  # for debug
    if not bFP32:
        command += " " + "--fp16" + " "
    if len(layerNameListInFP32) > 0:
        command += " " + "--precisionConstraints=prefer"
    for layerName, layerType in layerNameListInFP32:
        if layerType not in excludeList:
            command += " " + "--layerPrecisions=%s:fp32" % layerName
        else:
            #print("Skip Layer %s, Type = %s" % (layerName, layerType))
            return

    command += " " + "2>/dev/null"
    output = os.popen(command)

    time = "+inf"
    for line in output.readlines():
        #print(line)  # for debug
        if "[I] GPU Compute Time" in line:
            time = float(line.split("ms")[3].split("=")[1])

    with open(planFile, "rb") as f:
        engineString = f.read()
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)  # create inference Engine using Runtime

    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context = engine.create_execution_context()
    for key, value in dataMap.items():
        if engine.get_tensor_mode(key) == trt.TensorIOMode.INPUT:
            context.set_input_shape(key, value.shape)
    #for i in range(nIO):  # for debug
    #    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    bufferH = []
    for i in range(nInput):
        bufferH.append(np.ascontiguousarray(dataMap[lTensorName[i]]))
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))

    context.execute_async_v3(0)

    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    if bFP32:
        for i in range(nInput, nIO):
            dataMap[lTensorName[i]] = bufferH[i]
    else:
        maxErrorList = []
        mediumErrorList = []
        for i in range(nInput, nIO):
            a = bufferH[i]
            b = dataMap[lTensorName[i]]
            maxErrorList.append(np.max(np.abs(a - b)))
            mediumErrorList.append(np.median(np.abs(a - b)))

        resultList.append([time, maxErrorList, mediumErrorList, layerNameListInFP32])

    for b in bufferD:  # free the GPU memory buffer after all work
        cudart.cudaFree(b)

    return

# Build model in FP32 mode
run(True)

# Build model in FP16 mode
run(False)

for layer in tqdm(layerList):
    run(False, [layer])

print(resultList)

resultList = sorted(resultList, key=(lambda x: x[1][0]))  # sort by max absolute error

with open(reportFile, "w") as ff:
    for line in resultList:
        ff.write(str(line))
        ff.write("\n")

if targetAccuracy > 0:
    n = np.argmin(np.cumsum([x[1][0] for x in resultList]) >= resultList[-1][1][0] - targetAccuracy)

    for i in range(n):
        print(resultList[i])

    # double check
    #run(False, None)

    if bDrawPlot:
        from matplotlib import pyplot as plt
