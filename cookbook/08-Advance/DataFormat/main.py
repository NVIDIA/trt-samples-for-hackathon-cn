#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import ctypes
import os

import numpy as np
import tensorrt as trt
from cuda import cudart

os.chdir("/wili/tensorrt-cookbook/08-Advance/DataFormat")

np.set_printoptions(precision=3, linewidth=200, suppress=True)
np.random.seed(31193)
cudart.cudaDeviceSynchronize()

def printArrayInformation(x, info="", n=5):
    if 0 in x.shape:
        print('%s:%s' % (info, str(x.shape)))
        print()
        return
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t', x.reshape(-1)[:n], x.reshape(-1)[-n:])

def check(a, b, weak=False, checkEpsilon=1e-5):
    if weak:
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    print("check:%s, absDiff=%f, relDiff=%f" % (res, diff0, diff1))

def run(shape, dataType, format):
    testCase = "<shape=%s,dataType=%s,format=%s>" % (shape, dataType, format)
    print("Test %s" % testCase)
    logger = trt.Logger(trt.Logger.ERROR)

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    if dataType == trt.DataType.HALF:
        config.set_flag(trt.BuilderFlag.FP16)
    if dataType == trt.DataType.INT8:
        config.set_flag(trt.BuilderFlag.INT8)

    nDim = 4  # for normal cases, we use input tensor of 4 dimensions
    if dataType == trt.DataType.HALF and format in [trt.TensorFormat.CDHW32, trt.TensorFormat.DHWC8]:
        nDim = 5

    inputT0 = network.add_input("inputT0", dataType, [-1] * nDim)
    inputT0.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
    if dataType == trt.DataType.INT8:
        inputT0.set_dynamic_range(0, 384)

    profile.set_shape(inputT0.name, [1] * nDim, [64] * nDim, [64] * nDim)
    config.add_optimization_profile(profile)

    identityLayer = network.add_identity(inputT0)
    identityLayer.get_output(0).dtype = dataType
    identityLayer.get_output(0).allowed_formats = 1 << int(format)

    network.mark_output(identityLayer.get_output(0))
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        return
    print("Succeeded building engine!")
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context = engine.create_execution_context()
    context.set_input_shape(lTensorName[0], shape)
    #for i in range(nIO):
    #    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    bufferH = []
    bufferH.append(np.arange(np.prod(shape), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[0]))).reshape(shape))
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

    #print("Input: \n", bufferH[0])
    #print("Output:\n", bufferH[1])

    # check correctness manually
    if dataType == trt.DataType.FLOAT and format == trt.TensorFormat.LINEAR:
        check(bufferH[1], bufferH[0], weak=True)
        check(bufferH[0], bufferH[1], weak=True)

    elif dataType == trt.DataType.HALF and format == trt.TensorFormat.CHW2:
        if shape[1] % 2 == 0:  # no pad
            check(bufferH[1], bufferH[0].reshape(shape[0], shape[1] // 2, 2, shape[2], shape[3]).transpose(0, 1, 3, 4, 2).reshape(shape), weak=True)
            check(bufferH[0], bufferH[1].reshape(shape[0], shape[1] // 2, shape[2], shape[3], 2).transpose(0, 1, 4, 2, 3).reshape(shape), weak=True)
        else:  # need pad, this path is also correct when shape[1] % 2 == 0, but seems much more complex
            nTile = (shape[1] + 2 - 1) // 2
            nPadC = nTile * 2
            nPadWidth = nPadC - shape[1]
            padBuffer = np.concatenate([bufferH[0], np.zeros([shape[0], nPadWidth, shape[2], shape[3]], dtype=bufferH[0].dtype)], axis=1)
            buffer = padBuffer.reshape(shape[0], nTile, 2, shape[2], shape[3]).transpose(0, 1, 3, 4, 2).reshape(shape[0], nPadC, shape[2], shape[3])[:, :shape[1], :, :]
            check(bufferH[1], buffer, weak=True)
            padBuffer = np.concatenate([bufferH[1], np.zeros([shape[0], nPadWidth, shape[2], shape[3]], dtype=bufferH[1].dtype)], axis=1)
            buffer = padBuffer.reshape(shape[0], nTile, shape[2], shape[3], 2).transpose(0, 1, 4, 2, 3).reshape(shape[0], nPadC, shape[2], shape[3])[:, :shape[1], :, :]
            check(bufferH[0], buffer, weak=True)  # lose the last ((c + 1) // 2 * h * w - c * h * w // 2) element

    elif dataType == trt.DataType.HALF and format == trt.TensorFormat.HWC8:
        if shape[1] % 8 == 0:  # no pad
            check(bufferH[1], bufferH[0].reshape(shape[0], shape[1] // 8, 8, shape[2], shape[3]).transpose(0, 1, 3, 4, 2).reshape(shape), weak=True)
            check(bufferH[0], bufferH[1].reshape(shape[0], shape[1] // 8, shape[2], shape[3], 8).transpose(0, 1, 4, 2, 3).reshape(shape), weak=True)
        else:  # need pad, this path is also correct when shape[1] % 8 == 0, but seems much more complex
            nTile = (shape[1] + 8 - 1) // 8
            nPadC = nTile * 8
            nPadWidth = nPadC - shape[1]
            padBuffer = np.concatenate([bufferH[0], np.zeros([shape[0], nPadWidth, shape[2], shape[3]], dtype=bufferH[0].dtype)], axis=1)
            buffer = padBuffer.transpose(0, 2, 3, 1).reshape(shape[0], nPadC, shape[2], shape[3])[:, :shape[1], :, :]
            check(bufferH[1], buffer, weak=True)
            padBuffer = np.concatenate([bufferH[1], np.zeros([shape[0], nPadWidth, shape[2], shape[3]], dtype=bufferH[1].dtype)], axis=1)
            buffer = padBuffer.reshape(shape[0], nTile, shape[2], shape[3], 8).transpose(0, 1, 4, 2, 3).reshape(shape[0], nPadC, shape[2], shape[3])[:, :shape[1], :, :]
            check(bufferH[0], buffer, weak=True)  # lose the last ((c + 7) // 8 * 8) * (h * w-1) element

    elif dataType == trt.DataType.HALF and format == trt.TensorFormat.CHW4:
        if shape[1] % 4 == 0:  # no pad
            check(bufferH[1], bufferH[0].reshape(shape[0], shape[1] // 4, 4, shape[2], shape[3]).transpose(0, 1, 3, 4, 2).reshape(shape), weak=True)
            check(bufferH[0], bufferH[1].reshape(shape[0], shape[1] // 4, shape[2], shape[3], 4).transpose(0, 1, 4, 2, 3).reshape(shape), weak=True)
        else:  # need pad, this path is also correct when shape[1] % 4 == 0, but seems much more complex
            nTile = (shape[1] + 4 - 1) // 4
            nPadC = nTile * 4
            nPadWidth = nPadC - shape[1]
            padBuffer = np.concatenate([bufferH[0], np.zeros([shape[0], nPadWidth, shape[2], shape[3]], dtype=bufferH[0].dtype)], axis=1)
            buffer = padBuffer.reshape(shape[0], nTile, 4, shape[2], shape[3]).transpose(0, 1, 3, 4, 2).reshape(shape[0], nPadC, shape[2], shape[3])[:, :shape[1], :, :]
            check(bufferH[1], buffer, weak=True)
            padBuffer = np.concatenate([bufferH[1], np.zeros([shape[0], nPadWidth, shape[2], shape[3]], dtype=bufferH[1].dtype)], axis=1)
            buffer = padBuffer.reshape(shape[0], nTile, shape[2], shape[3], 4).transpose(0, 1, 4, 2, 3).reshape(shape[0], nPadC, shape[2], shape[3])[:, :shape[1], :, :]
            check(bufferH[0], buffer, weak=True)  # lose the last ((c + 1) // 4 * h * w - c * h * w // 4) element

    elif dataType == trt.DataType.HALF and format == trt.TensorFormat.CHW16:
        if shape[1] % 16 == 0:  # no pad
            check(bufferH[1], bufferH[0].reshape(shape[0], shape[1] // 16, 16, shape[2], shape[3]).transpose(0, 1, 3, 4, 2).reshape(shape), weak=True)
            check(bufferH[0], bufferH[1].reshape(shape[0], shape[1] // 16, shape[2], shape[3], 16).transpose(0, 1, 4, 2, 3).reshape(shape), weak=True)
        else:  # need pad, this path is also correct when shape[1] % 16 == 0, but seems much more complex
            nTile = (shape[1] + 16 - 1) // 16
            nPadC = nTile * 16
            nPadWidth = nPadC - shape[1]
            padBuffer = np.concatenate([bufferH[0], np.zeros([shape[0], nPadWidth, shape[2], shape[3]], dtype=bufferH[0].dtype)], axis=1)
            buffer = padBuffer.reshape(shape[0], nTile, 16, shape[2], shape[3]).transpose(0, 1, 3, 4, 2).reshape(shape[0], nPadC, shape[2], shape[3])[:, :shape[1], :, :]
            check(bufferH[1], buffer, weak=True)
            padBuffer = np.concatenate([bufferH[1], np.zeros([shape[0], nPadWidth, shape[2], shape[3]], dtype=bufferH[1].dtype)], axis=1)
            buffer = padBuffer.reshape(shape[0], nTile, shape[2], shape[3], 16).transpose(0, 1, 4, 2, 3).reshape(shape[0], nPadC, shape[2], shape[3])[:, :shape[1], :, :]
            check(bufferH[0], buffer, weak=True)  # lose the last ((c + 1) // 16 * h * w - c * h * w // 16) element

    elif dataType == trt.DataType.FLOAT and format == trt.TensorFormat.CHW32:
        if shape[1] % 32 == 0:  #  no pad
            check(bufferH[1], bufferH[0].reshape(shape[0], shape[1] // 32, 32, shape[2], shape[3]).transpose(0, 1, 3, 4, 2).reshape(shape), weak=True)
            check(bufferH[0], bufferH[1].reshape(shape[0], shape[1] // 32, shape[2], shape[3], 32).transpose(0, 1, 4, 2, 3).reshape(shape), weak=True)
        else:  # need pad, this path is also correct when shape[1] % 32 == 0, but seems much more complex
            nTile = (shape[1] + 31) // 32
            nPadC = nTile * 32
            nPadWidth = nPadC - shape[1]
            padBuffer = np.concatenate([bufferH[0], np.zeros([shape[0], nPadWidth, shape[2], shape[3]], dtype=bufferH[0].dtype)], axis=1)
            buffer = padBuffer.reshape(shape[0], nTile, 32, shape[2], shape[3]).transpose(0, 1, 3, 4, 2).reshape(shape[0], nPadC, shape[2], shape[3])[:, :shape[1], :, :]
            check(bufferH[1], buffer, weak=True)
            padBuffer = np.concatenate([bufferH[1], np.zeros([shape[0], nPadWidth, shape[2], shape[3]], dtype=bufferH[1].dtype)], axis=1)
            buffer = padBuffer.reshape(shape[0], nTile, shape[2], shape[3], 32).transpose(0, 1, 4, 2, 3).reshape(shape[0], nPadC, shape[2], shape[3])[:, :shape[1], :, :]
            check(bufferH[0], buffer, weak=True)  # lose the last ((c + 1) // 32 * h * w - c * h * w // 32) element

    elif dataType == trt.DataType.HALF and format == trt.TensorFormat.DHWC8:
        if shape[1] % 8 == 0:  # no pad
            check(bufferH[1], bufferH[0].reshape(shape[0], shape[1] // 8, 8, shape[2], shape[3], shape[4]).transpose(0, 1, 3, 4, 5, 2).reshape(shape), weak=True)
            check(bufferH[0], bufferH[1].reshape(shape[0], shape[1] // 8, shape[2], shape[3], shape[4], 8).transpose(0, 1, 5, 2, 3, 4).reshape(shape), weak=True)
        else:  # need pad, this path is also correct when shape[1] % 8 == 0, but seems much more complex
            nTile = (shape[1] + 8 - 1) // 8
            nPadC = nTile * 8
            nPadWidth = nPadC - shape[1]
            padBuffer = np.concatenate([bufferH[0], np.zeros([shape[0], nPadWidth, shape[2], shape[3], shape[4]], dtype=bufferH[0].dtype)], axis=1)
            buffer = padBuffer.transpose(0, 2, 3, 4, 1).reshape(shape[0], nPadC, shape[2], shape[3], shape[4])[:, :shape[1], :, :]
            check(bufferH[1], buffer, weak=True)
            padBuffer = np.concatenate([bufferH[1], np.zeros([shape[0], nPadWidth, shape[2], shape[3], shape[4]], dtype=bufferH[1].dtype)], axis=1)
            buffer = padBuffer.reshape(shape[0], nTile, shape[2], shape[3], shape[4], 8).transpose(0, 1, 5, 2, 3, 4).reshape(shape[0], nPadC, shape[2], shape[3], shape[4])[:, :shape[1], :, :, :]
            check(bufferH[0], buffer, weak=True)  # lose the last ((c + 7) // 8 * 8) * (h * w-1) element

    elif dataType == trt.DataType.HALF and format == trt.TensorFormat.CDHW32:
        if shape[1] % 32 == 0:  #  no pad
            check(bufferH[1], bufferH[0].reshape(shape[0], shape[1] // 32, 32, shape[2], shape[3], shape[4]).transpose(0, 1, 3, 4, 5, 2).reshape(shape), weak=True)
            check(bufferH[0], bufferH[1].reshape(shape[0], shape[1] // 32, shape[2], shape[3], shape[4], 32).transpose(0, 1, 5, 2, 3, 4).reshape(shape), weak=True)
        else:  # need pad, this path is also correct when shape[1] % 32 == 0, but seems much more complex
            nTile = (shape[1] + 32 - 1) // 32
            nPadC = nTile * 32
            nPadWidth = nPadC - shape[1]
            padBuffer = np.concatenate([bufferH[0], np.zeros([shape[0], nPadWidth, shape[2], shape[3], shape[4]], dtype=bufferH[0].dtype)], axis=1)
            buffer = padBuffer.reshape(shape[0], nTile, 32, shape[2], shape[3], shape[4]).transpose(0, 1, 3, 4, 5, 2).reshape(shape[0], nPadC, shape[2], shape[3], shape[4])[:, :shape[1], :, :, :]
            check(bufferH[1], buffer, weak=True)
            padBuffer = np.concatenate([bufferH[1], np.zeros([shape[0], nPadWidth, shape[2], shape[3], shape[4]], dtype=bufferH[1].dtype)], axis=1)
            buffer = padBuffer.reshape(shape[0], nTile, shape[2], shape[3], shape[4], 32).transpose(0, 1, 5, 2, 3, 4).reshape(shape[0], nPadC, shape[2], shape[3], shape[4])[:, :shape[1], :, :, :]
            check(bufferH[0], buffer, weak=True)  # lose the last ((c + 1) // 32 * h * w - c * h * w // 32) element

    elif dataType == trt.DataType.FLOAT and format == trt.TensorFormat.HWC:
        check(bufferH[1], bufferH[0].transpose(0, 2, 3, 1).reshape(shape), weak=True)
        check(bufferH[0], bufferH[1].reshape(shape[0], shape[2], shape[3], shape[1]).transpose(0, 3, 1, 2).reshape(shape), weak=True)

    elif dataType == trt.DataType.HALF and format == trt.TensorFormat.HWC16:
        if shape[1] % 16 == 0:  # no pad
            check(bufferH[1], bufferH[0].reshape(shape[0], shape[1] // 16, 16, shape[2], shape[3]).transpose(0, 1, 3, 4, 2).reshape(shape), weak=True)
            check(bufferH[0], bufferH[1].reshape(shape[0], shape[1] // 16, shape[2], shape[3], 16).transpose(0, 4, 1, 2, 3).reshape(shape), weak=True)
        else:  # need pad, this path is also correct when shape[1] % 16 == 0, but seems much more complex
            nTile = (shape[1] + 16 - 1) // 16
            nPadC = nTile * 16
            nPadWidth = nPadC - shape[1]
            padBuffer = np.concatenate([bufferH[0], np.zeros([shape[0], nPadWidth, shape[2], shape[3]], dtype=bufferH[0].dtype)], axis=1)
            buffer = padBuffer.transpose(0, 2, 3, 1).reshape(shape[0], nPadC, shape[2], shape[3])[:, :shape[1], :, :]
            check(bufferH[1], buffer, weak=True)
            padBuffer = np.concatenate([bufferH[1], np.zeros([shape[0], nPadWidth, shape[2], shape[3]], dtype=bufferH[1].dtype)], axis=1)
            buffer = padBuffer.reshape(shape[0], nTile, shape[2], shape[3], 16).transpose(0, 1, 4, 2, 3).reshape(shape[0], nPadC, shape[2], shape[3])[:, :shape[1], :, :]
            check(bufferH[0], buffer, weak=True)  # lose the last ((c + 7) // 16 * 16) * (h * w-1) element

    elif dataType == trt.DataType.FLOAT and format == trt.TensorFormat.DHWC:  # no change?
        check(bufferH[1], bufferH[0], weak=True)
        check(bufferH[0], bufferH[1], weak=True)
        #check(bufferH[1], bufferH[0].transpose(0, 2, 3, 1).reshape(shape), weak=True)
        #check(bufferH[0], bufferH[1].reshape(shape[0], shape[2], shape[3], shape[1]).transpose(0, 3, 1, 2).reshape(shape), weak=True)

    for b in bufferD:
        cudart.cudaFree(b)
    print("Test %s finish!\n" % testCase)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    run([1, 2, 3, 4], trt.DataType.FLOAT, trt.TensorFormat.LINEAR)
    run([1, 4, 2, 3], trt.DataType.HALF, trt.TensorFormat.CHW2)  # no pad
    run([1, 3, 2, 3], trt.DataType.HALF, trt.TensorFormat.CHW2)  # pad 1 channel
    run([1, 8, 2, 3], trt.DataType.HALF, trt.TensorFormat.HWC8)  # no pad
    run([1, 7, 2, 3], trt.DataType.HALF, trt.TensorFormat.HWC8)  # pad 1 channel
    run([1, 4, 2, 3], trt.DataType.HALF, trt.TensorFormat.CHW4)  # no pad
    run([1, 3, 2, 3], trt.DataType.HALF, trt.TensorFormat.CHW4)  # pad 1 channel
    run([1, 4, 2, 3], trt.DataType.HALF, trt.TensorFormat.CHW16)  # no pad
    run([1, 3, 2, 3], trt.DataType.HALF, trt.TensorFormat.CHW16)  # pad 1 channel
    run([1, 64, 2, 3], trt.DataType.FLOAT, trt.TensorFormat.CHW32)  # no pad
    run([1, 63, 2, 3], trt.DataType.FLOAT, trt.TensorFormat.CHW32)  # pad 1 channel
    run([1, 8, 1, 2, 3], trt.DataType.HALF, trt.TensorFormat.DHWC8)  # no pad
    run([1, 7, 1, 2, 3], trt.DataType.HALF, trt.TensorFormat.DHWC8)  # pad 1 channel
    run([1, 64, 1, 2, 3], trt.DataType.HALF, trt.TensorFormat.CDHW32)  # no pad
    run([1, 63, 1, 2, 3], trt.DataType.HALF, trt.TensorFormat.CDHW32)  # pad 1 channel
    run([1, 2, 3, 4], trt.DataType.FLOAT, trt.TensorFormat.HWC)
    #run([1, 2, 3, 4], trt.DataType.FLOAT, trt.TensorFormat.DLA_LINEAR)
    #run([1, 4, 2, 3], trt.DataType.HALF, trt.TensorFormat.DLA_HWC4)  # no pad
    #run([1, 3, 2, 3], trt.DataType.HALF, trt.TensorFormat.DLA_HWC4)  # pad 1 channel
    run([1, 16, 2, 3], trt.DataType.HALF, trt.TensorFormat.HWC16)  # no pad
    run([1, 15, 2, 3], trt.DataType.HALF, trt.TensorFormat.HWC16)  # pad 1 channel
    run([1, 2, 3, 4], trt.DataType.FLOAT, trt.TensorFormat.DHWC)
    print("Test all finish!")
