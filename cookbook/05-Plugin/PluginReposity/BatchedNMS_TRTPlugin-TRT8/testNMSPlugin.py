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

import os

import numpy as np
import tensorrt as trt
from cuda import cudart

nDataSize = 3840
nRetainSize = 2000
nImageHeight = 960
nImageWidth = 1024
dataFile = "data.npz"
np.random.seed(31193)

def printArrayInfomation(x, info="", n=5):
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t', x.reshape(-1)[:n], x.reshape(-1)[-n:])

def getBatchedNMSPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == "BatchedNMS_TRT":
            parameterList = []
            parameterList.append(trt.PluginField("shareLocation", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("backgroundLabelId", np.array([-1], dtype=np.int32), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("numClasses", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("topK", np.array([nDataSize], dtype=np.int32), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("keepTopK", np.array([nRetainSize], dtype=np.int32), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("scoreThreshold", np.array([0.7], dtype=np.float32), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("iouThreshold", np.array([0.7], dtype=np.float32), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("isNormalized", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
    return None

def run():
    trtFile = "./model.plan"
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    #ctypes.cdll.LoadLibrary(soFile)  # 不需要加载 .so
    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            print("Failed loading engine!")
            return
        print("Succeeded loading engine!")
    else:
        builder = trt.Builder(logger)
        builder.max_batch_size = 1
        network = builder.create_network()
        config = builder.create_builder_config()

        tensor1 = network.add_input("data1", trt.float32, (nDataSize, 1, 4))
        tensor2 = network.add_input("data2", trt.float32, (nDataSize, 1))
        scaleLayer = network.add_scale(tensor1, trt.ScaleMode.UNIFORM, np.array([0.0], dtype=np.float32), np.array([1 / max(nImageHeight, nImageWidth)], dtype=np.float32), np.array([1.0], dtype=np.float32))
        nmsLayer = network.add_plugin_v2([scaleLayer.get_output(0), tensor2], getBatchedNMSPlugin())

        network.mark_output(nmsLayer.get_output(0))
        network.mark_output(nmsLayer.get_output(1))
        network.mark_output(nmsLayer.get_output(2))
        network.mark_output(nmsLayer.get_output(3))
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    #print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    #for i in range(nInput):
    #    print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
    #for i in range(nInput, nInput + nOutput):
    #    print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

    data = np.load(dataFile)["prop"][:nDataSize]
    norm = max(nImageHeight, nImageWidth)
    data[:, :4] /= norm

    bufferH = []
    bufferH.append(np.ascontiguousarray(data[:, :4].reshape(nDataSize, 1, 4)))
    bufferH.append(np.ascontiguousarray(data[:, 4].reshape(nDataSize, 1)))
    for i in range(nOutput):
        bufferH.append(np.empty(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))
    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute(1, bufferD)

    for i in range(nOutput):
        cudart.cudaMemcpy(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for i in range(nInput):
        printArrayInfomation(bufferH[i], "Input %d" % i)
    for i in range(nOutput):
        printArrayInfomation(bufferH[nInput + i] if i != 1 else bufferH[nInput + i] * norm, "Output%d" % i)

if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    run()
