#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
from cuda import cudart  # 使用 cuda runtime API
import tensorrt as trt

# yapf:disable

trtFile = "./model.plan"

def run():
    logger = trt.Logger(trt.Logger.ERROR)                                       # 指定 Logger，可用等级：VERBOSE，INFO，WARNING，ERRROR，INTERNAL_ERROR
    if os.path.isfile(trtFile):                                                 # 如果有 .plan 文件则直接读取
        with open(trtFile, "rb") as f:
            engineString = f.read()
        if engineString == None:
            print("Failed getting serialized engine!")
            return
        print("Succeeded getting serialized engine!")
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
        if engine == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
    else:                                                                       # 没有 .plan 文件，从头开始创建
        builder = trt.Builder(logger)                                           # 网络元信息，Builder/Network/BuilderConfig/Profile 相关
        builder.max_batch_size = 3
        builder.max_workspace_size = 1 << 30
        network = builder.create_network()

        inputTensor = network.add_input("inputT0", trt.float32, [4, 5])  # 指定输入张量

        identityLayer = network.add_identity(inputTensor)                       # 恒等变换
        network.mark_output(identityLayer.get_output(0))                        # 标记输出张量

        engine = builder.build_cuda_engine(network)
        if engine == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, "wb") as f:                                          # 将序列化网络保存为 .plan 文件
            f.write(engine.serialize())
            print("Succeeded saving .plan file!")

    context = engine.create_execution_context()                                 # 创建 context（相当于 GPU 进程）
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])  # 获取 engine 绑定信息
    nOutput = engine.num_bindings - nInput
    for i in range(nInput):
        print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
    for i in range(nInput,nInput+nOutput):
        print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)              # 准备数据和 Host/Device 端内存
    bufferH = []
    bufferH.append(np.ascontiguousarray(data.reshape(-1)))
    for i in range(nInput, nInput + nOutput):
        bufferH.append(np.empty((3, ) + tuple(context.get_binding_shape(i)), dtype=trt.nptype(engine.get_binding_dtype(i))))
    bufferD = []
    for i in range(nInput + nOutput):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):                                                     # 首先将 Host 数据拷贝到 Device 端
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute(3, bufferD)                                                 # 运行推理计算

    for i in range(nInput, nInput + nOutput):                                   # 将结果从 Device 端拷回 Host 端
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for i in range(nInput + nOutput):
        print(engine.get_binding_name(i))
        print(bufferH[i].reshape((3, ) + tuple(context.get_binding_shape(i))))

    for b in bufferD:                                                           # 释放 Device 端内存
        cudart.cudaFree(b)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    run()                                                                       # 创建 TensorRT 引擎并推理
    run()                                                                       # 读取 TensorRT 引擎并推理
