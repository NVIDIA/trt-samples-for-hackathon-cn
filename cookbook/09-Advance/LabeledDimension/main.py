#
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt

onnxFile = "./model.onnx"

# 构造一个 .onnx 模型，两个输入张量的首维名字都叫做 "B"，并且两个张量的计算过程相互独立，理论上推理时两个输入张量的首维不相等也不影响计算过程
tensor0 = gs.Variable("tensor0", np.float32, ["B", 1, 1])
tensor1 = gs.Variable("tensor1", np.float32, ["B", 1])
tensor2 = gs.Variable("tensor2", np.float32, None)
tensor3 = gs.Variable("tensor3", np.float32, None)
node0 = gs.Node("Identity", "myIdentity0", inputs=[tensor0], outputs=[tensor2])
node1 = gs.Node("Identity", "myIdentity1", inputs=[tensor1], outputs=[tensor3])
graph = gs.Graph(nodes=[node0, node1], inputs=[tensor0, tensor1], outputs=[tensor2, tensor3])
onnx.save(gs.export_onnx(graph.cleanup().toposort()), onnxFile)

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()

parser = trt.OnnxParser(network, logger)
with open(onnxFile, "rb") as model:
    parser.parse(model.read())

inputT0 = network.get_input(0)
profile.set_shape(inputT0.name, (1, 1, 1), (4, 1, 1), (8, 1, 1))
inputT1 = network.get_input(1)
profile.set_shape(inputT1.name, (1, 1), (4, 1), (8, 1))
config.add_optimization_profile(profile)

engineString = builder.build_serialized_network(network, config)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
context = engine.create_execution_context()
context.set_binding_shape(0, [4, 1, 1])  # 绑定两个首维长度相等的形状
context.set_binding_shape(1, [4, 1])
print("Binding all? %s" % (["No", "Yes"][int(context.all_binding_shapes_specified)]))
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput
for i in range(engine.num_bindings):
    print("Bind[%2d]:i[%d]->" % (i, i) if engine.binding_is_input(i) else "Bind[%2d]:o[%d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

context.set_binding_shape(0, [4, 1, 1])  # 绑定两个首维长度不相等的形状
context.set_binding_shape(1, [5, 1])
print("Binding all? %s" % (["No", "Yes"][int(context.all_binding_shapes_specified)]))
nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
nOutput = engine.num_bindings - nInput
for i in range(engine.num_bindings):
    print("Bind[%2d]:i[%d]->" % (i, i) if engine.binding_is_input(i) else "Bind[%2d]:o[%d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
    '''
    data = cv2.imread(inferenceImage, cv2.IMREAD_GRAYSCALE).astype(np.float32).reshape(1, 1, nHeight, nWidth)
    bufferH = []
    bufferH.append(data)
    for i in range(nOutput):
        bufferH.append(np.empty(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))
    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)

    for i in range(nOutput):
        cudart.cudaMemcpy(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    print("inputH0 :", bufferH[0].shape)
    print("outputH0:", bufferH[-1].shape)
    print(bufferH[-1])

    for buffer in bufferD:
        cudart.cudaFree(buffer)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    run()
    run()
    '''
