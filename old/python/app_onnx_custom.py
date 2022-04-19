#!/usr/bin/python3

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

import torch
import numpy as np

src_onnx = 'custom.onnx'
dst_onnx = 'custom_surgeon.onnx'

class CustomModel(torch.nn.Module):
    def forward(self, x, grid):
        grid = torch.clamp(grid, -1.0, 1.0)
        return torch.nn.functional.grid_sample(input=x, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=True)

x = torch.randn(1, 3, 544, 960)
grid = torch.randn(1, 544, 960, 2)

custom = CustomModel()
print('output shape:', custom(x, grid).size())

input_names = ['x', 'grid']
output_names = ['y']
torch.onnx.export(custom, (x, grid), src_onnx, input_names=input_names, output_names=output_names, opset_version=11, verbose=True, 
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, do_constant_folding=False)

import onnx_graphsurgeon as gs
import onnx
import numpy as np

graph = gs.import_onnx(onnx.load(src_onnx))

for node in graph.nodes:
    if node.op == 'Resize' and node.i(2, 0).op == 'Concat':
    # actually not used in this sample
        node_concat = node.i(2, 0)
        
        values = []
        for i in range(len(node_concat.inputs)):
            c = node_concat.i(i, 0)
            # print(c)
            while c.op != 'Constant':
                c = c.i(0, 0)
            values.append(c.attrs['value'].values)
    
        #以下是不可靠的写法（不可靠地假定了0号父亲是Constant）
        #node_concat.i(0, 0).attrs['value'] = gs.Constant('', np.concatenate(values))
        #node.inputs[2] = node_concat.inputs[0]

        #以下是更可靠的写法
        node_constant = gs.Node(op="Constant", name=node_concat.name, attrs={'value':gs.Constant('', np.concatenate(values))})
        node_constant.outputs = node_concat.outputs[:]
        graph.nodes.append(node_constant)
        
        node_concat.outputs.clear()

    if node.op == 'Unsqueeze' and node.i(0, 0).op == 'Constant' and node.i(0, 0).attrs['value'].dtype == np.float64:
        node.i(0, 0).attrs['value'] = gs.Constant('', np.asarray([node.i(0, 0).attrs['value'].values], dtype=np.float32))
        
    if node.op == 'Clip':
        node_cast0 = node.i(1, 0)
        node_cast1 = node.i(2, 0)
        #change data type to fp32
        node_cast0.i(0, 0).attrs['value'] = gs.Constant('', np.asarray([-1.0], dtype=np.float32))
        node_cast1.i(0, 0).attrs['value'] = gs.Constant('', np.asarray([1.0], dtype=np.float32))
        #skip cast
        node.inputs = [node.inputs[0], node_cast0.inputs[0], node_cast1.inputs[0]]
        #cleanup cast
        node_cast0.outputs.clear()
        node_cast1.outputs.clear()

    if node.op == 'grid_sampler':
        #cleanup 3 unused inputs
        for i in [4, 3, 2]:
            node.i(i, 0).outputs.clear()
            del node.inputs[i]

graph.cleanup()
onnx.save(gs.export_onnx(graph), dst_onnx)

model = onnx.load(dst_onnx)

# May not work with non-standard ONNX op
#onnx.checker.check_model(model)
#print(onnx.helper.printable_graph(model.graph))

#trtexec --verbose --onnx=custom_surgeon.onnx --saveEngine=custom_surgeon.trt --plugins=./GridSamplerPlugin.so
