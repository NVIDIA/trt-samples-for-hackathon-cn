#!/usr/bin/env python3
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

import onnx
import onnx_graphsurgeon as gs
import numpy as np

@gs.Graph.register()
def shape(self, a):
    return self.layer(op="Shape", inputs=[a], outputs=["shape_out_gs"])[0]

@gs.Graph.register()
def reduce_prod(self, a, axes, keepdims=True):
    return self.layer(op="ReduceProd", inputs=[a], attrs={"axes": axes, "keepdims": int(keepdims)}, outputs=["reduce_prod_out_gs"])[0]

@gs.Graph.register()
def reshape(self, data, shape):
    return self.layer(op="Reshape", inputs=[data, shape], outputs=["reshape_out_gs"])[0]

@gs.Graph.register()
def gather(self, data, indices):
    return self.layer(op="Gather", inputs=[data, indices], outputs=["gather_out_gs"])[0]

@gs.Graph.register()
def concat(self, inputs, axis=0):
    return self.layer(op="Concat", inputs=inputs, attrs={"axis": axis}, outputs=["concat_out_gs"])[0]

graph = gs.Graph()

#graph.inputs = [gs.Variable("data", np.float32, (gs.Tensor.DYNAMIC, 3, gs.Tensor.DYNAMIC, 5))]  # shape=(A,3,B,5)
graph.inputs = [gs.Variable("data", np.float32, (2, 3, 4, 5))]  # shape=(A,3,B,5)

input_shape = graph.shape(graph.inputs[0])  # value=(A,3,B,5), shape=(4,)

volume = graph.reduce_prod(input_shape, axes=[0])  # value=(A*3*B*5), shape=()
flattened = graph.reshape(graph.inputs[0], volume)  # shape=(A*3*B*5,)

NC = graph.gather(input_shape, indices=[0, 1])  # value=(A,3), shape=(2,)
HW = graph.gather(input_shape, indices=[2, 3])  # value=(B,5), shape=(2,)
new_shape = graph.concat([NC, graph.reduce_prod(HW, axes=[0])])  # value=(A,3,B*5), shape=(3,)
partially_flattened = graph.reshape(graph.inputs[0], new_shape)  # shape=(A,3,B*5)

flattened.name = "flattened"
flattened.dtype = np.float32
partially_flattened.name = "partially_flattened"
partially_flattened.dtype = np.float32

graph.outputs = [flattened, partially_flattened]
onnx.save(gs.export_onnx(graph), "model.onnx")
