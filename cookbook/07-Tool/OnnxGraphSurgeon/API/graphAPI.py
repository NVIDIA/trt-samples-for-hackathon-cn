# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections import OrderedDict

import numpy as np
import onnx_graphsurgeon as gs

# the same model as ../01-CreateModel.py
tensor0 = gs.Variable("tensor0", np.float32, ["B", 3, 64, 64])
tensor1 = gs.Variable("tensor1", np.float32, ["B", 1, 64, 64])
tensor2 = gs.Variable("tensor2", np.float32, None)
tensor3 = gs.Variable("tensor3", np.float32, None)
constant0 = gs.Constant(name="constant0", values=np.ones(shape=[1, 3, 3, 3], dtype=np.float32))
constant1 = gs.Constant(name="constant1", values=np.ones(shape=[1], dtype=np.float32))
node0 = gs.Node("Conv", "myConv", inputs=[tensor0, constant0], outputs=[tensor1])
node0.attrs = OrderedDict([
    ["dilations", [1, 1]],
    ["kernel_shape", [3, 3]],
    ["pads", [1, 1, 1, 1]],
    ["strides", [1, 1]],
])
node1 = gs.Node("Add", "myAdd", inputs=[tensor1, constant1], outputs=[tensor2])
node2 = gs.Node("Relu", "myRelu", inputs=[tensor2], outputs=[tensor3])
graph = gs.Graph(nodes=[node0, node1, node2], inputs=[tensor0], outputs=[tensor3])

print(f"{graph.DEFAULT_OPSET = }")  # equivalent to graph.opset
print(f"{graph.GLOBAL_FUNC_MAP = }")
print(f"{graph.OPSET_FUNC_MAP = }")

with graph.node_ids():
    print(graph._get_node_id(node0))

print(graph.inputs)
print(graph.outputs)
print(graph.nodes)
print(graph.tensors())
print(f"{graph._local_tensors() = }")
print(f"{graph._foreign_tensors() = }")
print(f"{graph._functions() = }")

#print("graph._generate_name('myGraph') = %s" % graph._generate_name("myGraph"))

#print("graph._get_node_id() = %s" % graph._get_node_id(node0))
#print("graph._get_used_node_ids() = %s" % graph._get_used_node_ids(node0))

print(graph.name)
print(graph.opset)
print(graph.doc_string)
print(graph.import_domains)
print(graph.producer_name)
print(graph.producer_version)

print(graph.name_idx)

graph2 = graph.copy

graph.layer(inputs=[tensor0, constant0], outputs=[tensor1], op='MyOp')  # add nodes into graph

graph.cleanup().toposort().fold_constants()
"""
DEFAULT_OPSET           ++++
GLOBAL_FUNC_MAP++++++
OPSET_FUNC_MAP++++++

_foreign_tensors++++++++++++
_functions
_generate_name
_get_node_id
_get_used_node_ids
_local_tensors
_merge_subgraph_functions
cleanup++++++++
copy++++++++++
doc_string++++++++
fold_constants
functions
import_domains
inputs+++++++++++++++
layer
name+++++++++++++++
name_idx
node_ids ++++++++++
nodes++++++++++++++++++
opset      ++++++
outputs+++++++++++
producer_name
producer_version
register
subgraphs
tensors+++++++++++++
toposort++++++++++
"""
