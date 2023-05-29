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

from collections import OrderedDict

import numpy as np
import onnx_graphsurgeon as gs

# the same model as 08-Tool/OnnxGraphSurgeon/01-CreateModel
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

print("graph.DEFAULT_OPSET = %s" % graph.DEFAULT_OPSET)   # equivalent to graph.opset
print("graph.GLOBAL_FUNC_MAP = %s" % graph.GLOBAL_FUNC_MAP)
print("graph.OPSET_FUNC_MAP = %s" % graph.OPSET_FUNC_MAP)

print("graph._generate_name('myGraph') = %s" % graph._generate_name("myGraph"))
print("graph._local_tensors() = %s" % graph._local_tensors())  # equivalent to dict(graph.tensors())
print("graph._foreign_tensors() = %s" % graph._foreign_tensors())
#print("graph._get_node_id() = %s" % graph._get_node_id(node0))
#print("graph._get_used_node_ids() = %s" % graph._get_used_node_ids(node0))

print(graph.name)
print(graph.opset)
print(graph.doc_string)
print(graph.import_domains)
print(graph.producer_name)
print(graph.producer_version)
print(graph.inputs)
print(graph.outputs)
print(graph.nodes)
print(graph.tensors())
print(graph.graph.name_idx)
print(graph.node_ids())

graph2 = graph.copy

graph.layer(inputs=[tensor0, constant0], outputs=[tensor1], op='MyOp')  # add nodes into graph

"""
Member of IBuilder:
++++        shown above
----        not shown above
[no prefix] others

++++DEFAULT_OPSET
++++GLOBAL_FUNC_MAP
++++OPSET_FUNC_MAP
----__class__
__delattr__
__dict__
__dir__
__doc__
__eq__
__format__
__ge__
__getattr__
__getattribute__
__gt__
__hash__
__init__
__init_subclass__
__le__
__lt__
__module__
__name__
__ne__
__new__
__reduce__
__reduce_ex__
__repr__
__setattr__
__sizeof__
__str__
__subclasshook__
__weakref__
++++_foreign_tensors
++++_generate_name
++++_get_node_id
++++_get_used_node_ids
++++_local_tensors
----cleanup                                                                     refer to 08-Tool/OnnxGraphSurgeon/06-Fold.py
++++copy
++++doc_string
----fold_constants                                                              refer to 08-Tool/OnnxGraphSurgeon/06-Fold.py
++++import_domains
++++inputs
++++layer
++++name
++++name_idx
++++node_ids
++++nodes
++++opset
++++outputs
++++producer_name
++++producer_version
----register                                                                    refer to 08-Tool/OnnxGraphSurgeon/09-BuildModelWithAPI.py
++++tensors
----toposort                                                                    refer to 08-Tool/OnnxGraphSurgeon/06-Fold.py
"""