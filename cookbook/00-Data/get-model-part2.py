# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
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

from collections import OrderedDict

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from tensorrt_cookbook import add_node, case_mark, cookbook_path, initialize_random_seed

rng = initialize_random_seed()

model_path = cookbook_path("00-Data", "model")
model_path.mkdir(parents=True, exist_ok=True)
onnx_file_custom_op = model_path / "model-addscalar.onnx"
onnx_file_half_mnist = model_path / "model-half-mnist.onnx"
onnx_file_invalid = model_path / "model-invalid.onnx"
onnx_file_redundant = model_path / "model-redundant.onnx"
onnx_file_unknown = model_path / "model-unknown.onnx"
onnx_file_reshape = model_path / "model-reshape.onnx"
onnx_file_labeled = model_path / "model-labeled.onnx"

def export_graph(graph: gs.Graph, onnx_file):
    graph.cleanup().toposort()
    onnx_model = gs.export_onnx(graph)
    onnx.save(onnx_model, onnx_file)
    print(f"Succeed exporting {onnx_file}")

@case_mark
def case_custom_op():
    """Export a model that uses a custom AddScalar operator."""
    graph = gs.Graph(nodes=[], inputs=[], outputs=[])

    batch_name = "nBS"
    inputT0 = gs.Variable("inputT0", np.float32, [batch_name])

    n = 0
    scope_name = "CustomOpModel"

    tensor1, n = add_node(graph, "AddScalar", [inputT0], OrderedDict([["scalar", 1.0]]), np.float32, [batch_name], scope_name, "", n)

    graph.inputs = [inputT0]
    graph.outputs = [tensor1]

    export_graph(graph, onnx_file_custom_op)

@case_mark
def case_half_mnist():
    """Export a half-MNIST style backbone for quick convolution workflow demos."""
    graph = gs.Graph(nodes=[], inputs=[], outputs=[])

    inputT0 = gs.Variable("inputT0", np.float32, ["B", 1, 28, 28])
    constant32x1 = gs.Constant("constant32x1x5x5", np.ascontiguousarray(rng.uniform(-1.0, 1.0, size=(32, 1, 5, 5)).astype(np.float32)))
    constant32 = gs.Constant("constant32", np.ascontiguousarray(rng.uniform(-1.0, 1.0, size=(32, )).astype(np.float32)))
    constant64x32 = gs.Constant("constant64x32x5x5", np.ascontiguousarray(rng.uniform(-1.0, 1.0, size=(64, 32, 5, 5)).astype(np.float32)))
    constant64 = gs.Constant("constant64", np.ascontiguousarray(rng.uniform(-1.0, 1.0, size=(64, )).astype(np.float32)))

    n = 0
    scope_name = "HALF_MNIST"

    tensor1, n = add_node(graph, "Conv", [inputT0, constant32x1, constant32], OrderedDict([["kernel_shape", [5, 5]], ["pads", [2, 2, 2, 2]]]), np.float32, ["B", 32, 28, 28], scope_name, "", n)

    tensor2, n = add_node(graph, "Relu", [tensor1], OrderedDict(), np.float32, ["B", 32, 28, 28], scope_name, "", n)

    tensor3, n = add_node(graph, "MaxPool", [tensor2], OrderedDict([["kernel_shape", [2, 2]], ["pads", [0, 0, 0, 0]], ["strides", [2, 2]]]), np.float32, ["B", 32, 14, 14], scope_name, "", n)

    tensor4, n = add_node(graph, "Conv", [tensor3, constant64x32, constant64], OrderedDict([["kernel_shape", [5, 5]], ["pads", [2, 2, 2, 2]]]), np.float32, ["B", 64, 14, 14], scope_name, "", n)

    tensor5, n = add_node(graph, "Relu", [tensor4], OrderedDict(), np.float32, ["B", 64, 14, 14], scope_name, "", n)

    tensor6, n = add_node(graph, "MaxPool", [tensor5], OrderedDict([["kernel_shape", [2, 2]], ["pads", [0, 0, 0, 0]], ["strides", [2, 2]]]), np.float32, ["B", 64, 7, 7], scope_name, "", n)

    graph.inputs = [inputT0]
    graph.outputs = [tensor6]

    export_graph(graph, onnx_file_half_mnist)

@case_mark
def case_invalid():
    """Export an intentionally invalid model that includes division by zero."""
    graph = gs.Graph(nodes=[], inputs=[], outputs=[])

    batch_name = "nBS"
    inputT0 = gs.Variable("inputT0", np.float32, [batch_name])
    constant0 = gs.Constant("constant0", np.ascontiguousarray(np.zeros([1], dtype=np.float32)))

    n = 0
    scope_name = "InvalidModel"

    tensor1, n = add_node(graph, "Div", [inputT0, constant0], OrderedDict(), np.float32, [batch_name], scope_name, "", n)

    graph.inputs = [inputT0]
    graph.outputs = [tensor1]

    export_graph(graph, onnx_file_invalid)

@case_mark
def case_labeled():
    """Export a model with explicitly labeled input/output tensor dimensions."""
    graph = gs.Graph(nodes=[], inputs=[], outputs=[])

    inputT0 = gs.Variable("inputT0", np.float32, ["B", 1, 1])
    inputT1 = gs.Variable("inputT1", np.float32, ["B", 1])

    n = 0
    scope_name = "LabeledModel"

    tensor1, n = add_node(graph, "Identity", [inputT0], OrderedDict(), np.float32, inputT0.shape, scope_name, "", n)
    tensor2, n = add_node(graph, "Identity", [inputT1], OrderedDict(), np.float32, inputT1.shape, scope_name, "", n)

    graph.inputs = [inputT0, inputT1]
    graph.outputs = [tensor1, tensor2]

    export_graph(graph, onnx_file_labeled)

@case_mark
def case_redundant():
    """Export a graph with redundant shape logic to demonstrate constant folding."""
    graph = gs.Graph(nodes=[], inputs=[], outputs=[])

    batch_name = 7  # Use static shape to see the effect of constant fold
    inputT0 = gs.Variable("inputT0", np.float32, [batch_name, 2, 3, 4])
    constant0C1 = gs.Constant("constant0C1", np.ascontiguousarray(np.array([0, 1], dtype=np.int64)))
    constant2C3 = gs.Constant("constant2C3", np.ascontiguousarray(np.array([2, 3], dtype=np.int64)))

    n = 0
    scope_name = "RedundantModel"

    tensor1, n = add_node(graph, "Shape", [inputT0], OrderedDict(), np.int64, [4], scope_name, "", n)

    tensor2, n = add_node(graph, "ReduceProd", [tensor1], OrderedDict([["axes", [0]], ["keepdims", 1]]), np.int64, [1], scope_name, "", n)

    tensor3, n = add_node(graph, "Reshape", [inputT0, tensor2], OrderedDict(), np.float32, [f"{batch_name}*24"], scope_name, "", n)

    tensor4, n = add_node(graph, "Gather", [tensor1, constant0C1], OrderedDict(), np.int64, [2], scope_name, "", n)

    tensor5, n = add_node(graph, "Gather", [tensor1, constant2C3], OrderedDict(), np.int64, [2], scope_name, "", n)

    tensor6, n = add_node(graph, "ReduceProd", [tensor5], OrderedDict([["axes", [0]], ["keepdims", 1]]), np.int64, [1], scope_name, "", n)

    tensor7, n = add_node(graph, "Concat", [tensor4, tensor6], OrderedDict([["axis", 0]]), np.int64, [4], scope_name, "", n)

    tensor8, n = add_node(graph, "Reshape", [inputT0, tensor7], OrderedDict(), np.float32, [batch_name, 2, 12], scope_name, "", n)

    graph.inputs = [inputT0]
    graph.outputs = [tensor3, tensor8]

    export_graph(graph, onnx_file_redundant)

@case_mark
def case_reshape():
    """Export a model with a custom reshape-like operator and shape input index."""
    graph = gs.Graph(nodes=[], inputs=[], outputs=[])

    output_dimension = 3
    inputT0 = gs.Variable("inputT0", np.float32, [-1, -1, -1])
    inputT1 = gs.Variable("inputT1", np.int32, [output_dimension])

    n = 0
    scope_name = "ReshapeModel"

    tensor1, n = add_node(graph, "MyReshape", [inputT0, inputT1], OrderedDict([["tensorrt_plugin_shape_input_indices", np.array([1], dtype=np.int32)]]), np.float32, [-1 for _ in range(output_dimension)], scope_name, "", n)

    graph.inputs = [inputT0, inputT1]
    graph.outputs = [tensor1]

    export_graph(graph, onnx_file_reshape)

@case_mark
def case_unknown():
    """Export a model that intentionally contains unknown operator nodes."""
    graph = gs.Graph(nodes=[], inputs=[], outputs=[])

    inputT0 = gs.Variable("inputT0", np.float32, ["B"])

    n = 0
    scope_name = "UnknownModel"

    tensor1, n = add_node(graph, "Identity", [inputT0], OrderedDict(), np.float32, ["B"], scope_name, "", n)

    tensor2, n = add_node(graph, "UnknownNode1", [tensor1], OrderedDict(), np.float32, ["B"], scope_name, "", n)

    tensor3, n = add_node(graph, "Identity", [tensor2], OrderedDict(), np.float32, ["B"], scope_name, "", n)

    tensor4, n = add_node(graph, "UnknownNode2", [tensor3], OrderedDict(), np.float32, ["B"], scope_name, "", n)

    tensor5, n = add_node(graph, "Identity", [tensor4], OrderedDict(), np.float32, ["B"], scope_name, "", n)

    graph.inputs = [inputT0]
    graph.outputs = [tensor5]

    export_graph(graph, onnx_file_unknown)

if __name__ == "__main__":

    case_custom_op()
    case_half_mnist()
    case_invalid()
    case_labeled()
    case_redundant()
    case_reshape()
    case_unknown()

    print("Finish")
