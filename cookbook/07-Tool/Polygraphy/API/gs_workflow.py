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

from pathlib import Path

import onnx
import onnx_graphsurgeon as gs
from polygraphy.backend.onnx.loader import fold_constants
from tensorrt_cookbook import cookbook_path

onnx_file = cookbook_path("00-Data", "model", "model-redundant.onnx")
onnx_file_output = Path("model-redundant-gs.onnx")

onnx_file_path = onnx_file.resolve().parent

onnx_model = onnx.load(onnx_file, load_external_data=False)
onnx.load_external_data_for_model(onnx_model, onnx_file_path)
graph = gs.import_onnx(onnx_model)
n_node_input = len(graph.nodes)

# ---- The onnx-graphsurgeon half: state what we know about the model.
# This model computes its `Reshape` target shape at runtime (`Shape` -> `ReduceProd` -> `Gather` ->
# `Concat`), which is only necessary because the batch dimension is dynamic. Pinning the input
# shape is information that lives outside the file, so no automatic pass can invent it.
input_tensor = graph.inputs[0]
print(f"Input {input_tensor.name}: {input_tensor.shape} -> [7, 2, 3, 4]")
input_tensor.shape = [7, 2, 3, 4]

# `cleanup` drops nodes that no longer reach an output, `toposort` restores a valid node order.
# Both are gs APIs, and both are needed before handing the graph to another tool.
graph.cleanup().toposort()

# ---- The Polygraphy half: let it do the arithmetic we just made possible.
# With a static input shape the whole shape-computing chain has constant inputs, so constant
# folding evaluates it and replaces it with a single initializer. This is the same work
# `polygraphy surgeon sanitize --fold-constant --override-input-shapes` does from the CLI
# (see ../Surgeon/main.sh), just split so the gs side can be arbitrary Python.
onnx_model = gs.export_onnx(graph)
onnx_model = fold_constants(onnx_model, allow_onnxruntime_shape_inference=True)

graph = gs.import_onnx(onnx_model)
graph.cleanup().toposort()  # Folding leaves the now-unused nodes behind, `cleanup` removes them
onnx_model = gs.export_onnx(graph)

onnx.save(onnx_model, onnx_file_output, save_as_external_data=True, all_tensors_to_one_file=True, location=str(onnx_file_output.name) + ".weight")
print(f"Succeed saving {onnx_file_output.name}: {n_node_input} -> {len(graph.nodes)} Nodes, {len(graph.tensors().keys())} tensors")
