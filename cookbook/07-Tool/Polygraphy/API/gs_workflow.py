#
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

from pathlib import Path

import onnx
import onnx_graphsurgeon as gs
from polygraphy.backend.onnx.loader import fold_constants

onnx_file = Path("/trtcookbook/00-Data/model/model-redundant.onnx")
onnx_file_output = Path("model-redundant-gs.onnx")

onnx_file_path = onnx_file.resolve().parent

onnx_model = onnx.load(onnx_file, load_external_data=False)
onnx.load_external_data_for_model(onnx_model, onnx_file_path)
graph = gs.import_onnx(onnx_model)

# Do something with gs APIs

onnx_model = gs.export_onnx(graph)
onnx_model = fold_constants(onnx_model, allow_onnxruntime_shape_inference=True)
onnx.save(onnx_model, onnx_file_output, save_as_external_data=True, all_tensors_to_one_file=True, location=str(onnx_file_output.name) + ".weight")
print(f"Succeed saving {onnx_file_output.name}: {len(graph.nodes):5d} Nodes, {len(graph.tensors().keys()):5d} tensors")
