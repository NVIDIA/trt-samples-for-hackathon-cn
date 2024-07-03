#
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

onnx_file = "model-trained.onnx"
output_onnx_file = "model-external-weight.onnx"
output_weight_file = str(output_onnx_file).split("/")[-1] + ".weight"

print(f"Convert {onnx_file} to {output_onnx_file} and {output_weight_file}")

onnx_model = onnx.load(onnx_file, load_external_data=False)
onnx.save(onnx_model, output_onnx_file, save_as_external_data=True, all_tensors_to_one_file=True, location=output_weight_file)

print("Finish")
