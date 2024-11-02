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
import os
import onnx

from tensorrt_cookbook import case_mark

input_onnx_file = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "model" / "model-trained.onnx"
input_onnx_file_no_weight = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "model" / "model-trained-no-weight.onnx"
input_weight_file_path = input_onnx_file_no_weight.resolve().parent

output_onnx_file_external_weight = "model-external-weight.onnx"
output_weight_file = output_onnx_file_external_weight + ".weight"
output_onnx_file_internal_weight = "model-internal-weight.onnx"

@case_mark
def case_separate():
    print(f"Convert {input_onnx_file.name} to {output_onnx_file_external_weight} and {output_weight_file}")

    onnx_model = onnx.load(input_onnx_file, load_external_data=False)
    onnx.save(onnx_model, output_onnx_file_external_weight, save_as_external_data=True, all_tensors_to_one_file=True, location=output_weight_file)

@case_mark
def case_merge():
    print(f"Convert {input_onnx_file_no_weight.name} and {input_weight_file_path.name} to {output_onnx_file_internal_weight}")

    onnx_model = onnx.load(input_onnx_file_no_weight, load_external_data=False)
    onnx.load_external_data_for_model(onnx_model, str(input_weight_file_path))
    onnx.save(onnx_model, output_onnx_file_internal_weight, save_as_external_data=False)

if __name__ == "__main__":
    # Separate weight from a ONNX
    case_separate()
    # Merge weight and ONNX into one
    case_merge()

print("Finish")
