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
import numpy as np
import tensorrt as trt

from tensorrt_cookbook import TRTWrapperV1

data_path = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data"
model_path = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "model"
onnx_file = model_path / "model-trained.onnx"
data = {"x": np.load(data_path / "InferenceData.npy")}
shape = list(data["x"].shape)

tw = TRTWrapperV1()

parser = trt.OnnxParser(tw.network, tw.logger)

# Check whether one certain operator is supported by ONNX parser
print(f"{parser.supports_operator('LayerNormalization') = }")

# Query the plugin libraries needed to implement operations used by the parser in a version-compatible engine
print(f"{parser.get_used_vc_plugin_libraries() = }")

# Flags related
print(f"{parser.flags = }")
parser.set_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)
parser.get_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)
parser.clear_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)

# 4 equivalent methods to parse ONNX files
# 1.
res = parser.parse_from_file(str(onnx_file))  # parse from file
"""
# 2.
with open(onnx_file, "rb") as onnx_bytes:
    res = parser.parse(onnx_bytes.read())

# 3.
with open(onnx_file, "rb") as onnx_bytes:
    res, information = parser.supports_model(onnx_bytes)

# 4.
with open(onnx_file, "rb") as onnx_bytes:
    res, information = parser.parse_with_weight_descriptors(onnx_bytes)
"""
if not res:
    print(f"Fail parsing {onnx_file}")
    for i in range(parser.num_errors):  # Get error information
        error = parser.get_error(i)
        print(error)  # Print error information
        print(f"{error.code() = }")
        print(f"{error.file() = }")
        print(f"{error.func() = }")
        print(f"{error.line() = }")
        print(f"{error.local_function_stack_size() = }")
        print(f"{error.local_function_stack() = }")
        print(f"{error.node_name() = }")
        print(f"{error.node_operator() = }")
        print(f"{error.node() = }")
    parser.clear_errors()

input_tensor = tw.network.get_input(0)
tw.profile.set_shape(input_tensor.name, shape, [1] + shape[1:], [4] + shape[1:])
tw.config.add_optimization_profile(tw.profile)

layer = parser.get_layer_output_tensor("TopK", 1)  # Get layer from parser

tw.build()

tw.setup(data)
tw.infer()
