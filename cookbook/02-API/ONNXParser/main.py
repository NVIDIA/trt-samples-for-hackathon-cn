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

import ctypes
import numpy as np
import onnx
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, case_mark, check_api_coverage, cookbook_path, parse_onnx, print_enumerated_members

data_path = cookbook_path("00-Data", "data")
model_path = cookbook_path("00-Data", "model")
data = {"x": np.load(data_path / "InferenceData.npy")}
shape = list(data["x"].shape)

@case_mark
def case_normal():
    tw = TRTWrapperV1()
    parser = trt.OnnxParser(tw.network, tw.logger)

    check_api_coverage(parser)  # Sanity check, unnecessary in normal workflow

    print(f"\n{'=' * 64} Usage show")

    parser.set_builder_config(tw.builder_config)

    # Check whether one certain operator is supported by ONNX parser
    print(f"{parser.supports_operator('LayerNormalization') = }")

    # Query the plugin libraries needed to implement operations used by the parser in a version-compatible engine
    print(f"{parser.get_used_vc_plugin_libraries() = }")

    # Flags related
    print(f"{parser.flags = }")
    parser.set_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)
    print(f"{parser.get_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM) = }")
    parser.clear_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)
    print_enumerated_members(trt.OnnxParserFlag)

    print(f"{trt.get_nv_onnx_parser_version() = }")

    onnx_file = model_path / "model-trained.onnx"

    parser.parse_from_file(str(onnx_file))

    for i in range(parser.num_subgraphs):
        print(f"  subgraph[{i}] supported={parser.is_subgraph_supported(i)}, nodes={parser.get_subgraph_nodes(i)}")

    input_tensor = tw.network.get_input(0)
    tw.profile.set_shape(input_tensor.name, shape, [1] + shape[1:], [4] + shape[1:])

    layer = parser.get_layer_output_tensor("TopK", 1)  # Get layer from parser
    print(f"TopK output tensor from parser: {layer}")

    tw.build()

    tw.setup(data)
    tw.infer()

@case_mark
def case_parse(index: int):
    tw = TRTWrapperV1()
    parser = trt.OnnxParser(tw.network, tw.logger)
    parser.set_builder_config(tw.builder_config)

    onnx_file = model_path / "model-trained.onnx"

    # 6 equivalent methods to parse ONNX files
    match index:
        case 0:
            # 1. parse from file
            res = parser.parse_from_file(str(onnx_file))
        case 1:
            # 2. parse from bytes
            with open(onnx_file, "rb") as onnx_bytes:
                res = parser.parse(onnx_bytes.read())
        case 2:
            # 3. deprecated
            with open(onnx_file, "rb") as onnx_bytes:
                res, information = parser.supports_model(onnx_bytes.read(), str(model_path))
        case 3:
            # 4. check and parse from bytes
            with open(onnx_file, "rb") as onnx_bytes:
                res = parser.supports_model_v2(onnx_bytes.read(), str(model_path))
        case 4:
            # 5. deprecated
            with open(onnx_file, "rb") as onnx_bytes:
                res = parser.parse_with_weight_descriptors(onnx_bytes.read())
        case 5:
            # 6. parse model and weights separately
            with open(onnx_file, "rb") as onnx_bytes:
                res = parser.load_model_proto(onnx_bytes.read())
            # In this example, we extract weights binary data from the same ONNX model file
            # Actually the weights can be from another ONNX model files
            onnx_model = onnx.load(onnx_file)
            bufs = {}  # Keep buffers alive until parse_model_proto() finishes.
            for i in onnx_model.graph.initializer:
                bufs[i.name] = (ctypes.c_ubyte * len(i.raw_data)).from_buffer_copy(i.raw_data)
                assert parser.load_initializer(i.name, ctypes.addressof(bufs[i.name]), len(bufs[i.name]))

            res = parser.parse_model_proto()

    print(f"Parsing result: {res}")

    input_tensor = tw.network.get_input(0)
    tw.profile.set_shape(input_tensor.name, shape, [1] + shape[1:], [4] + shape[1:])

    tw.build()

    tw.setup(data)
    tw.infer()

@case_mark
def case_error():
    tw = TRTWrapperV1()
    parser = trt.OnnxParser(tw.network, tw.logger)

    onnx_file = model_path / "model-unknown.onnx"
    parse_onnx(onnx_file, tw.logger, tw.network, tw.builder_config, parser)
    res = parser.num_errors == 0

    assert res is False, "This ONNX model has errors, parsing should fail"
    print(f"Fail parsing {onnx_file} with {parser.num_errors} error(s).")

    public_member = check_api_coverage(parser.get_error(0))  # Sanity check, unnecessary in normal workflow
    print(f"\n{'=' * 64} Usage show")
    for i in range(parser.num_errors):  # Get error information
        error = parser.get_error(i)
        print(f"Error number {i}: {error = }")
        for member in public_member:
            try:
                if callable(getattr(error, member)):
                    result = getattr(error, member)()
                    print(f"error.{member}() = {result}")
                else:
                    assert False, "trt.ParserError has non-callable public members"
                    result = getattr(error, member)
                    print(f"error.{member} = {result}")
            except Exception as e:
                print(f"Exception: {e}")

    parser.clear_errors()

@case_mark
def case_subgraph():
    local_data = {"x": np.arange(30)}
    local_shape = list(local_data["x"].shape)

    tw = TRTWrapperV1()
    parser = trt.OnnxParser(tw.network, tw.logger)

    onnx_file = model_path / "model-for.onnx"
    parse_onnx(onnx_file, tw.logger, tw.network, tw.builder_config, parser)

    input_tensor = tw.network.get_input(0)
    tw.profile.set_shape(input_tensor.name, [1], local_shape, local_shape)

    print(f"{parser.num_subgraphs = }")
    for i in range(parser.num_subgraphs):
        subgraph_nodes = parser.get_subgraph_nodes(i)
        print(f"parser.get_subgraph_nodes({i}) = {subgraph_nodes}")
        print(f"parser.is_subgraph_supported({i}) = {parser.is_subgraph_supported(i)}")

    tw.build()

    tw.setup(local_data)
    tw.infer()

if __name__ == "__main__":

    case_normal()

    for i in range(6):
        case_parse(i)

    case_error()
    case_subgraph()

    print("Finish")
