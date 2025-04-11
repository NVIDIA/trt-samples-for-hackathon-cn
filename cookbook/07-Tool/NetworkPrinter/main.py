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

import os
from pathlib import Path

import tensorrt as trt

from tensorrt_cookbook import (TRTWrapperV1, build_mnist_network_trt, case_mark, export_network_as_onnx, print_network)

large_onnx_file = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "model" / "model-large.onnx"
output_mnist_onnx_file = Path("model-trained-network.onnx")
output_large_onnx_file = Path("model-large-network.onnx")

@case_mark
def case_mnist():
    tw = TRTWrapperV1()
    output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)
    for tensor in output_tensor_list:  # No need to build engine
        tw.network.mark_output(tensor)

    print_network(tw.network)
    export_network_as_onnx(tw.network, output_mnist_onnx_file, True)

@case_mark
def case_large():
    tw = TRTWrapperV1()

    parser = trt.OnnxParser(tw.network, tw.logger)
    with open(large_onnx_file, "rb") as model:
        parser.parse(model.read())

    export_network_as_onnx(tw.network, output_large_onnx_file, True)

if __name__ == "__main__":
    # Use a network of MNIST
    case_mnist()
    # Use large encodernetwork
    case_large()

    print("Finish")
