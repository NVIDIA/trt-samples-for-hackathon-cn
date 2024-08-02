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

import sys
from pathlib import Path

import tensorrt as trt

sys.path.append("/trtcookbook/include")
from utils import (TRTWrapperV1, build_mnist_network_trt, case_mark, export_network_as_onnx, print_network)

wenet_onnx_file = Path("/trtcookbook/00-Data/model/model-wenet.onnx")
output_mnist_onnx_file = Path("model-trained-network.onnx")
output_wenet_onnx_file = Path("model-wenet-network.onnx")

@case_mark
def case_mnist():
    tw = TRTWrapperV1()
    output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)
    for tensor in output_tensor_list:
        tw.network.mark_output(tensor)

    print_network(tw.network)
    export_network_as_onnx(tw.network, output_mnist_onnx_file, True)

@case_mark
def case_wenet():
    tw = TRTWrapperV1()

    parser = trt.OnnxParser(tw.network, tw.logger)
    with open(wenet_onnx_file, "rb") as model:
        parser.parse(model.read())

    export_network_as_onnx(tw.network, output_wenet_onnx_file, True)

if __name__ == "__main__":
    # Use a network of MNIST
    case_mnist()
    # Use wenet encodernetwork
    case_wenet()

    print("Finish")
