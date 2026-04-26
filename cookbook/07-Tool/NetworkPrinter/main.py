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

import os
from pathlib import Path

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import (TRTWrapperV1, build_mnist_network_trt, case_mark, datatype_cast, export_network_as_onnx, print_network)

large_onnx_file = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "model" / "model-large.onnx"
output_mnist_onnx_file = Path("model-trained-network.onnx")
output_large_onnx_file = Path("model-large-network.onnx")
output_loop_onnx_file = Path("model-loop-network.onnx")

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

    print_network(tw.network)
    export_network_as_onnx(tw.network, output_large_onnx_file, True)

@case_mark
def case_loop():
    data = {"x": np.ones((1, 4), dtype=np.float32)}
    tw = TRTWrapperV1()

    x = tw.network.add_input("x", datatype_cast(data["x"].dtype, "trt"), data["x"].shape)
    loop = tw.network.add_loop()
    rec = loop.add_recurrence(x)
    trip = tw.network.add_constant([1], np.array([3], dtype=np.int32)).get_output(0)
    loop.add_trip_limit(trip, trt.TripLimit.COUNT)
    one = tw.network.add_constant([1, 4], np.ones((1, 4), dtype=np.float32)).get_output(0)
    ew = tw.network.add_elementwise(rec.get_output(0), one, trt.ElementWiseOperation.SUM)
    rec.set_input(1, ew.get_output(0))
    out_layer = loop.add_loop_output(ew.get_output(0), trt.LoopOutput.LAST_VALUE, 0)

    tw.network.mark_output(out_layer.get_output(0))

    print_network(tw.network)
    export_network_as_onnx(tw.network, output_loop_onnx_file, True)

if __name__ == "__main__":
    # Use a network of MNIST
    # case_mnist()
    # Use large encoder network
    case_large()
    # Use a network with loop structure
    # case_loop()

    print("Finish")
