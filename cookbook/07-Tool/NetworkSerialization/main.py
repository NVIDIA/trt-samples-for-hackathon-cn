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

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import NetworkSerialization, TRTWrapperV1, case_mark, check_array, cookbook_path, load_mnist_network_trt, load_large_network_trt

@case_mark
def case_simple():
    json_file = Path("model-trained-network.json")
    para_file = Path("model-trained-network.npz")
    data = {"x": np.load(cookbook_path("00-Data", "data", "InferenceData.npy"))}

    tw = TRTWrapperV1(logger="VERBOSE")

    load_mnist_network_trt(tw)

    tw.build()
    tw.setup(data)
    tw.infer()

    output_ref = {name: tw.buffer[name][0] for name in tw.buffer.keys() if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT}

    # Initialization
    ns = NetworkSerialization(json_file, para_file)

    # Serialization
    ns.serialize(
        logger=tw.logger,
        builder=tw.builder,
        builder_config=tw.config,
        network=tw.network,
        optimization_profile_list=[tw.profile],  # More than one profile is acceptable
    )
    # Equivalent call
    # ns.serialize(tw=tw, optimization_profile_list=[tw.profile])

    del tw, ns  # Distinguish those objects between serialization and deserialization

    # Initialize another ns (maybe on another machine) to do deserialization
    ns = NetworkSerialization(json_file, para_file)

    # Deserialization
    ns.deserialize()

    # Build engine and do inference to see the result
    tw = TRTWrapperV1(logger=ns.logger)
    tw.builder, tw.network, tw.config = ns.builder, ns.network, ns.builder_config

    tw.build()
    tw.setup(data)
    tw.infer()

    output_rebuild = {name: tw.buffer[name][0] for name in tw.buffer.keys() if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT}

    for name in output_ref.keys():
        check_array(output_rebuild[name], output_ref[name], des=name)

@case_mark
def case_large_model():
    json_file = Path("model-large-network.json")
    para_file = Path("model-large-network.npz")
    data = {
        "input_ids": np.arange(4 * 5, dtype=np.int64).reshape(4, 5),
        "attention_mask": np.random.randint(0, 2, [4, 5]).astype(np.int64),
    }

    tw = TRTWrapperV1(logger="VERBOSE")

    load_large_network_trt(tw)

    tw.build()
    tw.setup(data, b_print_io=False)
    tw.infer(b_print_io=False)

    output_ref = {name: tw.buffer[name][0] for name in tw.buffer.keys() if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT}

    ns = NetworkSerialization(json_file, para_file)
    ns.serialize(tw=tw, optimization_profile_list=[tw.profile])
    del tw, ns

    ns = NetworkSerialization(json_file, para_file)
    ns.deserialize()
    tw = TRTWrapperV1(logger=ns.logger)
    tw.builder, tw.network, tw.config = ns.builder, ns.network, ns.builder_config

    tw.build()
    tw.setup(data, b_print_io=False)
    tw.infer(b_print_io=False)

    output_rebuild = {name: tw.buffer[name][0] for name in tw.buffer.keys() if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT}

    for name in output_ref.keys():
        check_array(output_rebuild[name], output_ref[name], des=name)

if __name__ == "__main__":
    # Use a network of MNIST
    case_simple()
    # Use a large network
    case_large_model()
    # TODO: a case with plugin v3 layer, and provide .so file when deserialization
    # TODO: a case with plugin v3 layer, but do not provide .so file when deserialization
    # TODO: a case with callback object
    # TODO: synchronize the cases with `cookbook/tests/NetworkSerialization`

    print("Finish")
