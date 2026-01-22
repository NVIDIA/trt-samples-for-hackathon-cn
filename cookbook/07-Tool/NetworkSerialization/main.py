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

import os
from pathlib import Path

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import (NetworkSerialization, TRTWrapperV1, build_large_network_trt, build_mnist_network_trt, case_mark, check_array)

data_path = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data"

mnist_json_file = Path("model-trained-network.json")
mnist_para_file = Path("model-trained-network.npz")
mnist_data = {"x": np.load(data_path / "InferenceData.npy")}

large_json_file = Path("model-large-network.json")
large_para_file = Path("model-large-network.npz")
large_data = {
    "input_ids": np.arange(4 * 5, dtype=np.int64).reshape(4, 5),
    "attention_mask": np.random.randint(0, 2, [4, 5]).astype(np.int64),
}

@case_mark
def case_simple(json_file, para_file, is_mnist: bool = True):
    tw = TRTWrapperV1(logger_level="VERBOSE")

    if is_mnist:
        output_tensor_list = build_mnist_network_trt(tw.logger, tw.config, tw.network, tw.profile)
    else:
        output_tensor_list = build_large_network_trt(tw.logger, tw.config, tw.network, tw.profile)

    tw.build(output_tensor_list)
    tw.setup(mnist_data if is_mnist else large_data)
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

    del tw, ns  # Note that the object used for serialization and deserialization is not the same one

    # Initialization
    ns = NetworkSerialization(json_file, para_file)

    # Deserialization
    ns.deserialize(print_network_before_return=False)

    # Build engine and do inference to see the result
    tw = TRTWrapperV1(logger=ns.logger)
    tw.builder, tw.network, tw.config = ns.builder, ns.network, ns.builder_config

    tw.build()
    tw.setup(mnist_data if is_mnist else large_data)
    tw.infer()

    output_rebuild = {name: tw.buffer[name][0] for name in tw.buffer.keys() if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT}

    for name in output_ref.keys():
        check_array(output_rebuild[name], output_ref[name], des=name)

if __name__ == "__main__":
    # Use a network of MNIST
    case_simple(mnist_json_file, mnist_para_file, True)
    # Use large encodernetwork
    for file_name in Path(".").glob('*.weight'):
        file_name.unlink()
    case_simple(large_json_file, large_para_file, False)

    print("Finish")
