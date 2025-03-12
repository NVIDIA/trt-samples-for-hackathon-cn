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

from tensorrt_cookbook import (NetworkSerialization, TRTWrapperV1, build_mnist_network_trt, build_large_network_trt, case_mark)

data_path = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data"
model_path = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "model"

mnist_json_file = Path("model-trained-network.json")
mnist_para_file = Path("model-trained-network.npz")
mnist_data = {"x": np.load(data_path / "InferenceData.npy")}

large_onnx_file = Path(model_path / "model-large.onnx")
large_json_file = Path("model-large-network.json")
large_para_file = Path("model-large-network.npz")
large_data = {"inputT0": np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)}

@case_mark
def case_serialization(json_file, para_file, is_mnist: bool = True):
    tw = TRTWrapperV1(logger_level=trt.Logger.Severity.VERBOSE)

    if is_mnist:
        output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)
    else:
        output_tensor_list = build_large_network_trt(tw.logger, tw.config, tw.network, tw.profile)

    tw.build(output_tensor_list)

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

def case_deserialization(json_file, para_file, is_mnist: bool = True):
    # Initialization
    ns = NetworkSerialization(json_file, para_file)

    # Deserialization
    ns.deserialize()

    # Build engine and do inference to see the result
    tw = TRTWrapperV1(logger=ns.logger)
    tw.builder = ns.builder
    tw.network = ns.network
    tw.config = ns.builder_config

    data = mnist_data if is_mnist else large_data

    tw.build()
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # Use a network of MNIST
    #case_serialization(mnist_json_file, mnist_para_file, True)
    #case_deserialization(mnist_json_file, mnist_para_file, True)
    # Use large encodernetwork
    #case_serialization(large_json_file, large_para_file, False)
    case_deserialization(large_json_file, large_para_file, False)

    print("Finish")
