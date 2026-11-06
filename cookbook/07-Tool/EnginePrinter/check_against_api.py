# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

import sys
from pathlib import Path

import tensorrt as trt

from tensorrt_cookbook import TRTWrapperV1, build_mnist_network_trt

output_path = Path(__file__).parent
trt_file = output_path / "model-trained.trt"

if __name__ == "__main__":
    """
    Build an engine, then assert the parser agrees with the API about it.
    This script is for catching the change of the TRT engine header, not for a user's example.
    """
    try:
        from tensorrt_cookbook import parse_engine_information
    except ImportError:
        print("Use `TRT_COOKBOOK_ENGINE_INTERNAL=1` to enable the feature")
        sys.exit(0)

    tw = TRTWrapperV1()
    tw.build(build_mnist_network_trt(tw))
    tw.serialize_engine(trt_file)
    info = parse_engine_information(trt_file)

    assert info["plan"]["trtVersion"] == trt.__version__, \
        f"TensorRT version of engine is {info['plan']['trtVersion']}, but current environment is {trt.__version__}"

    print("Finish")
