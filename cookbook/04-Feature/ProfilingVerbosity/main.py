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

import sys

import numpy as np
import tensorrt as trt

sys.path.append("/trtcookbook/include")
from utils import TRTWrapperV1, build_mnist_network_trt, case_mark

data = {"x": np.arange(28 * 28, dtype=np.float32).reshape(1, 1, 28, 28)}

@case_mark
def case_normal(verbosity):
    logger = trt.Logger(trt.Logger.VERBOSE)
    tw = TRTWrapperV1(logger)
    tw.config.profiling_verbosity = verbosity  # -> 02-API/BuilderConfig

    output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)
    tw.build(output_tensor_list)
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    case_normal(trt.ProfilingVerbosity.LAYER_NAMES_ONLY)  # default
    case_normal(trt.ProfilingVerbosity.NONE)
    case_normal(trt.ProfilingVerbosity.DETAILED)

    print("Finish")
