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
from tensorrt_cookbook import TRTWrapperV1, build_mnist_network_trt, case_mark

data_file = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data" / "InferenceData.npy"
data = {"x": np.load(data_file)}

@case_mark
def case_normal():
    tw = TRTWrapperV1(logger="INFO")  # Get budget information from INFO level
    tw.network = tw.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tw.config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)

    output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)

    tw.build(output_tensor_list)

    tw.runtime = trt.Runtime(tw.logger)
    tw.engine = tw.runtime.deserialize_cuda_engine(tw.engine_bytes)

    print(f"{tw.engine.get_weight_streaming_automatic_budget()=}B")  # Read-only
    # print(f"{tw.engine.minimum_weight_streaming_budget=}B")  # Read-only, not required in V2 API
    tw.engine.weight_streaming_budget_v2 = 1 << 20  # Modify budget as 1 MiB in this example
    # tw.engine.weight_streaming_budget = tw.engine.minimum_weight_streaming_budget + 100000  # Set a value greater or equal to minimum, not required in V2 API
    print(f"{tw.engine.weight_streaming_scratch_memory_size=}B")  # Read-only, changes with budget

    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # Use weight streaming
    case_normal()

    print("Finish")
