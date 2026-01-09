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

from pathlib import Path

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, case_mark

n_context = 2
trt_file = Path("model.trt")
data = {"inputT0": np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)}

@case_mark
def case_normal():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, [-1, -1, -1])
    tw.profile.set_shape(tensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
    tw.config.add_optimization_profile(tw.profile)

    # Add another optimization-profile
    profile1 = tw.builder.create_optimization_profile()
    profile1.set_shape(tensor.name, [1, 1, 1], [6, 8, 10], [9, 12, 15])
    tw.config.add_optimization_profile(profile1)

    identity_layer = tw.network.add_identity(tensor)

    tw.build([identity_layer.get_output(0)])

    tw.runtime = trt.Runtime(tw.logger)
    tw.engine = tw.runtime.deserialize_cuda_engine(tw.engine_bytes)

    context_list = [tw.engine.create_execution_context() for _ in range(n_context)]

    for i in range(n_context):
        print(f"Use optimization-profile {i}")
        tw.context = context_list[i]
        tw.context.set_optimization_profile_async(i, 0)  # We only have 1 stream, i.e., 0
        tw.setup(data)
        tw.infer(b_print_io=False)

if __name__ == "__main__":
    case_normal()

    print("Finish")
