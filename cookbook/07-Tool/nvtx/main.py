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
import nvtx
from tensorrt_cookbook import TRTWrapperV1, build_mnist_network_trt

trt_file = Path("model.trt")
data = {"x": np.arange(1 * 1 * 28 * 28, dtype=np.float32).reshape(1, 1, 28, 28)}

tw = TRTWrapperV1()

output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)

tw.build(output_tensor_list)

tw.setup(data)

# Do inference with random input with nvtx marks rather than using `infer()` directly.
for _ in range(10):
    with nvtx.annotate("Inference with nvtx.annotate", color='c'):
        tw.context.execute_async_v3(0)

for _ in range(10):
    nvtx.push_range("Inference with nvtx.push_range/nvtx.pop_range", color="blue")  # another way to use nvtx
    tw.context.execute_async_v3(0)
    nvtx.pop_range()

print("Finish")
