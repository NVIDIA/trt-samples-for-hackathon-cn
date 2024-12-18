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

import tensorrt as trt

from tensorrt_cookbook import TRTWrapperV1

tw = TRTWrapperV1()

tensor = tw.network.add_input("inputT0", trt.float32, [3, 4, 5])
layer = tw.network.add_identity(tensor)
tw.network.mark_output(layer.get_output(0))

engine_bytes = tw.builder.build_serialized_network(tw.network, tw.config)

print(f"{engine_bytes.dtype = }")
print(f"{engine_bytes.nbytes = }")
