#
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

import numpy as np
import tensorrt as trt

from tensorrt_cookbook import TRTWrapperV1

trt_file = "model.trt"
data = {"inputT0": np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)}

tw = TRTWrapperV1()
tw.config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)  # Set the flag of version compatibility

tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
layer = tw.network.add_identity(tensor)

tw.build([layer.get_output(0)])
tw.serialize_engine("model.trt")

tw.runtime = trt.Runtime(tw.logger)  # We need to initialize a runtime outside tw since we must enable a switch here
tw.runtime.engine_host_code_allowed = True  # Turn on the switch
tw.setup(data)

tw.infer()
