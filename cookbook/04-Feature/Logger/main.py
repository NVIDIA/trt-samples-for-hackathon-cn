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

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import MyLogger, TRTWrapperV1

data = {"inputT0": np.zeros([1], dtype=np.float32)}

my_logger = MyLogger()  # default severity is ERROR

print("{'='*64} Buildtime")
my_logger.min_severity = trt.ILogger.Severity.INFO  # change severity to INFO

tw = TRTWrapperV1(logger=my_logger)  # we need logger to create Builder

tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
layer = tw.network.add_identity(tensor)

tw.build([layer.get_output(0)])

print("{'='*64} Runtime")
my_logger.min_severity = trt.ILogger.Severity.VERBOSE  # change severity to VERBOSE
tw.runtime = trt.Runtime(logger=my_logger)  # we need logger to create Runtime
tw.setup(data)
tw.infer()
