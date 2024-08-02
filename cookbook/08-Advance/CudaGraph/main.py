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

import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cudart

sys.path.append("/trtcookbook/include")
from utils import TRTWrapperV1, build_mnist_network_trt

trt_file = Path("model.trt")
data = {"inputT0": np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)}

tw = TRTWrapperV1()

output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)

tw.build(output_tensor_list)
tw.setup(data)

# Do one inference before CUDA graph capture, do we need this?
tw.context.execute_async_v3(0)

# CUDA Graph capture
_, stream = cudart.cudaStreamCreate()
cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)

for name in tw.tensor_name_list:
    if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        cudart.cudaMemcpyAsync(tw.buffer[name][1], tw.buffer[name][0].ctypes.data, tw.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

tw.context.execute_async_v3(stream)

for name in tw.tensor_name_list:
    if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
        cudart.cudaMemcpyAsync(tw.buffer[name][0].ctypes.data, tw.buffer[name][1], tw.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

#cudart.cudaStreamSynchronize(stream)  # Do not synchronize during capture
_, graph = cudart.cudaStreamEndCapture(stream)
_, graphExe = cudart.cudaGraphInstantiate(graph, 0)

# CUDA graph launch
cudart.cudaGraphLaunch(graphExe, stream)
cudart.cudaStreamSynchronize(stream)

# If size of input tensors changes, we need to recapture the CUDA graph then launch it.

# Other work after CUDA graph asunch
for name in tw.tensor_name_list:
    print(name)
    print(tw.buffer[name][0])

for _, device_buffer, _ in tw.buffer.values():
    cudart.cudaFree(device_buffer)

print("Finish")
