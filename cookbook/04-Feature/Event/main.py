# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from tensorrt_cookbook import TRTWrapperV1, case_mark

@case_mark
def case_simple():
    tw = TRTWrapperV1()

    data = {"inputT0": np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)}

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_identity(tensor)
    layer.get_output(0).name = "outputT0"
    tw.build([layer.get_output(0)])

    tw.setup(data)

    stream = cudart.cudaStreamCreate()[1]
    event = cudart.cudaEventCreate()[1]

    status = tw.context.set_input_consumed_event(event)
    event_from_context = tw.context.get_input_consumed_event()
    print(f"{event = }")
    print(f"Return value of context.set_input_consumed_event(event) = {status}")
    print(f"Return value of context.get_input_consumed_event = {event_from_context}")

    input_name = "inputT0"
    output_name = "outputT0"

    # Run inference for the first time
    cudart.cudaMemcpyAsync(tw.buffer[input_name][1], tw.buffer[input_name][0].ctypes.data, tw.buffer[input_name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    tw.context.execute_async_v3(stream)

    # Trigger the event so that we can safely update the input buffer for the next inference
    cudart.cudaEventSynchronize(event)

    # Run inference again with updated input buffer
    tw.buffer[input_name][0].fill(1.0)
    cudart.cudaMemcpyAsync(tw.buffer[input_name][1], tw.buffer[input_name][0].ctypes.data, tw.buffer[input_name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    tw.context.execute_async_v3(stream)

    cudart.cudaMemcpyAsync(tw.buffer[output_name][0].ctypes.data, tw.buffer[output_name][1], tw.buffer[output_name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    print(f"output sample = {tw.buffer[output_name][0].reshape(-1)[:8]}")

    cudart.cudaEventDestroy(event)
    cudart.cudaStreamDestroy(stream)

if __name__ == "__main__":
    case_simple()

    print("Finish")
