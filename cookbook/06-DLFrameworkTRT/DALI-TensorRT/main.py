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

from importlib import util

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, case_mark

def has_module(module_name: str) -> bool:
    return util.find_spec(module_name) is not None

@case_mark
def case_dali_to_tensorrt():
    if not has_module("nvidia.dali"):
        print("[SKIP] nvidia.dali is unavailable")
        return

    from nvidia.dali import fn, pipeline_def, types

    @pipeline_def
    def dali_pipeline(height: int, width: int):
        image_hwc = fn.random.uniform(device="gpu", range=(0.0, 255.0), shape=(height, width, 3), dtype=types.FLOAT)
        image_chw = fn.transpose(image_hwc, perm=[2, 0, 1], device="gpu")
        image_nchw = fn.expand_dims(image_chw, axes=[0])
        image_normalized = image_nchw / 255.0
        return image_normalized

    height = 224
    width = 224

    pipeline = dali_pipeline(batch_size=1, num_threads=2, device_id=0, height=height, width=width)
    pipeline.build()

    dali_output = pipeline.run()[0]
    input_data = np.ascontiguousarray(dali_output.as_cpu().as_array()[0].astype(np.float32))

    data = {"inputT0": input_data}

    tw = TRTWrapperV1()
    input_tensor = tw.network.add_input("inputT0", trt.float32, [1, 3, height, width])
    layer = tw.network.add_identity(input_tensor)
    layer.get_output(0).name = "outputT0"
    tw.build([layer.get_output(0)])

    tw.setup(data)
    tw.infer()

    output_data = tw.buffer["outputT0"][0]
    np.testing.assert_allclose(output_data, input_data, rtol=1e-6, atol=1e-6)
    print("DALI -> TensorRT inference check passed")

if __name__ == "__main__":
    case_dali_to_tensorrt()

    print("Finish")
