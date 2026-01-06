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
from tensorrt_cookbook import TRTWrapperDDS, case_mark, datatype_np_to_trt

data = {"tensor": np.random.rand(60).astype(np.float32).reshape(5, 3, 4), "tensor1": np.random.rand(150).astype(np.float32).reshape(5, 3, 10)}

@case_mark
def case_simple():
    tw = TRTWrapperDDS()
    tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    tw.config.add_optimization_profile(tw.profile)

    layer_max_output = tw.network.add_constant([], np.int32(20).reshape(-1))
    layer = tw.network.add_nms(tensor0, tensor1, layer_max_output.get_output(0))
    layer.topk_box_limit = 100  # [OPtional] Modify maximum of operator TopK
    layer.bounding_box_format = trt.BoundingBoxFormat.CENTER_SIZES  # [Optional] Modify box format

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using NMS layer
    case_simple()

    print("Finish")
