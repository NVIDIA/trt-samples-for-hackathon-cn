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

from pathlib import Path

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import (CookbookAlgorithmSelector, TRTWrapperV1, case_mark, datatype_cast)

trt_file = Path("engine.trt")

@case_mark
def case_simple():
    data = {"x": np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)}
    callback = {"algorithm_selector": CookbookAlgorithmSelector(i_strategy=0)}
    tw = TRTWrapperV1(callback_object_dict=callback)

    w = trt.Weights(np.ones((1, 1, 3, 3), dtype=np.float32))
    b = trt.Weights(np.zeros((1, ), dtype=np.float32))

    x = tw.network.add_input("x", datatype_cast(data["x"].dtype, "trt"), data["x"].shape)
    conv = tw.network.add_convolution_nd(x, 1, [3, 3], w, b)

    tw.build([conv.get_output(0)])
    tw.serialize_engine(trt_file)

if __name__ == "__main__":

    case_simple()  # TODO: check this

    print("Finish")
