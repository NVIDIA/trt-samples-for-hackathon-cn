#
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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
#

import sys

import numpy as np
import tensorrt as trt

sys.path.append("/trtcookbook/include")
from utils import TRTWrapperV1, case_mark

nB, nC, nH, nW = 1, 3, 3, 3
data = np.tile(np.array([1, 2, 5], dtype=np.float32).reshape(nC, 1, 1), (1, nH, nW)).reshape(nB, nC, nH, nW)
data = {"inputT0": data}

@case_mark
def case_simple():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_lrn(tensor, 3, 1.0, 1.0, 0.0001)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_window_size_alpha_beta_k():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_lrn(tensor, 5, 0.0, 2.0, 1.0)
    layer.window_size = 3
    layer.alpha = 1.0
    layer.beta = 1.0
    layer.k = 0.0001

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using Softmax layer
    case_simple()
    # Modify parameters after the constructor
    case_window_size_alpha_beta_k()

    print("Finish")
