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

shape0 = [1, 3, 4, 5]
nB, nC, nH0, nW0 = shape0
shape1 = [6, 10]
nH1, nW1 = shape1
data0 = np.arange(nB).reshape(nB, 1, 1, 1) * 1000 + np.arange(nC).reshape(1, nC, 1, 1) * 100 + np.arange(nH0).reshape(1, 1, nH0, 1) * 10 + np.arange(nW0).reshape(1, 1, 1, nW0)
data0 = data0.astype(np.float32)
dataX = np.random.randint(0, nH0, [nB, nH1, nW1, 1], dtype=np.int32) / (nH0 - 1) * 2 - 1
dataY = np.random.randint(0, nW0, [nB, nH1, nW1, 1], dtype=np.int32) / (nW0 - 1) * 2 - 1
data1 = np.concatenate([dataX, dataY], axis=3).astype(np.float32)

data = {"inputT0": data0, "inputT1": data1}

@case_mark
def case_simple():
    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, shape0)
    inputT1 = tw.network.add_input("inputT1", trt.float32, [nB] + shape1 + [2])
    layer = tw.network.add_grid_sample(inputT0, inputT1)
    #layer.interpolation_mode = trt.InterpolationMode.LINEAR  # default value
    #layer.interpolation_mode = trt.InterpolationMode.NEAREST
    #layer.interpolation_mode = trt.InterpolationMode.CUBIC
    #layer.align_corners = False  # default value
    #layer.align_corners = True
    #layer.sample_mode = trt.SampleMode.FILL  # default value
    #layer.sample_mode = trt.SampleMode.DEFAULT  # the same as STRICT_BOUNDS, deprecated since TensorRT 8.5
    #layer.sample_mode = trt.SampleMode.STRICT_BOUNDS
    #layer.sample_mode = trt.SampleMode.WRAP
    #layer.sample_mode = trt.SampleMode.CLAMP
    #layer.sample_mode = trt.SampleMode.REFLECT

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using Grid sample layer
    case_simple()

    print("Finish")
