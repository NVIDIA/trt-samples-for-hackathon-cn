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

data = {"inputT0": np.arange(np.prod(60), dtype=np.float32).reshape(3, 4, 5)}

@case_mark
def case_simple():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_concatenation([tensor, tensor])

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_axis():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_concatenation([tensor, tensor])
    layer.axis = 2  # modify the axis to concatenate

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of concatenate two tensoers together.
    case_simple()
    # Modify the axis of concatenate axis after adding the layer.
    case_axis()

    print("Finish")
