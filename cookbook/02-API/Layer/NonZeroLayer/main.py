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
from utils import TRTWrapperDDS, case_mark

shape = [3, 4, 5]
data = np.zeros(shape).astype(np.float32)
data[0, 0, 1] = 1
data[0, 2, 3] = 2
data[0, 3, 4] = 3
data[1, 1, 0] = 4
data[1, 1, 1] = 5
data[1, 1, 2] = 6
data[1, 1, 3] = 7
data[1, 1, 4] = 8
data[2, 0, 1] = 9
data[2, 1, 1] = 10
data[2, 2, 1] = 11
data[2, 3, 1] = 12
data = {"inputT0": data}

@case_mark
def case_simple():
    tw = TRTWrapperDDS()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    data_layer = tw.network.add_non_zero(tensor)
    data_tensor = data_layer.get_output(0)
    data_tensor.name = "output_data"

    tw.build([data_tensor])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    case_simple()

    print("Finish")
