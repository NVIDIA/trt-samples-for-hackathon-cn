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

import tensorrt as trt
import numpy as np
sys.path.append("/trtcookbook/include")
from utils import TRTWrapperV1, case_mark

shape = [1, 3, 4, 5]
data = {"inputT0": np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1}
data_v2 = {"inputT0": data["inputT0"] - 1}

@case_mark
def case_simple():
    """
    # Behave like:
    if inputT0.reshape(-1)[0] != 0:
        return inputT0 * 2
    else:
        return inputT0
    """
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    # Extract the scalar first element
    layer_1 = tw.network.add_shuffle(tensor)
    layer_1.reshape_dims = [-1]
    layer_2 = tw.network.add_slice(layer_1.get_output(0), [0], [1], [1])
    layer_3 = tw.network.add_shuffle(layer_2.get_output(0))
    layer_3.reshape_dims = []
    layer_4 = tw.network.add_identity(layer_3.get_output(0))
    layer_4.set_output_type(0, trt.bool)
    layer_4.get_output(0).dtype = trt.bool
    
    if_structure = tw.network.add_if_conditional()
    layer_input = if_structure.add_input(tensor)
    layer_condition = if_structure.set_condition(layer_4.get_output(0))  # the condition tensor must be BOOL data type with 0 dimension

    # branch of condition is true
    layer_true = tw.network.add_elementwise(layer_input.get_output(0), layer_input.get_output(0), trt.ElementWiseOperation.SUM)

    # branch of condition is false
    layer_false = tw.network.add_identity(layer_input.get_output(0))

    layer_output = if_structure.add_output(layer_true.get_output(0), layer_false.get_output(0))

    tw.build([layer_output.get_output(0)])
    tw.setup(data)
    tw.infer()
    
    tw.setup(data_v2)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using layer
    case_simple()

    print("Finish")
