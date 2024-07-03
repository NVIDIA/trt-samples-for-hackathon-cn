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

nB, nS, nH = 3, 4, 5
data0 = np.arange(nB).reshape(nB, 1, 1) * 100
data0 = data0 + np.arange(nS).reshape(1, nS, 1) * 10  # "+=" operator does not work because output shape is not the same as input shape
data0 = data0 + np.arange(nH).reshape(1, 1, nH)
data0 = data0.astype(np.float32)
data = {"inputT0": data0}

@case_mark
def case_simple():
    data["inputT1"] = np.array([4, 3, 2, 1], dtype=np.int32)

    tw = TRTWrapperV1()

    tensor0 = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    tensor1 = tw.network.add_input("inputT1", trt.int32, data["inputT1"].shape)
    layer = tw.network.add_reverse_sequence(tensor0, tensor1)
    print(f"{layer.batch_axis = }, {layer.sequence_axis = }")

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_batch_prior():
    data["inputT1"] = np.array([3, 2, 1], dtype=np.int32)

    tw = TRTWrapperV1()

    tensor0 = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    tensor1 = tw.network.add_input("inputT1", trt.int32, data["inputT1"].shape)
    layer = tw.network.add_reverse_sequence(tensor0, tensor1)
    layer.batch_axis = 0
    layer.sequence_axis = 1
    print(f"{layer.batch_axis = }, {layer.sequence_axis = }")

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case to reverse the input tensor with dimension sequence 0, dimension batch 1 (default configuration).
    case_simple()
    # A case to reverse the input tensor with dimension batch 0, dimension sequence 1.
    case_batch_prior()

    print("Finish")
