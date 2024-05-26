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

shape = [1, 3, 4, 5]
data0 = np.full(shape, 2, dtype=np.float32)
data1 = np.full(shape, 3, dtype=np.float32)
data = {"inputT0": data0, "inputT1": data1}

@case_mark
def case_simple():
    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, shape)
    inputT1 = tw.network.add_input("inputT1", trt.float32, shape)
    layer = tw.network.add_elementwise(inputT0, inputT1, trt.ElementWiseOperation.POW)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_op():
    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, shape)
    inputT1 = tw.network.add_input("inputT1", trt.float32, shape)
    layer = tw.network.add_elementwise(inputT0, inputT1, trt.ElementWiseOperation.POW)
    layer.op = trt.ElementWiseOperation.SUM

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_broadcast():
    nB, nC, nH, nW = shape
    data0_local = np.full([nB, nC, 1, nW], 1, dtype=np.float32)
    data1_local = np.full([nB, 1, nH, 1], 2, dtype=np.float32)
    data_local = {"inputT0": data0_local, "inputT1": data1_local}

    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, data_local["inputT0"].shape)
    inputT1 = tw.network.add_input("inputT1", trt.float32, data_local["inputT1"].shape)
    layer = tw.network.add_elementwise(inputT0, inputT1, trt.ElementWiseOperation.SUM)

    tw.build([layer.get_output(0)])
    tw.setup(data_local)
    tw.infer()

if __name__ == "__main__":
    case_simple()
    case_op()
    case_broadcast()

    print("Finish")
