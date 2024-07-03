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

data = {"inputT0": np.ones([1, 3, 4, 5], dtype=np.float32)}
data1 = {"inputT0": np.ones([1, 3, 4, 5], dtype=np.float32), "inputT1": np.ones([1, 3], dtype=np.float32)}
data2 = {"inputT0": np.ones([1, 3, 4, 5], dtype=np.float32), "inputT1": np.ones([1, 4], dtype=np.float32)}

@case_mark
def case_buildtime_check(can_pass):
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer1 = tw.network.add_shape(tensor)
    layer2 = tw.network.add_slice(layer1.get_output(0), [3], [1], [1])
    if can_pass:  # assert(inputT0.shape[3] == 5), no error
        layerConstant = tw.network.add_constant([1], np.array([5], dtype=np.int64))
    else:  # assert(inputT0.shape[3] == 4), error at build time
        layerConstant = tw.network.add_constant([1], np.array([4], dtype=np.int64))
    layer3 = tw.network.add_elementwise(layer2.get_output(0), layerConstant.get_output(0), trt.ElementWiseOperation.EQUAL)
    layer4 = tw.network.add_identity(layer3.get_output(0))
    layer4.get_output(0).dtype = trt.bool
    # assert layer seems no use but actually works
    assert_layer = tw.network.add_assertion(layer4.get_output(0), "inputT0.shape[3] != 5")
    assert_layer.message += " [Something else you want to say]"  # optional, modify the assert message

    try:
        tw.build([layer4.get_output(0)])
    except Exception:
        pass

@case_mark
def case_runtime_check(can_pass):
    tw = TRTWrapperV1()

    shape = list(data["inputT0"].shape)
    inputT0 = tw.network.add_input("inputT0", trt.float32, [-1, -1] + shape[2:])
    tw.profile.set_shape(inputT0.name, [1, 1] + shape[2:], shape, [2, 6] + shape[2:])
    inputT1 = tw.network.add_input("inputT1", trt.float32, (-1, -1))
    tw.profile.set_shape(inputT1.name, [1, 1], shape[:2], [2, 6])
    tw.config.add_optimization_profile(tw.profile)

    layer1 = tw.network.add_shape(inputT0)
    layer2 = tw.network.add_slice(layer1.get_output(0), [1], [1], [1])
    layer3 = tw.network.add_shape(inputT1)
    layer4 = tw.network.add_slice(layer3.get_output(0), [1], [1], [1])
    # assert(inputT0.shape[1] == inputT1.shape[1])
    layer5 = tw.network.add_elementwise(layer2.get_output(0), layer4.get_output(0), trt.ElementWiseOperation.EQUAL)

    # Assert layer seems no use afterwards but actually works
    assert_layer = tw.network.add_assertion(layer5.get_output(0), "inputT0.shape[1] != inputT1.shape[1]")

    layer6 = tw.network.add_cast(layer5.get_output(0), trt.int32)

    tw.build([layer6.get_output(0)])
    try:
        if can_pass:
            tw.setup(data1)
        else:
            tw.setup(data2)  # assert error raised during call of context.set_input_shape
    except Exception:
        pass

if __name__ == "__main__":
    # Check during network building process.
    case_buildtime_check(True)
    case_buildtime_check(False)

    # Check during inference process.
    case_runtime_check(True)
    case_runtime_check(False)

    print("Finish")
