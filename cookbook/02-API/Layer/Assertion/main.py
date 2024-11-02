#
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
#

import numpy as np
import tensorrt as trt

from tensorrt_cookbook import TRTWrapperV1, case_mark, datatype_np_to_trt

data = {"tensor": np.ones([3, 4, 5], dtype=np.float32)}
data1 = {"tensor": np.ones([3, 4, 5], dtype=np.float32), "tensor1": np.ones([3, 4], dtype=np.float32)}
data2 = {"tensor": np.ones([3, 4, 5], dtype=np.float32), "tensor1": np.ones([3, 5], dtype=np.float32)}

@case_mark
def case_buildtime_check(b_can_pass):
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer1 = tw.network.add_shape(tensor)
    layer2 = tw.network.add_slice(layer1.get_output(0), [2], [1], [1])
    if b_can_pass:  # assert(tensor.shape[2] == 5), OK
        layerConstant = tw.network.add_constant([1], np.array([5], dtype=np.int64))
    else:  # assert(tensor.shape[2] == 4), error at buildtime
        layerConstant = tw.network.add_constant([1], np.array([4], dtype=np.int64))
    layer3 = tw.network.add_elementwise(layer2.get_output(0), layerConstant.get_output(0), trt.ElementWiseOperation.EQUAL)
    layer4 = tw.network.add_identity(layer3.get_output(0))
    layer4.get_output(0).dtype = trt.bool
    # Assert layer seems no use but actually works
    layer = tw.network.add_assertion(layer4.get_output(0), "tensor.shape[2] != 5")
    layer.message += " [Something else you want to say]"  # [Optional] Reset assert message later

    try:
        tw.build([layer4.get_output(0)])  # Do not mark assert layer since it has no output tensor
    except Exception:
        pass

@case_mark
def case_runtime_check(b_can_pass):
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data1["tensor"].dtype), [-1, -1, -1])
    tw.profile.set_shape(tensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data1["tensor1"].dtype), [-1, -1])
    tw.profile.set_shape(tensor1.name, [1, 1], [3, 4], [6, 8])
    tw.config.add_optimization_profile(tw.profile)
    layer1 = tw.network.add_shape(tensor)
    layer2 = tw.network.add_slice(layer1.get_output(0), [1], [1], [1])
    layer3 = tw.network.add_shape(tensor1)
    layer4 = tw.network.add_slice(layer3.get_output(0), [1], [1], [1])
    # assert(tensor.shape[0] == tensor1.shape[0])
    layer5 = tw.network.add_elementwise(layer2.get_output(0), layer4.get_output(0), trt.ElementWiseOperation.EQUAL)
    # Assert layer seems no use but actually works
    layer = tw.network.add_assertion(layer5.get_output(0), "tensor.shape[1] != tensor1.shape[1]")
    layer6 = tw.network.add_cast(layer5.get_output(0), trt.int32)

    tw.build([layer6.get_output(0)])
    try:
        if b_can_pass:
            tw.setup(data1)
        else:
            tw.setup(data2)  # Assert error raised during call of `context.set_input_shape`
    except Exception:
        pass

if __name__ == "__main__":
    # Check during buildtime
    case_buildtime_check(True)
    case_buildtime_check(False)
    # Check during runtime
    case_runtime_check(True)
    case_runtime_check(False)

    print("Finish")
