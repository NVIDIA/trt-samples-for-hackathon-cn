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

onnx_file = "./model-labeled.onnx"

tw = TRTWrapperV1()
parser = trt.OnnxParser(tw.network, tw.logger)
with open(onnx_file, "rb") as model:
    parser.parse(model.read())

inputT0 = tw.network.get_input(0)
tw.profile.set_shape(inputT0.name, [1, 1, 1], [4, 1, 1], [8, 1, 1])
inputT1 = tw.network.get_input(1)
tw.profile.set_shape(inputT1.name, [1, 1], [4, 1], [8, 1])
tw.config.add_optimization_profile(tw.profile)

tw.build()

@case_mark
def case_correct():
    input_data = {"inputT0": np.zeros([4, 1, 1], dtype=np.float32), "inputT1": np.zeros([4, 1], dtype=np.float32)}
    tw.setup(input_data)
    return

@case_mark
def case_incorrect():
    input_data = {"inputT0": np.zeros([4, 1, 1], dtype=np.float32), "inputT1": np.zeros([5, 1], dtype=np.float32)}
    try:
        tw.setup(input_data)
    except:
        print("Length of the first dimension of two input tensors is different")
    return

if __name__ == "__main__":
    case_correct()
    case_incorrect()

    print("Finish")
