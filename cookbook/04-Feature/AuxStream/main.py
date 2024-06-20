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
from utils import TRTWrapperV1

nGEMM = 10
nMKN = 128
shape = [1, 4, nMKN, nMKN]
data = {"inputT0": np.random.rand(np.prod(shape)).astype(np.float32).reshape(shape) * 2 - 1}

tw = TRTWrapperV1()
tw.config.max_aux_streams = 2

input_tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
tensor0 = input_tensor
tensor1 = input_tensor
for i in range(nGEMM):
    w = np.ascontiguousarray(np.random.rand(1, 1, nMKN, nMKN).astype(np.float32))
    constant_layer = tw.network.add_constant(w.shape, trt.Weights(w))
    layer = tw.network.add_matrix_multiply(tensor0, trt.MatrixOperation.NONE, constant_layer.get_output(0), trt.MatrixOperation.NONE)
    tensor0 = layer.get_output(0)

    w = np.ascontiguousarray(np.random.rand(1, 1, nMKN, nMKN).astype(np.float32))
    constant_layer = tw.network.add_constant(w.shape, trt.Weights(w))
    layer = tw.network.add_matrix_multiply(tensor1, trt.MatrixOperation.NONE, constant_layer.get_output(0), trt.MatrixOperation.NONE)
    tensor1 = layer.get_output(0)

tw.build([tensor0, tensor1])
tw.setup(data)
tw.infer(b_print_io=False, b_get_timeline=True)
