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
from tensorrt_cookbook import TRTWrapperV1, case_mark

n_gemm = 10
n_branch = 10
n_mkn = 64
shape = [1, 4, n_mkn, n_mkn]
data = {"inputT0": np.random.rand(np.prod(shape)).astype(np.float32).reshape(shape) * 2 - 1}

@case_mark
def case_(n_max_aux_streams):
    tw = TRTWrapperV1()
    tw.config.max_aux_streams = n_max_aux_streams

    input_tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    tensor_list = [input_tensor] * n_branch
    for i in range(n_gemm):
        for j in range(n_branch):
            w = np.ascontiguousarray(np.random.rand(1, 1, n_mkn, n_mkn).astype(np.float32))
            constant_layer = tw.network.add_constant(w.shape, trt.Weights(w))
            layer = tw.network.add_matrix_multiply(tensor_list[j], trt.MatrixOperation.NONE, constant_layer.get_output(0), trt.MatrixOperation.NONE)
            layer = tw.network.add_activation(layer.get_output(0), trt.ActivationType.RELU)
            tensor_list[j] = layer.get_output(0)

    tw.build(tensor_list)
    tw.setup(data)
    tw.infer(b_print_io=False)

if __name__ == "__main__":
    #case_(0)
    #case_(1)
    case_(2)

    print("Finish")
