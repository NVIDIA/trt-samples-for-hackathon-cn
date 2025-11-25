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

from time import time_ns

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

from tensorrt_cookbook import TRTWrapperV1, case_mark

shape = [4, 1024, 64]
data = {"inputT0": np.random.rand(*shape).reshape(shape).astype(np.float32) * 2 - 1}
n_warmup = 10
n_iteration = 30

@case_mark
def case_normal(n_level):
    print(f"Test Level = {n_level}")

    tw = TRTWrapperV1()
    tw.config.builder_optimization_level = n_level

    tensor = tw.network.add_input("inputT0", trt.float32, [-1] + shape[1:])
    tw.profile.set_shape(tensor.name, [1] + shape[1:], shape, [16] + shape[1:])
    tw.config.add_optimization_profile(tw.profile)

    # We build a "complex" network to see the performance differences
    for i in range(64, 128):
        w = np.random.rand(1, i, i + 1).astype(np.float32)
        b = np.random.rand(1, 1, i + 1).astype(np.float32)
        layer_weight = tw.network.add_constant(w.shape, trt.Weights(np.ascontiguousarray(w)))
        layer = tw.network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, layer_weight.get_output(0), trt.MatrixOperation.NONE)
        layer_bias = tw.network.add_constant(b.shape, trt.Weights(np.ascontiguousarray(b)))
        layer = tw.network.add_elementwise(layer.get_output(0), layer_bias.get_output(0), trt.ElementWiseOperation.SUM)
        layer = tw.network.add_activation(layer.get_output(0), trt.ActivationType.RELU)
        tensor = layer.get_output(0)

    t0 = time_ns()
    tw.build([tensor])
    t1 = time_ns()
    print(f"Time of building: {(t1 - t0) / (10 ** 6)} ms")

    tw.setup(data)

    # We skip memory copy and just care about the enqueue part

    # warming up
    for _ in range(n_warmup):
        tw.context.execute_async_v3(0)

    t0 = time_ns()
    for _ in range(n_iteration):
        tw.context.execute_async_v3(0)
    cudart.cudaDeviceSynchronize()
    t1 = time_ns()
    print(f"Time of inference: {(t1 - t0) / (10 ** 6)} ms")

if __name__ == "__main__":
    case_normal(0)
    case_normal(1)
    case_normal(2)
    case_normal(3)
    case_normal(4)
    case_normal(5)

    print("Finish")
