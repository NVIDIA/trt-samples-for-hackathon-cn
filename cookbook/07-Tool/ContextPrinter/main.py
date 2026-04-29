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

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import (TRTWrapperV1, cookbook_path, load_mnist_network_trt, print_context_io_information)
from tensorrt_cookbook import (TRTWrapperV1, TRTWrapperV2, case_mark, datatype_cast)

@case_mark
def case_simple():
    data = {"x": np.load(cookbook_path("00-Data", "data", "InferenceData.npy"))}

    tw = TRTWrapperV1()

    load_mnist_network_trt(tw)
    tw.build()

    tw.setup(data)

    # Print shape of all input / output tensors in the context after setting input shape
    print_context_io_information(tw.context)

@case_mark
def case_dds_and_shape_input():

    data = {
        "tensor": np.random.permutation(np.arange(60, dtype=np.float32)).reshape(3, 4, 5),
        "tensor1": np.array([2], dtype=np.int32),  # One more shape input tensor
    }

    tw = TRTWrapperV2()  # Use Data-Dependent-Shape and Shape-Input mode at the same time

    tensor = tw.network.add_input("tensor", datatype_cast(data["tensor"].dtype, "trt"), [-1, -1, -1])
    tensor1 = tw.network.add_input("tensor1", datatype_cast(data["tensor1"].dtype, "trt"), [])
    tw.profile.set_shape(tensor.name, [1, 2, 1], data["tensor"].shape, data["tensor"].shape)
    tw.profile.set_shape_input(tensor1.name, [1], [2], [3])
    tw.config.add_optimization_profile(tw.profile)

    layer = tw.network.add_topk(tensor, trt.TopKOperation.MAX, 1, 1 << 1)
    layer.set_input(1, tensor1)

    tw.build([layer.get_output(0), layer.get_output(1)])
    tw.setup(data)

    print_context_io_information(tw.context)
    tw.infer()

if __name__ == "__main__":
    # Use a network of MNIST
    case_simple()

    # Use a network with Data-Dependent-Shape and Shape-Input
    case_dds_and_shape_input()

    print("Finish")
