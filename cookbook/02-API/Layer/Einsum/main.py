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

import numpy as np
import tensorrt as trt

from tensorrt_cookbook import TRTWrapperV1, case_mark, datatype_np_to_trt

# Input data varies among examples, so we do not prepare it here

@case_mark
def case_contraction():
    data0 = np.arange(np.prod(12), dtype=np.float32).reshape(1, 3, 4)
    data1 = np.arange(np.prod(30), dtype=np.float32).reshape(2, 3, 5)
    data = {"tensor": data0, "tensor1": data1}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    layer = tw.network.add_einsum([tensor, tensor1], "ijk,pjr->ikpr")
    layer.equation = "ijk,pjr->ikpr"  # [Optional] Reset equation of computation later

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_transpose():
    data = {"tensor": np.arange(np.prod(12), dtype=np.float32).reshape(1, 3, 4)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_einsum([tensor], "ijk->jki")

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_sum_reduce():
    data = {"tensor": np.arange(np.prod(12), dtype=np.float32).reshape(1, 3, 4)}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_einsum([tensor], "ijk->ij")

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_dot_product():
    if True:
        shape0 = 1, 1, 4
        shape1 = 1, 1, 4
        equation = "ijk,pqk->"
    elif True:  # Alternative example 1
        shape0 = 1, 2, 4
        shape1 = 1, 3, 4
        equation = "ijk,pqk->"
    else:  # Alternative example 2
        shape0 = 1, 2, 4
        shape1 = 1, 3, 4
        equation = "ijk,pqk->j"
    data0 = np.arange(np.prod(shape0), dtype=np.float32).reshape(shape0)
    data1 = np.ones(np.prod(shape1), dtype=np.float32).reshape(shape1)
    data = {"tensor": data0, "tensor1": data1}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    layer = tw.network.add_einsum([tensor, tensor1], equation)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_matrix_multiplication():
    data0 = np.arange(np.prod(12), dtype=np.float32).reshape(2, 2, 3)
    data1 = np.ones(np.prod(24), dtype=np.float32).reshape(2, 3, 4)
    data = {"tensor": data0, "tensor1": data1}

    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    layer = tw.network.add_einsum([tensor, tensor1], "ijk,ikl->ijl")

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_multi_tensor_contraction():
    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", trt.float32, [1, 2, 3])
    tensor1 = tw.network.add_input("tensor1", trt.float32, [4, 3, 2])
    tensor2 = tw.network.add_input("tensor2", trt.float32, [4, 5])
    layer = tw.network.add_einsum([tensor, tensor1, tensor2], "abc,dcb,de->ae")

    try:
        tw.build([layer.get_output(0)])
    except Exception:
        pass

@case_mark
def case_diagnal():
    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", trt.float32, [1, 4, 4])
    layer = tw.network.add_einsum([tensor], "ijj->ij")

    try:
        tw.build([layer.get_output(0)])
    except Exception:
        pass

@case_mark
def case_ellipsis():
    tw = TRTWrapperV1()
    tensor = tw.network.add_input("tensor", trt.float32, [1, 3, 4])
    layer = tw.network.add_einsum([tensor], "...j->...j")

    try:
        tw.build([layer.get_output(0)])
    except Exception:
        pass

if __name__ == "__main__":
    # A simple case of contraction with einsum layer
    case_contraction()
    # Use einsum layer for transpose
    case_transpose()
    # Use einsum layer for reduce
    case_sum_reduce()
    # Use einsum layer for dot product
    case_dot_product()
    # Use einsum layer for matrix multiplication
    case_matrix_multiplication()
    # Use einsum layer for multiple matrix contraction (not supported yet)
    case_multi_tensor_contraction()
    # Use einsum layer for extracting diagnal elements (not supported yet)
    case_diagnal()
    # Use einsum layer for ellipsis (Not supported yet)
    case_ellipsis()

    print("Finish")
