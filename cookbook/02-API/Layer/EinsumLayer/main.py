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

# input data varies among examples, so we do not prepare them here

@case_mark
def case_contraction():
    shape0 = 1, 3, 4
    shape1 = 2, 3, 5
    data0 = np.arange(np.prod(shape0), dtype=np.float32).reshape(shape0)
    data1 = np.arange(np.prod(shape1), dtype=np.float32).reshape(shape1)
    data = {"inputT0": data0, "inputT1": data1}

    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, shape0)
    inputT1 = tw.network.add_input("inputT1", trt.float32, shape1)
    layer = tw.network.add_einsum([inputT0, inputT1], "ijk,pjr->ikpr")

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_equation():
    shape0 = 1, 3, 4
    shape1 = 2, 3, 5
    data0 = np.arange(np.prod(shape0), dtype=np.float32).reshape(shape0)
    data1 = np.arange(np.prod(shape1), dtype=np.float32).reshape(shape1)
    data = {"inputT0": data0, "inputT1": data1}

    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, shape0)
    inputT1 = tw.network.add_input("inputT1", trt.float32, shape1)
    layer = tw.network.add_einsum([inputT0, inputT1], "ijk,pqr->ijkpqr")  # can not be empty equation
    layer.equation = "ijk,pjr->ikpr"

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_transpose():
    shape0 = 1, 3, 4
    data0 = np.arange(np.prod(shape0), dtype=np.float32).reshape(shape0)
    data = {"inputT0": data0}

    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, shape0)
    layer = tw.network.add_einsum([inputT0], "ijk->jki")

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_sum_reduce():
    shape0 = 1, 3, 4
    data0 = np.arange(np.prod(shape0), dtype=np.float32).reshape(shape0)
    data = {"inputT0": data0}

    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, shape0)
    layer = tw.network.add_einsum([inputT0], "ijk->ij")

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_dot_product():
    if True:
        shape0 = 1, 1, 4
        shape1 = 1, 1, 4
        equation = "ijk,pqk->"
    elif True:  # substitutive example 1
        shape0 = 1, 2, 4
        shape1 = 1, 3, 4
        equation = "ijk,pqk->"
    else:  # substitutive example 2
        shape0 = 1, 2, 4
        shape1 = 1, 3, 4
        equation = "ijk,pqk->j"
    data0 = np.arange(np.prod(shape0), dtype=np.float32).reshape(shape0)
    data1 = np.ones(np.prod(shape1), dtype=np.float32).reshape(shape1)
    data = {"inputT0": data0, "inputT1": data1}

    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, shape0)
    inputT1 = tw.network.add_input("inputT1", trt.float32, shape1)
    layer = tw.network.add_einsum([inputT0, inputT1], equation)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_matrix_multiplication():
    shape0 = 2, 2, 3
    shape1 = 2, 3, 4
    data0 = np.arange(np.prod(shape0), dtype=np.float32).reshape(shape0)
    data1 = np.ones(np.prod(shape1), dtype=np.float32).reshape(shape1)
    data = {"inputT0": data0, "inputT1": data1}

    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, shape0)
    inputT1 = tw.network.add_input("inputT1", trt.float32, shape1)
    layer = tw.network.add_einsum([inputT0, inputT1], "ijk,ikl->ijl")

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_multi_tensor_contraction():
    shape0 = 1, 2, 3
    shape1 = 4, 3, 2
    shape2 = 4, 5

    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, shape0)
    inputT1 = tw.network.add_input("inputT1", trt.float32, shape1)
    inputT2 = tw.network.add_input("inputT2", trt.float32, shape2)
    layer = tw.network.add_einsum([inputT0, inputT1, inputT2], "abc,dcb,de->ae")

    try:
        tw.build([layer.get_output(0)])
    except Exception:
        pass

@case_mark
def case_diagnal():
    shape = 1, 4, 4

    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, shape)
    layer = tw.network.add_einsum([inputT0], "ijj->ij")

    try:
        tw.build([layer.get_output(0)])
    except Exception:
        pass

@case_mark
def case_ellipsis():
    shape = 1, 3, 4

    tw = TRTWrapperV1()

    inputT0 = tw.network.add_input("inputT0", trt.float32, shape)
    layer = tw.network.add_einsum([inputT0], "...j->...j")

    try:
        tw.build([layer.get_output(0)])
    except Exception:
        pass

if __name__ == "__main__":
    case_contraction()
    case_equation()
    case_transpose()
    case_sum_reduce()
    case_dot_product()
    case_matrix_multiplication()
    case_multi_tensor_contraction()
    case_diagnal()
    case_ellipsis()

    print("Finish")
