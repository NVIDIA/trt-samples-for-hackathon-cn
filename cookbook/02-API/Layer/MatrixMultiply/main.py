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

data = np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5)
data = {"tensor": data}

@case_mark
def case_simple():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    weight_shape = data["tensor"].transpose(0, 1, 3, 2).shape
    layer_weight = tw.network.add_constant(weight_shape, trt.Weights(np.ascontiguousarray(np.ones(weight_shape, dtype=np.float32))))
    layer = tw.network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, layer_weight.get_output(0), trt.MatrixOperation.NONE)
    layer.op0 = trt.MatrixOperation.NONE  # [Optional] Reset op later
    layer.op1 = trt.MatrixOperation.NONE  # [Optional] Reset op later

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_transpose():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    weight_shape = data["tensor"].shape  # No transpose compared with `case_simple`
    layer_weight = tw.network.add_constant(weight_shape, trt.Weights(np.ascontiguousarray(np.ones(weight_shape, dtype=np.float32))))
    layer = tw.network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, layer_weight.get_output(0), trt.MatrixOperation.TRANSPOSE)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_vector():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    weight_shape = data["tensor"].transpose(0, 1, 3, 2).shape[:-1]  # One less dimension compared with `case_simple`
    layer_weight = tw.network.add_constant(weight_shape, trt.Weights(np.ascontiguousarray(np.ones(weight_shape, dtype=np.float32))))
    layer = tw.network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, layer_weight.get_output(0), trt.MatrixOperation.VECTOR)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_broadcast():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    weight_shape = (1, 1) + data["tensor"].transpose(0, 1, 3, 2).shape[-2:]  # [1,1,5,4]
    layer_weight = tw.network.add_constant(weight_shape, trt.Weights(np.ascontiguousarray(np.ones(weight_shape, dtype=np.float32))))
    layer = tw.network.add_matrix_multiply(tensor, trt.MatrixOperation.NONE, layer_weight.get_output(0), trt.MatrixOperation.NONE)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using matrix multiplication layer
    case_simple()
    # Use transpose before matrix multiplication
    case_transpose()
    # Use vector to multiplication
    case_vector()
    # Use broadcast operation before multiplication
    case_broadcast()

    print("Finish")
