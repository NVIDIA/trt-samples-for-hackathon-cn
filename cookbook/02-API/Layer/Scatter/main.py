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

import sys

import numpy as np
import tensorrt as trt

from tensorrt_cookbook import TRTWrapperV1, case_mark, check_array, datatype_np_to_trt


@case_mark
def case_element_mode():
    shape = 1, 3, 4, 5
    data0 = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    data1 = np.tile(np.arange(shape[2], dtype=np.int32), [shape[0], shape[1], 1, shape[3]]).reshape(shape)
    data2 = - data0
    data = {"tensor": data0, "tensor1": data1, "tensor2": data2}
    scatter_axis = 2

    def scatter_cpu_element(data0, data1, data2, axis):
        nB, nC, nH, nW = data0.shape
        output = data0
        for n in range(nB):
            for c in range(nC):
                for h in range(nH):
                    for w in range(nW):
                        match(axis):
                            case 0:
                                output[data1[n, c, h, w], c, h, w] = data2[n, c, h, w]
                            case 1:
                                output[n, data1[n, c, h, w], h, w] = data2[n, c, h, w]
                            case 2:
                                output[n, c, data1[n, c, h, w], w] = data2[n, c, h, w]
                            case 3:
                                output[n, c, h, data1[n, c, h, w]] = data2[n, c, h, w]
                            case _:
                                print("Fail scattering at axis %d " % axis)
                                return None
                        #print(f"<{n},{c},{h},{w}>->{data2[n,c,h,w]}")
                        #print(output)
        return output

    tw = TRTWrapperV1()

    tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    tensor2 = tw.network.add_input("tensor2", datatype_np_to_trt(data["tensor2"].dtype), data["tensor2"].shape)
    layer = tw.network.add_scatter(tensor0, tensor1, tensor2, trt.ScatterMode.ELEMENT)
    layer.axis = scatter_axis
    layer.get_output(0).name = "outputT0"

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

    res_cpu = scatter_cpu_element(tw.buffer["tensor"][0], tw.buffer["tensor1"][0], tw.buffer["tensor2"][0], scatter_axis)
    check_array(tw.buffer["outputT0"][0], res_cpu, weak=False)

@case_mark
def case_nd_mode():
    shape = [2, 3, 4, 5]
    data0 = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    data1 = np.array([[[0, 2, 1, 1], [1, 0, 3, 2], [0, 1, 2, 3]], [[1, 2, 1, 1], [0, 0, 3, 2], [1, 1, 2, 3]]], dtype=np.int32)
    data2 = -np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape[0], shape[1])
    data = {"tensor": data0, "tensor1": data1, "tensor2": data2}

    def scatter_cpu_nd(data0, data1, data2):
        output = data0
        for i in range(data1.shape[0]):
            for j in range(data1.shape[1]):
                #print(f"{i=},{j=},index={data1[i,j]},updateValue={data2[i, j]}")
                output[data1[i, j][0], data1[i, j][1], data1[i, j][2], data1[i, j][3]] = data2[i, j]
        return output

    tw = TRTWrapperV1()

    tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    tensor2 = tw.network.add_input("tensor2", datatype_np_to_trt(data["tensor2"].dtype), data["tensor2"].shape)
    layer = tw.network.add_scatter(tensor0, tensor1, tensor2, trt.ScatterMode.ND)
    layer.get_output(0).name = "outputT0"

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

    res_cpu = scatter_cpu_nd(data0, data1, data2)
    check_array(tw.buffer["outputT0"][0], res_cpu, weak=False)

@case_mark
def case_nd_mode_2():
    shape = [2, 3, 4, 5]
    data0 = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    data1 = np.array([[0, 2, 1], [1, 0, 3], [0, 1, 2], [1, 2, 1], [0, 0, 3], [1, 1, 2]], dtype=np.int32)
    data2 = -np.arange(6 * 5, dtype=np.float32).reshape(6, 5)
    data = {"tensor": data0, "tensor1": data1, "tensor2": data2}

    def scatter_cpu_nd(data0, data1, data2):
        output = data0
        for i in range(data1.shape[0]):
            #print(f"{i=},index={data1[i]},updateValue={data2[i]}")
            output[data1[i][0], data1[i][1], data1[i][2]] = data2[i]
        return output

    tw = TRTWrapperV1()

    tensor0 = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    tensor1 = tw.network.add_input("tensor1", datatype_np_to_trt(data["tensor1"].dtype), data["tensor1"].shape)
    tensor2 = tw.network.add_input("tensor2", datatype_np_to_trt(data["tensor2"].dtype), data["tensor2"].shape)
    layer = tw.network.add_scatter(tensor0, tensor1, tensor2, trt.ScatterMode.ND)
    layer.get_output(0).name = "outputT0"

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

    res_cpu = scatter_cpu_nd(data0, data1, data2)
    check_array(tw.buffer["outputT0"][0], res_cpu, weak=False)

if __name__ == "__main__":
    # A simple case of using layer
    case_element_mode()
    #
    case_nd_mode()
    #
    case_nd_mode_2()

    print("Finish")
