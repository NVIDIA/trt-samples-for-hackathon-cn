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

shape = [1, 1, 3, 3]
nB, nC, nH, nW = shape
nCOut, nHKernel, nWKernel = 1, 3, 3
data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1
data = {"inputT0": data}
w = np.power(10, range(4, -5, -1), dtype=np.float32).reshape(nCOut, nC, nHKernel, nWKernel)
w = np.ascontiguousarray(w)
b = np.zeros(nCOut, dtype=np.float32)
b = np.ascontiguousarray(b)

@case_mark
def case_simple():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_deconvolution_nd(tensor, nCOut, [nHKernel, nWKernel], trt.Weights(w), trt.Weights(b))

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_parameter():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_convolution_nd(tensor, 1, [1, 1], np.zeros(1, dtype=np.float32))
    layer.num_output_maps = nCOut
    layer.kernel_size_nd = [nHKernel, nWKernel]
    layer.kernel = trt.Weights(w)
    layer.bias = trt.Weights(b)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_stride_dilation_pad():
    nHStride, nWStride = 2, 2
    nHDilation, nWDilation = 2, 2
    nHPadding, nWPadding = 1, 1

    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_deconvolution_nd(tensor, nCOut, [nHKernel, nWKernel], trt.Weights(w), trt.Weights(b))
    # Modify stride, dilation and padding
    layer.stride_nd = [nHStride, nWStride]
    layer.dilation_nd = [nHDilation, nWDilation]
    # Priority: padding_mode > pre_padding = post_padding > padding_nd
    layer.padding_nd = [nHPadding, nWPadding]
    layer.pre_padding = [nHPadding, nWPadding]
    layer.post_padding = [nHPadding, nWPadding]
    layer.padding_mode = trt.PaddingMode.SAME_UPPER

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_group():
    nC_local = 2  # nC in this example is 2
    nGroup = 2
    nCOut, nHKernel, nWKernel = nGroup, 3, 3
    data_local = np.tile(data["inputT0"], [1, nC_local // nC, 1, 1])
    data_local = {"inputT0": data_local}
    w_local = np.concatenate([w, -w], 0)  # double the kernel as shape of [nGroup, nHKernel, nWKernel]
    w_local = np.ascontiguousarray(w_local)
    b = np.zeros(nCOut, dtype=np.float32)

    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data_local["inputT0"].shape)
    layer = tw.network.add_deconvolution_nd(tensor, nCOut, [nHKernel, nWKernel], trt.Weights(w_local), trt.Weights(b))
    layer.num_groups = nGroup

    tw.build([layer.get_output(0)])
    tw.setup(data_local)
    tw.infer()

@case_mark
def case_3d():
    nC_local = 2  # nC in this example is 2
    data_local = np.tile(data["inputT0"], [1, nC_local // nC, 1, 1]).reshape([nB, 1, nC_local, nH, nW])
    data_local = {"inputT0": data_local}
    w_local = np.concatenate([w, -w], 0)
    w_local = np.ascontiguousarray(w_local)

    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data_local["inputT0"].shape)  # dimension of input tensor must be 5 or more
    layer = tw.network.add_deconvolution_nd(tensor, nCOut, [nHKernel, nWKernel], trt.Weights(w_local), trt.Weights(b))

    tw.build([layer.get_output(0)])
    tw.setup(data_local)
    tw.infer()

def case_int8qdq():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer_q0_weight = tw.network.add_constant([], np.array([1], dtype=np.float32))
    layer_q1_weight = tw.network.add_constant([], np.array([1], dtype=np.float32))
    layer_weight = tw.network.add_constant(w.shape, trt.Weights(w))
    layer_q0 = tw.network.add_quantize(tensor, layer_q0_weight.get_output(0))
    layer_q0.axis = 0
    layer_dq0 = tw.network.add_dequantize(layer_q0.get_output(0), layer_q1_weight.get_output(0))
    layer_dq0.axis = 0
    layer_d1 = tw.network.add_quantize(layer_weight.get_output(0), layer_q0_weight.get_output(0))
    layer_d1.axis = 0
    layer_dq1 = tw.network.add_dequantize(layer_d1.get_output(0), layer_q1_weight.get_output(0))
    layer_dq1.axis = 0

    layer = tw.network.add_deconvolution_nd(layer_dq0.get_output(0), nCOut, [nHKernel, nWKernel], trt.Weights(), trt.Weights(np.ascontiguousarray(b)))  # weight as empty
    layer.set_input(1, layer_dq1.get_output(0))

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using deconvolution layer.
    case_simple()
    # Modify kernel weights and bias after adding the layer.
    case_parameter()
    # Modify deconvolution parameters.
    case_stride_dilation_pad()
    # A case of group deconvolution.
    case_group()
    # A case of 3D deconvolution.
    case_3d()
    # A case of INT8 deconvolution with weights from another layer.
    case_int8qdq()

    print("Finish")
