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

data = {
    "q": np.arange(96, dtype=np.float32).reshape(1, 4, 3, 8) / 96,
    "k": np.ones(96, dtype=np.float32).reshape(1, 4, 3, 8) / 96,
    "v": -np.arange(96, dtype=np.float32).reshape(1, 4, 3, 8) / 96,
}

@case_mark
def case_simple():

    tw = TRTWrapperV1()
    tw.network = tw.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    tensor_q = tw.network.add_input("q", datatype_np_to_trt(data["q"].dtype), data["q"].shape)
    tensor_k = tw.network.add_input("k", datatype_np_to_trt(data["k"].dtype), data["k"].shape)
    tensor_v = tw.network.add_input("v", datatype_np_to_trt(data["v"].dtype), data["v"].shape)

    attention = tw.network.add_attention(tensor_q, tensor_k, tensor_v, trt.AttentionNormalizationOp.SOFTMAX, False)
    attention.name = "A cute Attention structure"

    attention.norm_op = trt.AttentionNormalizationOp.SOFTMAX  # [Optional] The normalization operator for qk
    attention.causal = False  # [Optional] Whether to use causal mask
    attention.mask = tw.network.add_constant([1, 4, 3, 3], np.ones([1, 4, 3, 3], dtype=bool)).get_output(0)  # [Optional] Cusotmerized mask when attention.causal is False
    attention.decomposable = True  # Allow to use fallback non-fused kernels if no fused kernel is available, default value: False
    print(f"{attention.num_inputs = }")
    print(f"{attention.num_outputs = }")
    """
    attention.normalization_quantize_scale = # ITensor The quantization scale for the attention normalization output.
    attention.normalization_quantize_to_type – DataType The datatype the attention normalization is quantized to.
    """
    output_tensor = attention.get_output(0)
    output_tensor.name = 'attention_output'
    tw.build([output_tensor])
    tw.setup(data)
    tw.infer()

    shape_q = data["q"].shape
    num_head = shape_q[1]
    head_width = shape_q[3]
    q = data["q"].transpose(0, 2, 1, 3).reshape(shape_q[0], shape_q[2], -1)

    shape_k = data["k"].shape
    k = data["k"].transpose(0, 1, 3, 2).reshape(shape_k[0], -1, shape_k[2])

    s = np.matmul(q, k) / np.sqrt(num_head * head_width)
    s = np.exp(s - np.max(s)) / np.sum(np.exp(s - np.max(s)), axis=1)

    shape_v = data["v"].shape
    v = data["v"].transpose(0, 2, 1, 3).reshape(shape_v[0], shape_v[2], -1)

    o = np.matmul(s, v)
    o = o.reshape(o.shape[0], o.shape[1], num_head, head_width).transpose(0, 2, 1, 3)

    diff = o - tw.buffer['attention_output'][0]
    print(f"{np.max(diff) = }, {np.min(diff) = }")

@case_mark
def case_quantization():

    tw = TRTWrapperV1()
    tw.network = tw.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    tensor_q = tw.network.add_input("q", datatype_np_to_trt(data["q"].dtype), data["q"].shape)
    tensor_k = tw.network.add_input("k", datatype_np_to_trt(data["k"].dtype), data["k"].shape)
    tensor_v = tw.network.add_input("v", datatype_np_to_trt(data["v"].dtype), data["v"].shape)

    attention = tw.network.add_attention(tensor_q, tensor_k, tensor_v, trt.AttentionNormalizationOp.SOFTMAX, False)
    attention.decomposable = True

    # attention.normalization_quantize_scale =
    # attention.normalization_quantize_to_type =

    output_tensor = attention.get_output(0)
    output_tensor.name = 'attention_output'
    tw.build([output_tensor])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using attention structure
    case_simple()
    # A quantization attention
    case_quantization()  # not finished

    print("Finish")
