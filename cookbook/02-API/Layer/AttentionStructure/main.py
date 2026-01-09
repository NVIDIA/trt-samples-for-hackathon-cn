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
from tensorrt_cookbook import TRTWrapperV1, case_mark, datatype_np_to_trt, check_array

np.random.seed(31193)

@case_mark
def case_simple():
    nBS, nHead, nSLq, nHeadWidth = 1, 4, 3, 8
    nSLkv = 3
    data = {
        "q": np.random.rand(np.prod([nBS, nHead, nSLq, nHeadWidth])).astype(np.float32).reshape([nBS, nHead, nSLq, nHeadWidth]),
        "k": np.random.rand(np.prod([nBS, nHead, nSLkv, nHeadWidth])).astype(np.float32).reshape([nBS, nHead, nSLkv, nHeadWidth]),
        "v": -np.random.rand(np.prod([nBS, nHead, nSLkv, nHeadWidth])).astype(np.float32).reshape([nBS, nHead, nSLkv, nHeadWidth]),
    }

    tw = TRTWrapperV1()
    tw.network = tw.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    tensor_q = tw.network.add_input("q", datatype_np_to_trt(data["q"].dtype), data["q"].shape)
    tensor_k = tw.network.add_input("k", datatype_np_to_trt(data["k"].dtype), data["k"].shape)
    tensor_v = tw.network.add_input("v", datatype_np_to_trt(data["v"].dtype), data["v"].shape)

    attention = tw.network.add_attention(tensor_q, tensor_k, tensor_v, trt.AttentionNormalizationOp.SOFTMAX, False)
    attention.name = "A cute Attention structure"

    attention.norm_op = trt.AttentionNormalizationOp.SOFTMAX  # [Optional] The normalization operator for qk
    attention.decomposable = True  # Allow to use fallback non-fused kernels if no fused kernel is available, default value: False
    attention.causal = False  # [Optional] Whether to use causal mask
    print(f"{attention.num_inputs = }")
    print(f"{attention.num_outputs = }")

    output_tensor = attention.get_output(0)
    output_tensor.name = 'attention_output'
    tw.build([output_tensor])
    tw.setup(data)
    tw.infer()

    s = np.matmul(data["q"], data["k"].transpose(0, 1, 3, 2))
    s = np.exp(s - np.max(s)) / np.sum(np.exp(s - np.max(s)), axis=-1, keepdims=True)
    o = np.matmul(s, data["v"])

    check_array(tw.buffer['attention_output'][0], o, True)

@case_mark
def case_mask():
    nBS, nHead, nSLq, nHeadWidth = 1, 4, 3, 8
    nSLkv = 3
    data = {
        "q": np.random.rand(np.prod([nBS, nHead, nSLq, nHeadWidth])).astype(np.float32).reshape([nBS, nHead, nSLq, nHeadWidth]),
        "k": np.random.rand(np.prod([nBS, nHead, nSLkv, nHeadWidth])).astype(np.float32).reshape([nBS, nHead, nSLkv, nHeadWidth]),
        "v": -np.random.rand(np.prod([nBS, nHead, nSLkv, nHeadWidth])).astype(np.float32).reshape([nBS, nHead, nSLkv, nHeadWidth]),
    }

    tw = TRTWrapperV1()
    tw.network = tw.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    tensor_q = tw.network.add_input("q", datatype_np_to_trt(data["q"].dtype), data["q"].shape)
    tensor_k = tw.network.add_input("k", datatype_np_to_trt(data["k"].dtype), data["k"].shape)
    tensor_v = tw.network.add_input("v", datatype_np_to_trt(data["v"].dtype), data["v"].shape)

    mask_layer = tw.network.add_constant([nBS, nHead, nSLq, nSLkv], np.ones([nBS, nHead, nSLq, nSLkv], dtype=bool))

    attention = tw.network.add_attention(tensor_q, tensor_k, tensor_v, trt.AttentionNormalizationOp.SOFTMAX, False)
    attention.decomposable = True
    attention.causal = False
    attention.mask = mask_layer.get_output(0)
    print(f"{attention.num_inputs = }")
    print(f"{attention.num_outputs = }")

    output_tensor = attention.get_output(0)
    tw.build([output_tensor])
    tw.setup(data)
    tw.infer()

@case_mark
def case_quantization():
    nBS, nHead, nSLq, nHeadWidth = 1, 32, 16, 32
    nSLkv = 16
    data = {
        "q": np.random.rand(np.prod([nBS, nHead, nSLq, nHeadWidth])).astype(np.float32).reshape([nBS, nHead, nSLq, nHeadWidth]),
        "k": np.random.rand(np.prod([nBS, nHead, nSLkv, nHeadWidth])).astype(np.float32).reshape([nBS, nHead, nSLkv, nHeadWidth]),
        "v": -np.random.rand(np.prod([nBS, nHead, nSLkv, nHeadWidth])).astype(np.float32).reshape([nBS, nHead, nSLkv, nHeadWidth]),
    }

    tw = TRTWrapperV1()
    tw.network = tw.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    tensor_q = tw.network.add_input("q", datatype_np_to_trt(data["q"].dtype), data["q"].shape)
    tensor_k = tw.network.add_input("k", datatype_np_to_trt(data["k"].dtype), data["k"].shape)
    tensor_v = tw.network.add_input("v", datatype_np_to_trt(data["v"].dtype), data["v"].shape)

    qdq_data_type = trt.DataType.FP8  # Quantization data type can be either `trt.DataType.FP8` or `trt.DataType.INT8`

    q_q_scale = tw.network.add_constant([], np.array([60 / 127], dtype=np.float32))
    q_dq_scale = tw.network.add_constant([], np.array([1], dtype=np.float32))
    q_layer_q = tw.network.add_quantize(tensor_q, q_q_scale.get_output(0), qdq_data_type)
    q_layer_dq = tw.network.add_dequantize(q_layer_q.get_output(0), q_dq_scale.get_output(0), trt.DataType.FLOAT)

    k_q_scale = tw.network.add_constant([], np.array([60 / 127], dtype=np.float32))
    k_dq_scale = tw.network.add_constant([], np.array([1], dtype=np.float32))
    k_layer_q = tw.network.add_quantize(tensor_k, k_q_scale.get_output(0), qdq_data_type)
    k_layer_dq = tw.network.add_dequantize(k_layer_q.get_output(0), k_dq_scale.get_output(0), trt.DataType.FLOAT)

    v_q_scale = tw.network.add_constant([], np.array([60 / 127], dtype=np.float32))
    v_dq_scale = tw.network.add_constant([], np.array([1], dtype=np.float32))
    v_layer_q = tw.network.add_quantize(tensor_v, v_q_scale.get_output(0), qdq_data_type)
    v_layer_dq = tw.network.add_dequantize(v_layer_q.get_output(0), v_dq_scale.get_output(0), trt.DataType.FLOAT)

    fp8_scale_layer = tw.network.add_constant((1, ), trt.Weights(np.array([1.0 / 240.0], dtype=np.float32)))

    attention = tw.network.add_attention(q_layer_dq.get_output(0), k_layer_dq.get_output(0), v_layer_dq.get_output(0), trt.AttentionNormalizationOp.SOFTMAX, False)
    attention.decomposable = True
    attention.normalization_quantize_scale = fp8_scale_layer.get_output(0)
    # attention.normalization_quantize_to_type = qdq_data_type

    output_tensor = attention.get_output(0)
    output_tensor.name = 'attention_output'
    tw.build([output_tensor])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using attention structure
    case_simple()
    # Attention with customerized mask
    case_mask()
    # Quantization attention
    case_quantization()
    print("Finish")
