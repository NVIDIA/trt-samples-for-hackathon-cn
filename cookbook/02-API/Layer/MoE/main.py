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
from tensorrt_cookbook import TRTWrapperV1, case_mark, datatype_cast

@case_mark
def case_api_minimal():
    data = {
        "hidden_states": np.ones((1, 2, 8), dtype=np.float16),
        "selected_experts_for_tokens": np.zeros((1, 2, 1), dtype=np.int32),
        "scores_for_selected_experts": np.ones((1, 2, 1), dtype=np.float16),
    }

    tw = TRTWrapperV1()
    hidden_states = tw.network.add_input("hidden_states", datatype_cast(data["hidden_states"].dtype, "trt"), data["hidden_states"].shape)
    selected_experts = tw.network.add_input("selected_experts_for_tokens", datatype_cast(data["selected_experts_for_tokens"].dtype, "trt"), data["selected_experts_for_tokens"].shape)
    scores = tw.network.add_input("scores_for_selected_experts", datatype_cast(data["scores_for_selected_experts"].dtype, "trt"), data["scores_for_selected_experts"].shape)

    layer = tw.network.add_moe(hidden_states, selected_experts, scores)
    if layer is None:
        print("`add_moe` failed on current platform. According to TensorRT docs, IMoELayer is currently supported on SM110 (Thor).")
        return

    layer.activation_type = trt.MoEActType.SILU
    layer.metadata = "moe-minimal"
    layer.num_ranks = 1
    print(f"IMoELayer public members: {[m for m in dir(layer) if not m.startswith('__')]}")
    output_tensor = layer.get_output(0)
    output_tensor.name = "output"
    tw.build([output_tensor])
    tw.setup(data)
    tw.infer()

@case_mark
def case_api_methods():
    data = {
        "hidden_states": np.ones((1, 2, 8), dtype=np.float16),
        "selected_experts_for_tokens": np.zeros((1, 2, 1), dtype=np.int32),
        "scores_for_selected_experts": np.ones((1, 2, 1), dtype=np.float16),
        "fc_gate_weights": np.ones((2, 8, 4), dtype=np.float16),
        "fc_up_weights": np.ones((2, 8, 4), dtype=np.float16),
        "fc_down_weights": np.ones((2, 4, 8), dtype=np.float16),
        "fc_gate_biases": np.zeros((2, 4), dtype=np.float16),
        "fc_up_biases": np.zeros((2, 4), dtype=np.float16),
        "fc_down_biases": np.zeros((2, 8), dtype=np.float16),
        "fc_down_activation_scale": np.ones((1, ), dtype=np.float16),
        "fc_down_activation_dbl_q_scale": np.ones((1, ), dtype=np.float16),
    }

    tw = TRTWrapperV1()
    tw.network = tw.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    hidden_states = tw.network.add_input("hidden_states", datatype_cast(data["hidden_states"].dtype, "trt"), data["hidden_states"].shape)
    selected_experts = tw.network.add_input("selected_experts_for_tokens", datatype_cast(data["selected_experts_for_tokens"].dtype, "trt"), data["selected_experts_for_tokens"].shape)
    scores = tw.network.add_input("scores_for_selected_experts", datatype_cast(data["scores_for_selected_experts"].dtype, "trt"), data["scores_for_selected_experts"].shape)

    def add_const(name):
        return tw.network.add_constant(data[name].shape, trt.Weights(np.ascontiguousarray(data[name]))).get_output(0)

    layer = tw.network.add_moe(hidden_states, selected_experts, scores)
    if layer is None:
        print("`add_moe` failed on current platform. According to TensorRT docs, IMoELayer is currently supported on SM110 (Thor).")
        return

    layer.metadata = "moe-methods"
    layer.num_ranks = 1
    layer.activation_type = trt.MoEActType.SILU

    layer.set_gated_weights(
        add_const("fc_gate_weights"),
        add_const("fc_up_weights"),
        add_const("fc_down_weights"),
        trt.MoEActType.SILU,
    )
    layer.set_gated_biases(
        add_const("fc_gate_biases"),
        add_const("fc_up_biases"),
        add_const("fc_down_biases"),
    )
    layer.set_quantization_static(add_const("fc_down_activation_scale"), trt.DataType.FP8)
    layer.set_swiglu_params(10.0, 1.702, 1.0)
    layer.set_quantization_dynamic_dbl_q(
        add_const("fc_down_activation_dbl_q_scale"),
        trt.DataType.FP8,
        trt.Dims([32]),
        trt.DataType.E8M0,
    )

    print(f"{layer.activation_type = }")
    print(f"{layer.quantization_to_type = }")
    print(f"{layer.quantization_block_shape = }")
    print(f"{layer.dyn_q_output_scale_type = }")
    print(f"{layer.swiglu_param_limit = }, {layer.swiglu_param_alpha = }, {layer.swiglu_param_beta = }")

    output_tensor = layer.get_output(0)
    output_tensor.name = "output"
    if not tw.build([output_tensor]):
        print("Build failed for advanced MoE method demo on current platform/configuration")
        return

    tw.setup({
        "hidden_states": data["hidden_states"],
        "selected_experts_for_tokens": data["selected_experts_for_tokens"],
        "scores_for_selected_experts": data["scores_for_selected_experts"],
    })
    tw.infer()

if __name__ == "__main__":
    # Minimal MoE API example (with hardware guard)
    case_api_minimal()  # TODO: check this
    # Advanced MoE methods API example (with hardware/config guard)
    case_api_methods()  # TODO: check this

    print("Finish")
