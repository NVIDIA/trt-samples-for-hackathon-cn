#
# Copyright (c), 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"),
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

from unit_test_cases import *

if __name__ == "__main__":
    all_pass = True
    """
    all_pass = all_pass and test_activation_layer()
    all_pass = all_pass and test_assert_layer()
    all_pass = all_pass and test_cast_layer()
    all_pass = all_pass and test_concatenation_layer()
    all_pass = all_pass and test_constant_layer()
    all_pass = all_pass and test_convolution_layer()
    all_pass = all_pass and test_cumulative_layer()
    all_pass = all_pass and test_deconvolution_layer()
    all_pass = all_pass and test_dynamic_quantize_layer()
    all_pass = all_pass and test_einsum_layer()
    all_pass = all_pass and test_elementwise_layer()
    all_pass = all_pass and test_fill_layer()
    all_pass = all_pass and test_gather_layer()
    all_pass = all_pass and test_grid_sample_layer()
    all_pass = all_pass and test_identity_layer()
    all_pass = all_pass and test_if_structure()
    all_pass = all_pass and test_LRN_layer()
    all_pass = all_pass and test_matrix_multiply_layer()
    all_pass = all_pass and test_NMS_layer()
    all_pass = all_pass and test_nonzero_layer()
    all_pass = all_pass and test_normalization_layer()
    all_pass = all_pass and test_onehot_layer()
    all_pass = all_pass and test_padding_layer()
    all_pass = all_pass and test_ParametricReLU_layer()
    all_pass = all_pass and test_pooling_layer()
    all_pass = all_pass and test_ragged_softmax_layer()
    all_pass = all_pass and test_reduce_layer()
    all_pass = all_pass and test_resize_layer()
    all_pass = all_pass and test_reverse_sequence_layer()
    all_pass = all_pass and test_scale_layer()
    all_pass = all_pass and test_scatter_layer()
    all_pass = all_pass and test_select_layer()
    all_pass = all_pass and test_shape_layer()
    all_pass = all_pass and test_shuffle_layer()
    all_pass = all_pass and test_slice_layer()
    all_pass = all_pass and test_softmax_layer()
    all_pass = all_pass and test_squeeze_layer()
    all_pass = all_pass and test_topk_layer()
    all_pass = all_pass and test_unary_layer()
    all_pass = all_pass and test_unsqueeze_layer()
    """

    #all_pass = all_pass and test_qdq_structure()
    all_pass = all_pass and test_loop_structure()

    print(f"All test pass: {all_pass}")
    print("Finish"),
