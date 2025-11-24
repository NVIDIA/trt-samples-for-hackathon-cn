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

# Ground truth data
data0 = np.arange(np.prod([3, 4, 5]), dtype=np.float32).reshape([3, 4, 5])
data1 = np.zeros([2, 2], dtype=np.int32)

output_data_npy = np.load("output_data.npy")
output_data_npz = np.load("output_data.npz")

assert np.all(output_data_npy == data0), "Mismatch in output_data.npy"
assert np.all(output_data_npz["data0"] == data0), "Mismatch in output_data.npz['data0']"
assert np.all(output_data_npz["data1"] == data1), "Mismatch in output_data.npz['data1']"

print("Finish")
