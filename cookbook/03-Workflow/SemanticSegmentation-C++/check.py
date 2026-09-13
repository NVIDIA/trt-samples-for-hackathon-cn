# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""Compare the class map the C++ runtime wrote against the PyTorch reference.

Without this the example would only prove that the C++ ran, not that its pre-processing
layout, its argmax stride and its normalisation constants all agree with the model. Those
are exactly the three things that fail silently.
"""

from pathlib import Path

import numpy as np

output_path = Path(__file__).parent
reference = np.load(output_path / "data-reference.npy")
result = np.fromfile(output_path / "data-output.raw", dtype=np.uint8).reshape(reference.shape)

agree = int((reference == result).sum())
total = reference.size
print(f"Reference histogram: {np.bincount(reference.reshape(-1), minlength=4).tolist()}")
print(f"C++ histogram      : {np.bincount(result.reshape(-1), minlength=4).tolist()}")
print(f"Pixels agreeing    : {agree} / {total} ({agree / total * 100:.3f}%)")

assert agree == total, f"{total - agree} pixels differ between the C++ runtime and the PyTorch reference"
print("\nFinish")
