#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
#

import numpy as np
import torch as t
import torch.nn.functional as F

np.set_printoptions(precision=3, suppress=True)
h2 = 5
w2 = 9

inputData = t.Tensor(np.array([7, 5, 6, 4, 4, 2, 5, 3, 3, 9, 9, 7]).reshape(1, 1, 3, 4).astype(np.float32))

print("input data:")
print(inputData)
print("bilinear interpolate with align_corners=False:")
print(F.interpolate(inputData, size=((h2, w2)), mode="bilinear", align_corners=False).data.numpy())
print("bilinear interpolate with align_corners=True:")
print(F.interpolate(inputData, size=((h2, w2)), mode="bilinear", align_corners=True).data.numpy())
