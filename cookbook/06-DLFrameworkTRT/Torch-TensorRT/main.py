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

from pathlib import Path

import numpy as np
import torch
import torch_tensorrt

model_file = Path("/trtcookbook/00-Data/model/model-trained.pth")
data_file = Path("/trtcookbook/00-Data/data/InferneceData.np")
ts_model_file = "model.ts"
shape = [1, 1, 28, 28]

model = torch.load(model_file)
data = np.load(data_file)

ts_model = torch.jit.trace(model, torch.randn(*shape, device="cuda"))  # torch script model
trt_model = torch_tensorrt.compile(ts_model, inputs=[torch.randn(*shape, device="cuda").float()], enabled_precisions={torch.float})

input_data = torch.from_numpy(data).cuda()
output_data = trt_model(input_data)  # run inference in TensorRT
print(output_data)

torch.jit.save(trt_model, ts_model_file)  # save TRT embedded Torchscript as .ts file

print("Finish")
