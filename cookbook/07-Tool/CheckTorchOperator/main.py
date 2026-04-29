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
import torch
from tensorrt_cookbook import case_mark, check_torch_operator

DimTorch = torch.export.dynamic_shapes.Dim

@case_mark
def case_static_repeat_interlace():

    class Net(torch.nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.rep = torch.tensor([1, 2, 3], dtype=torch.int64, device="cuda")

        def forward(self, x, y):
            # [Optional] dynamic shapes check
            assert x.shape[0] >= 1 and x.shape[0] <= 4
            assert x.shape[0] == y.shape[0]
            assert y.shape[1] == 1
            z = torch.repeat_interleave(x, self.rep, dim=0)
            w = x + y
            return z, w

    shape = 3, 4
    data = {
        "x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        "y": np.full((shape[0], 1), 2, dtype=np.float32),
    }
    d0 = DimTorch("BatchSize", min=1, max=4)
    d1 = DimTorch("SequenceLength", min=1, max=16)
    dynamic_shapes = {
        "x": {
            0: d0,
            1: d1
        },
        "y": {
            0: d0,
            1: 1
        },
    }
    check_torch_operator(Net, data, dynamic_shapes)
    # Save medium ONNX model locally for later inspection
    # from pathlib import Path
    # onnx_file = Path(f"model-{sys._getframe().f_code.co_name}.onnx")
    # check_torch_operator(Net, data, dynamic_shapes, onnx_file)
    return

@case_mark
def case_dynamic_repeat_interlace():

    class Net(torch.nn.Module):

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x, y):
            z = torch.repeat_interleave(x, y, dim=0)
            w = x + 2
            return z, w

    shape = 3, 4
    data = {
        "x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        "y": np.array([1, 2, 3], dtype=np.int64),
    }
    d0 = DimTorch("BatchSize", min=1, max=4)
    d1 = DimTorch("SequenceLength", min=1, max=16)
    dynamic_shapes = {
        "x": {
            0: d0,
            1: d1
        },
        "y": {
            0: 3
        },
    }
    check_torch_operator(Net, data, dynamic_shapes)
    return

@case_mark
def case_not_supported_trt():

    class Net(torch.nn.Module):

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            # 27th April, 2026
            # This operator is supported in PyTorch-2.11 / ONNX-1.18 / Onnxruntime-1.24 but not in TensorRT-10.16
            z = torch.unique(x)
            return z

    shape = 3, 3
    data = {
        "x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
    }
    d0 = DimTorch("BatchSize", min=1, max=4)
    d1 = DimTorch("SequenceLength", min=1, max=16)
    dynamic_shapes = {
        "x": {
            0: d0,
            1: d1
        },
    }
    check_torch_operator(Net, data, dynamic_shapes)
    return

@case_mark
def case_not_supported_onnx():

    class Net(torch.nn.Module):

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            # 27th April, 2026
            # This operator is supported in PyTorch-2.11 but not in ONNX-1.18
            # Candidate operators: torch.kthvalue / torch.searchsorted
            z = torch.histc(x, bins=10, min=0, max=9)
            return z

    shape = 3, 3
    data = {
        "x": np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
    }
    d0 = DimTorch("BatchSize", min=1, max=4)
    d1 = DimTorch("SequenceLength", min=1, max=16)
    dynamic_shapes = {
        "x": {
            0: d0,
            1: d1
        },
    }
    check_torch_operator(Net, data, dynamic_shapes)
    return

if __name__ == "__main__":
    # An supported normal operator
    case_static_repeat_interlace()
    # An supported data-dependent-shape operator
    case_dynamic_repeat_interlace()
    # An unsupported operator in TensorRT
    case_not_supported_trt()
    # An unsupported operator in ONNX
    case_not_supported_onnx()

    print("Finish")
