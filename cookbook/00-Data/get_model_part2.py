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

import sys
from pathlib import Path

import numpy as np

sys.path.append("/trtcookbook/include")
from utils import (build_custom_op_network_onnx, build_half_mnist_network_onnx, build_invalid_network_onnx, build_labeled_network_onnx, build_redundant_network_onnx, build_reshape_network_onnx, build_unknown_network_onnx, case_mark)

np.random.seed(31193)
model_path = Path("/trtcookbook/00-Data/model/")
onnx_file_custom_op = model_path / "model-addscalar.onnx"
onnx_file_half_mnist = model_path / "model-half-mnist.onnx"
onnx_file_invalid = model_path / "model-invalid.onnx"
onnx_file_redundant = model_path / "model-redundant.onnx"
onnx_file_unknown = model_path / "model-unknown.onnx"
onnx_file_reshape = model_path / "model-reshape.onnx"
onnx_file_labeled = model_path / "model-labeled.onnx"

@case_mark
def case_custom_op():
    build_custom_op_network_onnx(onnx_file_custom_op)
    print(f"Succeed exporting {onnx_file_custom_op}")

@case_mark
def case_half_mnist():
    build_half_mnist_network_onnx(onnx_file_half_mnist)
    print(f"Succeed exporting {onnx_file_half_mnist}")

@case_mark
def case_invalid():
    build_invalid_network_onnx(onnx_file_invalid)
    print(f"Succeed exporting {onnx_file_invalid}")

@case_mark
def case_redundant():
    build_redundant_network_onnx(onnx_file_redundant)
    print(f"Succeed exporting {onnx_file_redundant}")

@case_mark
def case_unknown():
    build_unknown_network_onnx(onnx_file_unknown)
    print(f"Succeed exporting {onnx_file_unknown}")

@case_mark
def case_reshape():
    build_reshape_network_onnx(onnx_file_reshape)
    print(f"Succeed exporting {onnx_file_reshape}")

@case_mark
def case_labeled():
    build_labeled_network_onnx(onnx_file_labeled)
    print(f"Succeed exporting {onnx_file_labeled}")

if __name__ == "__main__":
    case_custom_op()
    case_half_mnist()
    case_invalid()
    case_redundant()
    case_unknown()
    case_reshape()
    case_labeled()

    print("Finish")
