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

import onnx
import onnx_graphsurgeon as gs

sys.path.append("/trtcookbook/include")
from utils_onnx import build_mnist_network_onnx, print_graph

onnx_file = f"model-{__file__.split('/')[-1].split('.')[0]}.onnx"

build_mnist_network_onnx(onnx_file)

graph = gs.import_onnx(onnx.load(onnx_file))

print_graph(graph)
