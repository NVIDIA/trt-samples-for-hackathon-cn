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

# import os
# import sys

# var_name = "TRT_COOKBOOK_PATH"

# path = os.environ.get(var_name)
# if path is None or not os.path.exists(path):
#     print(f"[ERROR] Environment variable `{var_name}` is not set or the path is invalid, please set it to the root directory of the TensorRT Cookbook repository!", file=sys.stderr)
#     sys.exit(1)

from .utils_class import *  # isort:disable
from .utils_cookbook import *  # isort:disable
from .utils_function import *  # isort:disable
from .utils_network import *  # isort:disable
from .utils_network_serialization import *  # isort:disable
from .utils_onnx import *  # isort:disable
from .utils_plugin import *  # isort:disable
