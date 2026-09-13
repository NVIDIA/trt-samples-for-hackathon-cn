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
"""The input both API comparisons read, built from source rather than checked in.

`model-flat.onnx` is the 4-block MLP of `../02-PackageSurvey`. No `.onnx` in this
project is committed -- every one of them is produced by the code that describes
it -- so this rebuilds the file when it is not on disk, which also makes
`04-APIChoice` runnable without having run `02-PackageSurvey` first.

Kept in its own module so that `outline_with_onnx_api.py` and
`outline_with_graphsurgeon.py` each spend exactly one line on it and the 73 vs 51
line comparison stays about the outlining code alone.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "02-PackageSurvey"))
from demo_rewriter_as_function import build_flat_model

onnx_file = Path(__file__).parent.parent / "02-PackageSurvey" / "model-flat.onnx"
if not onnx_file.exists():
    build_flat_model(onnx_file)

SRC = str(onnx_file)
