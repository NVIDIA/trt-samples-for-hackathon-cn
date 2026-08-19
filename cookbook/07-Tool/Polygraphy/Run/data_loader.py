# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
"""Feed real MNIST images to `polygraphy run --data-loader-script`.

Without this, Polygraphy generates random inputs, which is fine for checking that two backends
agree but useless for checking that the model is *right*: a random 28x28 image has no digit in it.

This file used to serve the INT8 calibration example of `../Convert/`. TensorRT 11 removed the
calibrator API (`--int8` raises `AttributeError: module 'tensorrt' has no attribute
'IInt8EntropyCalibrator2'`, see `../More/06-Int8IsNowExplicit/`), so its only remaining use is
here: a data loader is just a generator of feed dicts, and `run` accepts one.
"""

import numpy as np
from tensorrt_cookbook import cookbook_path

data = np.load(cookbook_path("00-Data", "data", "CalibrationData.npy"))
n_iteration = min(data.shape[0], 10)

def load_data():
    """Yield one feed dict per iteration. The function name is what `--data-loader-func-name` selects."""
    for i in range(n_iteration):
        yield {"x": data[i:i + 1]}
