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

from pathlib import Path
import hashlib

import numpy as np
from tensorrt_cookbook import (CookbookAlgorithmSelector, TRTWrapperV1, case_mark, load_mnist_network_trt)

trt_file = Path("deterministic.engine")

def _build_with_strategy(i_strategy: int = 0) -> bytes | None:
    data = {"x": np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)}
    callback_object_dict = {"algorithm_selector": CookbookAlgorithmSelector(i_strategy=i_strategy)}
    tw = TRTWrapperV1(callback_object_dict=callback_object_dict)

    load_mnist_network_trt(tw)

    tw.build()
    return tw.engine_bytes

def _hash_engine(engine_bytes: bytes) -> str:
    return hashlib.sha256(engine_bytes).hexdigest()

@case_mark
def case_compare():
    engine_bytes_0 = _build_with_strategy(0)
    engine_bytes_1 = _build_with_strategy(2)
    engine_bytes_2 = _build_with_strategy(2)
    print(f"Hash of engine with strategy=0        : {_hash_engine(engine_bytes_0)}")
    print(f"Hash of engine with strategy=2 (run 1): {_hash_engine(engine_bytes_1)}")
    print(f"Hash of engine with strategy=2 (run 2): {_hash_engine(engine_bytes_2)}")

if __name__ == "__main__":
    trt_file.unlink(missing_ok=True)

    case_compare()

    print("Finish")
