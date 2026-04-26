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

import subprocess

import numpy as np
import pytest
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV2

def get_gpu_count() -> int:
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return sum(1 for line in output.splitlines() if line.strip().startswith("GPU"))
    except Exception:
        return 0

@pytest.mark.skip(reason="TODO")
class TestTemplateLayer:

    def test_case_simple(self, trt_cookbook_tester):
        if get_gpu_count() <= 1:
            pytest.skip("dist_collective requires at least 2 GPUs")

        def build_network(tw: TRTWrapperV2):
            data = {"tensor": np.zeros([31193], dtype=np.float32)}

            input_tensor = tw.network.add_input("input", trt.float32, [4])
            layer = tw.network.add_dist_collective(
                input_tensor,
                trt.CollectiveOperation.ALL_REDUCE,
                trt.ReduceOperation.SUM,
                -1,
                [0],
            )
            if layer is None:
                pytest.fail("dist_collective: layer creation failed")

            layer.metadata = "regression-dist"
            layer.num_ranks = 1
            output_tensor = layer.get_output(0)
            output_tensor.name = "output"
            tw.network.mark_output(output_tensor)

            return [output_tensor], data

        assert trt_cookbook_tester(build_network)
