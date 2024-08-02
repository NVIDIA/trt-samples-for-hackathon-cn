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

from collections import OrderedDict

import numpy as np
import polygraphy.backend.trt as p
import tensorrt as trt

onnx_file = "model-trained.onnx"
trt_file = "model-trained.trt"
timing_cache_file = "model-trained.TimingCache"
input_data = OrderedDict([("x", np.load("/trtcookbook/00-Data/data/InferenceData.npy"))])

builder, network, parser = p.network_from_onnx_path(onnx_file)

builderConfig = p.CreateConfig( \
    tf32=False,
    fp16=True,
    int8=False,
    profiles=[p.Profile().add("x", [1, 1, 28, 28], [4, 1, 28, 28], [16, 1, 28, 28])],
    calibrator=None,
    precision_constraints=None,
    load_timing_cache=None,
    algorithm_selector=None,
    sparse_weights=False,
    tactic_sources=None,
    restricted=False,
    use_dla=False,
    allow_gpu_fallback=False,
    profiling_verbosity=None,
    memory_pool_limits={trt.MemoryPoolType.WORKSPACE:1<<30},
    refittable=False,
    strip_plan=False,
    preview_features=None,
    engine_capability=None,
    direct_io=False,
    builder_optimization_level=None,
    fp8=False,
    hardware_compatibility_level=None,
    max_aux_streams=4,
    version_compatible=False,
    exclude_lean_runtime=False,
    quantization_flags=None,
    error_on_timing_cache_miss=False,
    bf16=False,
    disable_compilation_cache=False,
    progress_monitor=None,
    weight_streaming=False,
    )

engine_bytes = p.engine_from_network([builder, network], config=builderConfig, save_timing_cache=timing_cache_file)

p.save_engine(engine_bytes, path=trt_file)

runner = p.TrtRunner(engine_bytes, name=None, optimization_profile=0)

runner.activate()

output = runner.infer(input_data, check_inputs=True)

runner.deactivate()

print(output)

print("Finish")
