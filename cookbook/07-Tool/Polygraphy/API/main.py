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

from collections import OrderedDict
from pathlib import Path

import numpy as np
import polygraphy.backend.trt as p
import tensorrt as trt
from tensorrt_cookbook import cookbook_path

onnx_file = cookbook_path("00-Data", "model", "model-trained.onnx")
trt_file = Path("model-trained.trt")
timing_cache_file = Path("model-trained.TimingCache")
input_data = OrderedDict([("x", np.load(cookbook_path("00-Data", "data", "InferenceData.npy")))])

builder, network, parser = p.network_from_onnx_path(str(onnx_file))

# Every keyword `CreateConfig` accepts in Polygraphy 0.50.3, with its default value.
# Seven of them are dead on TensorRT 11 and are marked below: they still exist in the signature and
# in `polygraphy run --help`, and they only fail once the config is applied to a network, which is
# why they are listed here rather than silently dropped.
builderConfig = p.CreateConfig( \
    tf32=False,
    fp16=False,  # DEAD on TRT 11: raises `PolygraphyException`, see ../More/02-ComparingBackends/
    int8=False,  # DEAD on TRT 11: raises `AttributeError: ... 'IInt8EntropyCalibrator2'`, see ../More/06-Int8IsNowExplicit/
    profiles=[p.Profile().add("x", [1, 1, 28, 28], [4, 1, 28, 28], [16, 1, 28, 28])],
    calibrator=None,  # DEAD on TRT 11: the whole calibrator API was removed
    precision_constraints=None,  # DEAD on TRT 11: `OBEY_/PREFER_PRECISION_CONSTRAINTS` were removed, see ../More/13-PerLayerPrecision/
    load_timing_cache=None,
    algorithm_selector=None,  # DEAD on TRT 11: `IAlgorithmSelector` was removed, see ../More/12-TacticsAndReproducibility/
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
    fp8=False,  # DEAD on TRT 11: raises `PolygraphyException`, like `fp16`
    hardware_compatibility_level=None,
    max_aux_streams=4,
    version_compatible=False,
    exclude_lean_runtime=False,
    quantization_flags=None,
    error_on_timing_cache_miss=False,
    bf16=False,  # DEAD on TRT 11: raises `PolygraphyException`, like `fp16`
    disable_compilation_cache=False,
    progress_monitor=None,
    weight_streaming=False,
    runtime_platform=None,  # e.g. `trt.RuntimePlatform.WINDOWS_AMD64` to cross-build a plan for another OS
    tiling_optimization_level=None,  # e.g. `trt.TilingOptimizationLevel.MODERATE`, trades build time for tiling search
    )

engine_bytes = p.engine_from_network([builder, network], config=builderConfig, save_timing_cache=str(timing_cache_file))

p.save_engine(engine_bytes, path=str(trt_file))

runner = p.TrtRunner(engine_bytes, name=None, optimization_profile=0)

runner.activate()

output = runner.infer(input_data, check_inputs=True)

runner.deactivate()

print(output)

print("Finish")
