#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
#

from collections import OrderedDict

import numpy as np
import polygraphy.backend.trt as p
import tensorrt as trt

onnxFile = "./modelA.onnx"
trtFile = "./modelA.plan"
cacheFile = "./modelA.cache"

builder, network, parser = p.network_from_onnx_path(onnxFile)

profileList = [p.Profile().add("tensorX", [1, 1, 28, 28], [4, 1, 28, 28], [16, 1, 28, 28])]

builderConfig = p.CreateConfig( \
    tf32=False,
    fp16=True,
    int8=False,
    profiles=profileList,
    calibrator=None,
    precision_constraints=None,
    strict_types=False,
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
    preview_features=None,
    engine_capability=None,
    direct_io=False,
    builder_optimization_level=None,
    fp8=False,
    hardware_compatibility_level=None,
    max_aux_streams=4,
    version_compatible=False,
    exclude_lean_runtime=False)

engineString = p.engine_from_network([builder, network], config=builderConfig, save_timing_cache=cacheFile)

p.save_engine(engineString, path=trtFile)

runner = p.TrtRunner(engineString, name=None, optimization_profile=0)

runner.activate()

output = runner.infer(OrderedDict([("tensorX", np.ascontiguousarray(np.random.rand(4, 1, 28, 28).astype(np.float32) * 2 - 1))]), check_inputs=True)

runner.deactivate()

print(output)
"""
methods of polygraphy.backend.trt:
'Algorithm'
'BytesFromEngine'
'Calibrator'
'CreateConfig'
'CreateNetwork'
'EngineBytesFromNetwork'
'EngineFromBytes'
'EngineFromNetwork'
'LoadPlugins'
'ModifyNetworkOutputs'
'NetworkFromOnnxBytes'
'NetworkFromOnnxPath'
'OnnxLikeFromNetwork'
'Profile'
'SaveEngine'
'ShapeTuple'
'TacticRecorder'
'TacticReplayData'
'TacticReplayer'
'TrtRunner'
'__builtins__'
'__cached__'
'__doc__'
'__file__'
'__loader__'
'__name__'
'__package__'
'__path__'
'__spec__'
'algorithm_selector'
'bytes_from_engine'
'calibrator'
'create_config'
'create_network'
'engine_bytes_from_network'
'engine_from_bytes'
'engine_from_network'
'get_trt_logger'
'load_plugins'
'loader'
'modify_network_outputs'
'network_from_onnx_bytes'
'network_from_onnx_path'
'onnx_like_from_network'
'profile'
'register_logger_callback'
'runner'
'save_engine'
'util'

"""
