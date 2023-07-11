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

from polygraphy.backend.trt import network_from_onnx_path, CreateConfig, engine_from_network, save_engine

onnxFile = "./model.onnx"
planFile = "./model.plan"

network = network_from_onnx_path(onnxFile)

builderConfig = CreateConfig()

engineString = engine_from_network(network, config=builderConfig)

save_engine(engineString, path=planFile)

"""
network_from_onnx_path(path, explicit_precision=None)

CreateConfig( \
    max_workspace_size=None, 
    tf32=None, 
    fp16=None, 
    int8=None, 
    profiles=None, 
    calibrator=None, 
    obey_precision_constraints=None, 
    precision_constraints=None, 
    strict_types=None,
    load_timing_cache=None, 
    algorithm_selector=None, 
    sparse_weights=None, 
    tactic_sources=None, 
    restricted=None, 
    use_dla=None, 
    allow_gpu_fallback=None, 
    profiling_verbosity=None, 
    memory_pool_limits=None)

engine_from_network(network, config=None, save_timing_cache=None)

save_engine(engine, path)

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