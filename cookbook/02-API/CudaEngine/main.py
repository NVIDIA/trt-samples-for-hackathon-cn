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

import numpy as np
import tensorrt as trt
from cuda import cudart

shape = [1, 4, 8, 8]
data = (np.arange(1, 1 + np.prod(shape), dtype=np.float32) / np.prod(shape) * 128).astype(np.float32).reshape(shape)
np.set_printoptions(precision=3, edgeitems=8, linewidth=300, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)
inputT0 = network.add_input("inputT0", trt.float32, [-1] + shape[1:])
profile.set_shape(inputT0.name, [1] + shape[1:], [2] + shape[1:], [4] + shape[1:])
config.add_optimization_profile(profile)
layer = network.add_identity(inputT0)
layer.name = "MyIdentityLayer"
layer.get_output(0).dtype = trt.int8
layer.set_output_type(0, trt.int8)
layer.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.CHW4)  # use a uncommon data format
layer.get_output(0).dynamic_range = [-128, 128]
network.mark_output(layer.get_output(0))
engineString = builder.build_serialized_network(network, config)

engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

print("engine.__len__() = %d" % len(engine))
print("engine.__sizeof__() = %d" % engine.__sizeof__())
print("engine.__str__() = %s" % engine.__str__())

print("\nEngine related ========================================================")
# All member functions with "binding" in name are deprecated since TEnsorRT 8.5
nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)  # count of input / output tensor
nOutput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.OUTPUT)
#nIO = engine.num_bindings  # deprecated, and this nIO is different from that got by Tensor API, refer to 09-Advance/MultiOptimizationProfile
#nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])
#nOutput = engine.num_bindings - nInput

print("engine.name = %s" % engine.name)
print("engine.device_memory_size = %d" % engine.device_memory_size)
print("engine.engine_capability = %d" % engine.engine_capability)  # refer to 02-API/BuilderConfig
print("engine.hardware_compatibility_level = %d" % engine.hardware_compatibility_level)
print("engine.num_aux_streams = %d" % engine.num_aux_streams)
print("engine.has_implicit_batch_dimension = %s" % engine.has_implicit_batch_dimension)
#print("engine.max_batch_size = %d" % engine.max_batch_size)  # used in Implicit Batch mode, deprecated since TensorRT 8.4, use Dyanmic Shape mode instead
print("engine.num_io_tensors = %d" % engine.num_io_tensors)
#print("engine.num_bindings = %d" % engine.num_bindings)  # deprecated since TensorRT 8.5
print("engine.num_layers = %d" % engine.num_layers)
print("engine.num_optimization_profiles = %d" % engine.num_optimization_profiles)
print("engine.refittable = %s" % engine.refittable)  # refer to 09-Advance/Refit
print("engine.tactic_sources = %d" % engine.tactic_sources)  # refer to 09-Advance/TacticSource

print("\nLayer related =========================================================")
print("engine.get_tensor_location(%s): %s" % (layer.get_output(0).name, engine.get_tensor_location(layer.get_output(0).name)))

print("\nInput / Output tensor related =========================================")
print("No. Input  output:                   %s 0,%s 1" % (" " * 56, " " * 56))
print("engine.get_tensor_name():            %58s,%58s" % (engine.get_tensor_name(0), engine.get_tensor_name(1)))
#print("get_binding_name():                  %58s,%58s" % (engine.get_binding_name(0), engine.get_binding_name(1)))
print("get_tensor_shape():                  %58s,%58s" % (engine.get_tensor_shape(lTensorName[0]), engine.get_tensor_shape(lTensorName[1])))
#print("get_binding_shape():                 %58s,%58s" % (engine.get_binding_shape(0), engine.get_binding_shape(1)))
print("get_tensor_dtype():                  %58s,%58s" % (engine.get_tensor_dtype(lTensorName[0]), engine.get_tensor_dtype(lTensorName[1])))
#print("get_binding_dtype():                 %58s,%58s" % (engine.get_binding_dtype(0), engine.get_binding_dtype(1)))
print("get_tensor_format():                 %58s,%58s" % (engine.get_tensor_format(lTensorName[0]), engine.get_tensor_format(lTensorName[1])))
#print("get_binding_format():                %58s,%58s" % (engine.get_binding_format(0), engine.get_binding_format(1)))
print("get_tensor_format_desc():            %58s,%58s" % (engine.get_tensor_format_desc(lTensorName[0]), engine.get_tensor_format_desc(lTensorName[1])))
#print("get_binding_format_desc():           %58s,%58s" % (engine.get_binding_format_desc(0), engine.get_binding_format_desc(1)))
print("get_tensor_bytes_per_component():    %58d,%58d" % (engine.get_tensor_bytes_per_component(lTensorName[0]), engine.get_tensor_bytes_per_component(lTensorName[1])))
#print("get_binding_bytes_per_component():   %58d,%58d" % (engine.get_binding_bytes_per_component(0), engine.get_binding_bytes_per_component(1)))
print("get_tensor_components_per_element(): %58d,%58d" % (engine.get_tensor_components_per_element(lTensorName[0]), engine.get_tensor_components_per_element(lTensorName[1])))
#print("get_binding_components_per_element():%58d,%58d" % (engine.get_binding_components_per_element(0), engine.get_binding_components_per_element(1)))
print("get_tensor_vectorized_dim():         %58d,%58d" % (engine.get_tensor_vectorized_dim(lTensorName[0]), engine.get_tensor_vectorized_dim(lTensorName[1])))
#print("get_binding_vectorized_dim():        %58d,%58d" % (engine.get_binding_vectorized_dim(0), engine.get_binding_vectorized_dim(1)))
print("")
print("get_tensor_mode():                   %58s,%58s" % (engine.get_tensor_mode(lTensorName[0]), engine.get_tensor_mode(lTensorName[1])))
#print("binding_is_input():                  %58s,%58s" % (engine.binding_is_input(0), engine.binding_is_input(1)))
print("get_tensor_location():               %58s,%58s" % (engine.get_tensor_location(lTensorName[0]), engine.get_tensor_location(lTensorName[0])))
print("Comment: Execution input / output tensor is on Device, while Shape input / output tensor is on CPU")
#print("get_location(int):                   %58s,%58s" % (engine.get_location(0), engine.get_location(1)))
#print("get_location(str):                   %58s,%58s" % (engine.get_location(lTensorName[0]), engine.get_location(lTensorName[1])))
print("is_shape_inference_io():             %58s,%58s" % (engine.is_shape_inference_io(lTensorName[0]), engine.is_shape_inference_io(lTensorName[0])))
#print("is_execution_binding():              %58s,%58s" % (engine.is_execution_binding(0), engine.is_execution_binding(1)))
#print("is_shape_binding():                  %58s,%58s" % (engine.is_shape_binding(0), engine.is_shape_binding(1)))
print("get_tensor_profile_shape():          %58s,%58s" % (engine.get_tensor_profile_shape(lTensorName[0], 0), "Optimization Profile is only for input tensor"))
#print("get_profile_shape():                 %58s,%58s" % (engine.get_profile_shape(0, 0), "Optimization Profile is only for input tensor"))
#print("get_profile_shape_input():           %58s,%58s" % ("No input shape tensor in this network", ""))
print("__getitem__(int):                    %58s,%58s" % (engine[0], engine[1]))
print("__getitem__(str):                    %58d,%58d" % (engine[lTensorName[0]], engine[lTensorName[1]]))

#print("get_binding_index:                   %58d,%58d" % (engine.get_binding_index(lTensorName[0]), engine.get_binding_index(lTensorName[1])))

context = engine.create_execution_context()
"""
Member of ICudaEngine:
++++        shown above
====        shown in binding part
~~~~        deprecated
----        not shown above
[no prefix] others

----__class__
__del__
__delattr__
__dir__
__doc__
__enter__
__eq__
__exit__
__format__
__ge__
__getattribute__
++++__getitem__
__gt__
__hash__
__init__
__init_subclass__
__le__
++++__len__
__lt__
__module__
__ne__
__new__
----__pybind11_module_local_v4_gcc_libstdcpp_cxxabi1013__
__reduce__
__reduce_ex__
__repr__
__setattr__
++++__sizeof__
++++__str__
__subclasshook__
++++binding_is_input
----create_engine_inspector                                                     refer to 09-Advance/EngineInspector
++++create_execution_context
----create_execution_context_without_device_memory                              refer to 0-Advance/CreateExecutionContextWithoutDeviceMemory
++++device_memory_size
++++engine_capability
----error_recorder                                                              refer to 09-Advance/ErrorRecorder
++++get_binding_bytes_per_component
++++get_binding_components_per_element
++++get_binding_dtype
++++get_binding_format
++++get_binding_format_desc
++++get_binding_index
++++get_binding_name
++++get_binding_shape
++++get_binding_vectorized_dim
++++get_location
++++get_profile_shape
++++get_profile_shape_input
++++get_tensor_bytes_per_component
++++get_tensor_components_per_element
++++get_tensor_dtype
++++get_tensor_format
++++get_tensor_format_desc
++++get_tensor_location
++++get_tensor_mode
++++get_tensor_name
++++get_tensor_profile_shape
++++get_tensor_shape
++++get_tensor_vectorized_dim
++++hardware_compatibility_level                                                refer to 02-API/BuilderConfig
++++has_implicit_batch_dimension
++++is_execution_binding
++++is_shape_binding
++++is_shape_inference_io
++++max_batch_size
++++name
++++num_aux_streams                                                             refer to 09-Advance/AuxStream
++++num_bindings
++++num_io_tensors
++++num_layers
++++num_optimization_profiles
----profiling_verbosity                                                         refer to 09-Advance/ProfilingVerbosity
++++refittable
----serialize                                                                   refer to 01-SimpleDemo/TensorRT8.5
++++tactic_sources
"""
