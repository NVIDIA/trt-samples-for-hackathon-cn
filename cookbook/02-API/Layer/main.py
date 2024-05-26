#
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

import tensorrt as trt

shape = [3, 4, 5]

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network()
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)  # use Int8 mode in this example for using certain APIs
inputT0 = network.add_input("inputT0", trt.float32, [-1] * len(shape))
profile.set_shape(inputT0.name, [1, 1, 1], shape, shape)
config.add_optimization_profile(profile)

# Set the layer
layer = network.add_identity(inputT0)
layer.name = "Identity Layer"
layer.metadata = "My message"
layer.precision = trt.int8
layer.get_output(0).dtype = trt.int8
layer.set_output_type(0, trt.int8)
layer.reset_output_type(0)
layer.set_output_type(0, trt.int8)
layer.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.CHW4)
layer.get_output(0).dynamic_range = [-128, 128]

# Just for ensuring the network is self-consistent
network.mark_output(layer.get_output(0))
engineString = builder.build_serialized_network(network, config)

# Print information of the layer
print(f"{layer.name = }")
print(f"{layer.__class__ = }")  # use for type casting from ILayer to exact type of layer
print(f"{layer.metadata = }")
print(f"{layer.type = }")
print(f"{layer.__sizeof__() = }")
print(f"{layer.__str__ = }")
print(f"{layer.precision = }")
print(f"{layer.precision_is_set = }")
print(f"{layer.num_inputs = }")
for i in range(layer.num_inputs):
    print(f"\tlayer.get_input({i}) = {layer.get_input(i)}")  # get input tensor from layer
print(f"{layer.num_outputs = }")
for i in range(layer.num_outputs):
    print(f"\tlayer.get_output({i}) = {layer.get_output(i)}")  # get output tensor from layer
    print(f"\tlayer.get_output_type({i}) = {layer.get_output_type(i)}")
    print(f"\tlayer.output_type_is_set({i}) = {layer.output_type_is_set(i)}")

# More settings
layer.set_input(0, inputT0)  # set input tensor rather than add_* API
layer.reset_precision()
print(f"{layer.precision = }")
print(f"{layer.precision_is_set = }")
"""
Member of ILayer:
++++        shown above
----        not shown above
[no prefix] others

++++__class__
__delattr__
__dir__
__doc__
__eq__
__format__
__ge__
__getattribute__
__gt__
__hash__
__init__
__init_subclass__
__le__
__lt__
__module__
__ne__
__new__
__pybind11_module_local_v4_gcc_libstdcpp_cxxabi1013__
__reduce__
__reduce_ex__
__repr__
__setattr__
++++__sizeof__
++++__str__
__subclasshook__
++++get_input
++++get_output
++++get_output_type
++++metadata
++++name
++++num_inputs
++++num_outputs
++++output_type_is_set
++++precision
++++precision_is_set
++++reset_output_type
++++reset_precision
++++set_input
++++set_output_type
++++type
"""
