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

from cuda import cudart
import numpy as np
import tensorrt as trt

nB, nC, nH, nW = 1, 4, 8, 8  # nC % 4 ==0, safe shape
#nB, nC, nH, nW = 1, 3, 8, 8  # nC % 4 !=0, may lose data in FP16 mode CHW4 format
data = (np.arange(1, 1 + nB * nC * nH * nW, dtype=np.float32) / np.prod(nB * nC * nH * nW) * 128).astype(np.float32).reshape(nB, nC, nH, nW)

np.set_printoptions(precision=3, edgeitems=8, linewidth=300, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)
inputT0 = network.add_input("inputT0", trt.float32, (-1, nC, nH, nW))
profile.set_shape(inputT0.name, [1, nC, nH, nW], [nB, nC, nH, nW], [nB * 2, nC, nH, nW])
config.add_optimization_profile(profile)

layer = network.add_identity(inputT0)
layer.name = "Identity Layer"
layer.metadata = "My message"  # since TensorRT 8.6
layer.precision = trt.int8
layer.reset_precision()
layer.precision = trt.int8
layer.get_output(0).dtype = trt.int8
layer.set_output_type(0, trt.int8)
layer.reset_output_type(0)
layer.set_output_type(0, trt.int8)
layer.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.CHW4)
layer.get_output(0).dynamic_range = [-128, 128]

network.mark_output(layer.get_output(0))

engineString = builder.build_serialized_network(network, config)

print("layer.name = %s" % layer.name)
print("layer.metadata = %s" % layer.metadata)
print("layer.type = %s" % layer.type)
print("layer.__sizeof__() = %s" % layer.__sizeof__())
print("layer.__str__ = %s" % layer.__str__())
print("layer.num_inputs = %d" % layer.num_inputs)
for i in range(layer.num_inputs):
    print("\tlayer.get_input(%d) = %s" % (i, layer.get_input(i)))
print("layer.num_outputs = %d" % layer.num_outputs)
for i in range(layer.num_outputs):
    print("\tlayer.get_output(%d) = %s" % (i, layer.get_output(i)))
    print("\tlayer.get_output_type(%d) = %s" % (i, layer.get_output_type(i)))
    print("\tlayer.output_type_is_set(%d) = %s" % (i, layer.output_type_is_set(i)))
print("layer.precision = %s" % layer.precision)
print("layer.precision_is_set = %s" % layer.precision_is_set)
"""
Member of ILayer:
++++        shown above
----        not shown above
[no prefix] others

----__class__
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
++++name
++++num_inputs
++++num_outputs
++++output_type_is_set
++++precision
++++precision_is_set
++++reset_precision
----set_input refer to 02-API/Layer/ShuffleLayer/DynamicShuffleWithShapeTensor.py
++++set_output_type
++++type
"""
