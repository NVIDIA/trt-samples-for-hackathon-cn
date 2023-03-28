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

nB, nC, nH, nW = 1, 3, 4, 5
data = (np.arange(nB * nC * nH * nW, dtype=np.float32) / np.prod(nB * nC * nH * nW) * 128).astype(np.float32).reshape(nB, nC, nH, nW)

def formatToString(formatBitMask):
    output = ""
    if formatBitMask & (1 << int(trt.TensorFormat.LINEAR)):  # 0
        output += "LINEAR,"
    elif formatBitMask & (1 << int(trt.TensorFormat.CHW2)):  # 1
        output += "CHW2,"
    elif formatBitMask & (1 << int(trt.TensorFormat.HWC8)):  # 2
        output += "HWC8,"
    elif formatBitMask & (1 << int(trt.TensorFormat.CHW4)):  # 3
        output += "CHW4,"
    elif formatBitMask & (1 << int(trt.TensorFormat.CHW16)):  # 4
        output += "CHW16,"
    elif formatBitMask & (1 << int(trt.TensorFormat.CHW32)):  # 5
        output += "CHW32,"
    elif formatBitMask & (1 << int(trt.TensorFormat.DHWC8)):  # 6
        output += "DHWC8,"
    elif formatBitMask & (1 << int(trt.TensorFormat.CDHW32)):  # 7
        output += "CDHW32,"
    elif formatBitMask & (1 << int(trt.TensorFormat.HWC)):  # 8
        output += "HWC,"
    elif formatBitMask & (1 << int(trt.TensorFormat.DLA_LINEAR)):  # 9
        output += "DLA_LINEAR,"
    elif formatBitMask & (1 << int(trt.TensorFormat.DLA_HWC4)):  # 10
        output += "DLA_HWC4,"
    elif formatBitMask & (1 << int(trt.TensorFormat.HWC16)):  # 11
        output += "DHWC16,"
    if len(output) == 0:
        output = "None"
    else:
        output = output[:-1]
    return output

np.set_printoptions(precision=3, edgeitems=8, linewidth=300, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)
inputT0 = network.add_input("inputT0", trt.float32, (-1, nC, nH, nW))
inputT0.set_dimension_name(0, "Batch Size")
profile.set_shape(inputT0.name, [1, nC, nH, nW], [nB, nC, nH, nW], [nB * 2, nC, nH, nW])
config.add_optimization_profile(profile)

layer = network.add_identity(inputT0)
layer.precision = trt.int8
layer.set_output_type(0, trt.int8)

tensor = layer.get_output(0)
tensor.name = "Identity Layer Output Tensor 0"
tensor.dtype = trt.int8
tensor.allowed_formats = 1 << int(trt.TensorFormat.CHW4)
tensor.dynamic_range = [-128, 128]
tensor.reset_dynamic_range()
tensor.dynamic_range = [0, 128]

network.mark_output(tensor)

print("tensor.name = %s" % tensor.name)
print("tensor.shape = %s" % tensor.shape)
print("tensor.location = %s" % tensor.location)
print("tensor.__sizeof__() = %s" % tensor.__sizeof__())
print("tensor.__str__() = %s" % tensor.__str__())
print("tensor.broadcast_across_batch = %s" % tensor.broadcast_across_batch)
print("tensor.dtype = %s" % tensor.dtype)
print("tensor.allowed_formats = %s" % formatToString(tensor.allowed_formats))
print("tensor.dynamic_range = [%d, %d]" % (tensor.dynamic_range[0], tensor.dynamic_range[1]))
print("tensor.is_execution_tensor = %s" % tensor.is_execution_tensor)
print("tensor.is_shape_tensor = %s" % tensor.is_shape_tensor)
print("tensor.is_network_input = %s" % tensor.is_network_input)
print("tensor.is_network_output = %s" % tensor.is_network_output)

print("inputT0.get_dimension_name() = %s" % inputT0.get_dimension_name(0))  # only for input tensor
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
++++allowed_formats
++++broadcast_across_batch
++++dtype
++++dynamic_range
++++get_dimension_name
++++is_execution_tensor
++++is_network_input
++++is_network_output
++++is_shape_tensor
++++location
++++name
++++reset_dynamic_range
++++set_dimension_name
++++set_dynamic_range
++++shape
"""
