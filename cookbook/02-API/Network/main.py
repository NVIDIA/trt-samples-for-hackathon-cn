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
data = np.arange(nB * nC * nH * nW, dtype=np.float32).astype(np.float32).reshape(nB, nC, nH, nW)

np.set_printoptions(precision=3, edgeitems=8, linewidth=300, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)

network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
#network = builder.create_network((1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) | (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)))  # EXPLICIT_PRECISION is deprecated since TensorRT 8.5
network.name = "Identity Network"

profile = builder.create_optimization_profile()
config = builder.create_builder_config()
inputT0 = network.add_input("inputT0", trt.float32, (-1, nC, nH, nW))
profile.set_shape(inputT0.name, [1, nC, nH, nW], [nB, nC, nH, nW], [nB * 2, nC, nH, nW])
config.add_optimization_profile(profile)

layer = network.add_identity(inputT0)

network.mark_output(layer.get_output(0))
network.unmark_output(layer.get_output(0))
network.mark_output(layer.get_output(0))
#engineString = builder.build_serialized_network(network, config)

print("network.name = %s" % network.name)
print("network.__len__() = %d" % len(network))
print("network.__sizeof__() = %d" % network.__sizeof__())
print("network.__str__() = %s" % network.__str__())

print("network.num_inputs = %d" % network.num_inputs)
for i in range(network.num_inputs):
    print("\tnetwork.get_input(%d) = %s" % (i, network.get_input(i)))

print("network.num_outputs = %d" % network.num_outputs)
for i in range(network.num_outputs):
    print("\tnetwork.get_output(%d) = %s" % (i, network.get_output(i)))

print("network.num_layers = %d" % network.num_layers)
for i in range(network.num_layers):
    print("\tnetwork.get_layer(%d) = %s" % (i, network.get_layer(i)))
    #print("\tnetwork.__getitem__(%d) = %s" % (i, network.__getitem__(i)))  # same as get_layer()

print("netwrok.has_explicit_precision = %s" % network.has_explicit_precision)
print("netwrok.has_implicit_batch_dimension = %s" % network.has_implicit_batch_dimension)
"""
Member of INetwork:
++++        shown above
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
++++__getitem__ same as get_layer
__gt__
__hash__
__init__
__init_subclass__
__le__
+++__len__
__lt__
__module__
__ne__
__new__
__reduce__
__reduce_ex__
__repr__
__setattr__
+++__sizeof__
+++__str__
__subclasshook__
----add_activation all layers refer to 02-API/Layer
----add_assertion
----add_concatenation
----add_constant
----add_convolution
----add_convolution_nd
----add_deconvolution
----add_deconvolution_nd
----add_dequantize
----add_einsum
----add_elementwise
----add_fill
----add_fully_connected
----add_gather
----add_gather_v2
----add_grid_sample
----add_identity
----add_if_conditional
----add_input
----add_loop
----add_lrn
----add_matrix_multiply
----add_nms
----add_non_zero
----add_one_hot
----add_padding
----add_padding_nd
----add_parametric_relu
----add_plugin_v2
----add_pooling
----add_pooling_nd
----add_quantize
----add_ragged_softmax
----add_reduce
----add_resize
----add_rnn_v2
----add_scale
----add_scale_nd
----add_scatter
----add_select
----add_shape
----add_shuffle
----add_slice
----add_softmax
----add_topk
----add_unary
----error_recorder refer to 09-Advance/ErrorRecorder
++++get_input
++++get_layer
++++get_output
++++has_explicit_precision
++++has_implicit_batch_dimension
++++mark_output
----mark_output_for_shapes refer to 02-API/Layer/ShuffleLayer/DynamicShuffleWithShapeTensor.py
++++name
++++num_inputs
++++num_layers
++++num_outputs
----remove_tensor refer to 09-Advance/TensorRTGraphSurgeon
----set_weights_name refer to 09-Advance/Refit
++++unmark_output
----unmark_output_for_shapes unmark_output() for shape tensor, reder to 02-API/Layer/ShuffleLayer/DynamicShuffleWithShapeTensor.py
"""