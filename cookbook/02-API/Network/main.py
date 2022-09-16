#
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
from cuda import cudart
import tensorrt as trt

nB, nC, nH, nW = 1, 4, 8, 8  # nC % 4 ==0，全部值得到保存
#nB, nC, nH, nW = 1, 3, 8, 8  # nC % 4 !=0，会丢值
data = (np.arange(1, 1 + nB * nC * nH * nW, dtype=np.float32) / np.prod(nB * nC * nH * nW) * 128).astype(np.float32).reshape(nB, nC, nH, nW)

np.set_printoptions(precision=3, edgeitems=8, linewidth=300, suppress=True)
cudart.cudaDeviceSynchronize()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
#network = builder.create_network((1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) | (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)))  # EXPLICIT_PRECISION 已经被废弃
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
inputT0 = network.add_input("inputT0", trt.float32, (-1, nC, nH, nW))
profile.set_shape(inputT0.name, [1, nC, nH, nW], [nB, nC, nH, nW], [nB * 2, nC, nH, nW])
config.add_optimization_profile(profile)
#config.set_flag(trt.BuilderFlag.STRICT_TYPES)
#config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
config.set_flag(trt.BuilderFlag.DIRECT_IO)
config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)

layer = network.add_identity(inputT0)
layer.precision = trt.int8
layer.get_output(0).dtype = trt.int8
layer.set_output_type(0, trt.int8)
layer.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.CHW4)
layer.get_output(0).dynamic_range = [-128, 128]

network.mark_output(layer.get_output(0))
network.unmark_output(layer.get_output(0))
network.mark_output(layer.get_output(0))
#engineString = builder.build_serialized_network(network, config)

network.name = "Identity Network"

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
    #print("\tnetwork.__getintem__(%d) = %s" % (i, network.__getintem__(i)))

print("netwrok.has_explicit_precision = %s" % network.has_explicit_precision)

print("netwrok.has_implicit_batch_dimension = %s" % network.has_implicit_batch_dimension)

"""
INetwork 的成员方法
++++ 表示代码中进行了用法展示
---- 表示代码中没有进行展示
无前缀表示其他内部方法

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
++++__getitem__ 同 get_layer
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
----add_activation 见 02-API/Layer
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
----error_recorder 见 09-Advance/ErrorRecorder
++++get_input
++++get_layer
++++get_output
++++has_explicit_precision
++++has_implicit_batch_dimension
++++mark_output
----mark_output_for_shapes 见 02-API/Layer/ShuffleLayer/DynamicShuffleWithShapeTensor.py
++++name
++++num_inputs
++++num_layers
++++num_outputs
----remove_tensor 一般不能单独使用
----set_weights_name 见 09-Advance/Refit
----unmark_output
----unmark_output_for_shapes 针对 Shape Tensor 的 unmark_output，见 02-API/Layer/ShuffleLayer/DynamicShuffleWithShapeTensor.py
"""
