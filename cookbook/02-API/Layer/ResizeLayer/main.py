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

import sys

import numpy as np
import tensorrt as trt

sys.path.append("/trtcookbook/include")
from utils import TRTWrapperShapeInput, TRTWrapperV1, case_mark

shape_input = 1, 3, 4, 5
shape_output = 2, 3, 6, 10
data = np.arange(np.prod(shape_input), dtype=np.float32).reshape(shape_input)
data = {"inputT0": data}

@case_mark
def case_simple():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_resize(tensor)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_static_shape():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    constant_layer = tw.network.add_constant([4], np.array(shape_output, dtype=np.int32))
    layer = tw.network.add_resize(tensor)
    layer.set_input(1, constant_layer.get_output(0))

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_shape():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_resize(tensor)
    layer.shape = shape_output

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_scale():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_resize(tensor)
    layer.scales = np.array(shape_output) / np.array(shape_input)

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_mode():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_resize(tensor)
    layer.resize_mode = trt.InterpolationMode.CUBIC
    layer.cubic_coeff = 0.5

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_selector_for_single_pixel():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_resize(tensor)
    layer.shape = [shape_input[0], shape_output[1], 1, 1]
    layer.resize_mode = trt.InterpolationMode.LINEAR
    layer.selector_for_single_pixel = trt.ResizeSelector.UPPER

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_nearest_rounding():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_resize(tensor)
    layer.shape = shape_output
    layer.nearest_rounding = trt.ResizeRoundMode.CEIL  # 设置最近邻插值舍入方法，默认值 FLOOR

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_coordinate_transformation():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_resize(tensor)
    layer.resize_mode = trt.InterpolationMode.NEAREST
    layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ALIGN_CORNERS

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_resize_as_another():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)

    shapeLayer = tw.network.add_shape(tensor)

    # Do something on the shape
    layer1 = tw.network.add_slice(shapeLayer.get_output(0), [0], [3], [1])
    layer2 = tw.network.add_slice(shapeLayer.get_output(0), [3], [1], [1])

    layer3 = tw.network.add_elementwise(layer2.get_output(0), layer2.get_output(0), trt.ElementWiseOperation.SUM)
    layer4 = tw.network.add_concatenation([layer1.get_output(0), layer3.get_output(0)])

    layer = tw.network.add_resize(tensor)
    layer.set_input(1, layer4.get_output(0))
    #layer.shape = tensor.shape[:2] + [tensor.shape[3] * 2]  # wrong because shape may contain -1 and cannot be used new shape

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

@case_mark
def case_shape_input():
    data_v2 = {"inputT0": data["inputT0"], "inputT1": np.array(shape_output, dtype=np.int32)}

    tw = TRTWrapperShapeInput()

    tensor0 = tw.network.add_input("inputT0", trt.float32, data_v2["inputT0"].shape)
    tensor1 = tw.network.add_input("inputT1", trt.int32, (4, ))
    tw.profile.set_shape_input(tensor1.name, [1 for _ in shape_input], shape_output, shape_output)
    tw.config.add_optimization_profile(tw.profile)

    layer = tw.network.add_resize(tensor0)
    layer.set_input(1, tensor1)

    tw.build([layer.get_output(0)])
    tw.setup(data_v2)
    tw.infer()

@case_mark
def case_exclude_outside():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_resize(tensor)
    layer.shape = shape_output
    layer.exclude_outside = 1

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A simple case of using Resize layer without any parameter
    case_simple()
    # Modify parameters after the constructor
    case_static_shape()
    #
    case_shape()
    #
    case_scale()

    case_mode()
    #
    case_selector_for_single_pixel()
    #
    case_nearest_rounding()

    case_coordinate_transformation()

    case_resize_as_another()

    case_shape_input()

    case_exclude_outside()

    print("Finish")
