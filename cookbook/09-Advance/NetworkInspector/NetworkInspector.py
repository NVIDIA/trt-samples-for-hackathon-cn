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

import json
import numpy as np
import tensorrt as trt

def extractBuilder(builder, builderConfig, network):

    dBuilder = {}  # Dictionary of Builder
    dBuilder["nMaxThread"] = builder.max_threads
    dBuilder["bPlatformHasTF32"] = builder.platform_has_tf32
    dBuilder["bPlatformHasFastFP16"] = builder.platform_has_fast_fp16
    dBuilder["bPlatformHasFastINT8"] = builder.platform_has_fast_int8
    dBuilder["nMaxDLABatchSize"] = builder.max_DLA_batch_size
    dBuilder["nDLACore"] = builder.num_DLA_cores
    dBuilder["bNetworkSupported"] = builder.is_network_supported(network, builderConfig)

    if int(trt.__version__.split(".")[0]) < 8:  # deprecated since TensorRT 8
        dBuilder["nMaxBatchSize"] = builder.max_batch_size
        dBuilder["nMaxWorkspaceSize"] = builder.max_workspace_size

    return dBuilder

def extractBuilderConfig(builderConfig, network, lOptimizationProfile):

    dBuilderConfig = {}  # Dictionary of BuilderConfig

    if int(trt.__version__.split(".")[0]) < 8:  # deprecated since TensorRT 8
        dBuilderConfig["nMaxWorkspaceSize"] = builderConfig.max_workspace_size
    else:
        dMemory = {}  # Dictionary of Memory management
        dMemory["WORKSPACE"] = builderConfig.get_memory_pool_limit(trt.MemoryPoolType.WORKSPACE)
        dMemory["DLA_MANAGED_SRAM"] = builderConfig.get_memory_pool_limit(trt.MemoryPoolType.DLA_MANAGED_SRAM)
        dMemory["DLA_LOCAL_DRAM"] = builderConfig.get_memory_pool_limit(trt.MemoryPoolType.DLA_LOCAL_DRAM)
        dMemory["DLA_GLOBAL_DRAM"] = builderConfig.get_memory_pool_limit(trt.MemoryPoolType.DLA_GLOBAL_DRAM)
        dBuilderConfig["nMemoryPoolLimit"] = dMemory

    dBuilderConfig["kDefaultDeviceType"] = int(builderConfig.default_device_type)  # save as int
    dBuilderConfig["nDLACore"] = builderConfig.DLA_core

    dBuilderConfig["kEngineCapability"] = int(builderConfig.engine_capability)  # save as int
    dBuilderConfig["nFlag"] = builderConfig.flags  # save as bit mask
    dBuilderConfig["nQuantizationFlag"] = int(builderConfig.quantization_flags)  # save as int
    dBuilderConfig["nOptimizationProfile"] = builderConfig.num_optimization_profiles
    dBuilderConfig["nProfileStream"] = builderConfig.profile_stream
    dBuilderConfig["kProfilingVerbosity"] = int(builderConfig.profiling_verbosity)  # save as int
    dBuilderConfig["nAverageTimingIteration"] = builderConfig.avg_timing_iterations
    dBuilderConfig["nTacticSource"] = builderConfig.get_tactic_sources()  # save as bit mask

    #dBuilderConfig["int8_calibrator"] = builderConfig.int8_calibrator # TODO
    #dBuilderConfig["algorithmSelector"] = builderConfig.algorithm_selector # TODO
    #dBuilderConfig["get_device_type"] = builderConfig.get_device_type() # TODO, layer related
    #dBuilderConfig["bCanRunOnDLA"] = builderConfig.can_run_on_DLA # TODO, layer related

    lAllOP = []  # List of All set of Optimization Profile
    if lOptimizationProfile is not None:
        assert (len(lOptimizationProfile) == builderConfig.num_optimization_profiles)
        for i in range(builderConfig.num_optimization_profiles):
            optimizationProfile = lOptimizationProfile[i]
            lOP = []  # List of one set of Optimization Profile
            for j in range(network.num_inputs):
                tensor = network.get_input(j)
                if tensor.is_shape_tensor:
                    shapeList = optimizationProfile.get_shape_input(tensor.name)
                else:
                    shapeList = optimizationProfile.get_shape(tensor.name)
                if shapeList == []:
                    print("[INFO from NetowrkInspector]: No profile for input tensor %d, continue" % j)
                    lOP.append({})  # place holder for input tensor j
                else:
                    dOP = {}  # Dictionary of Optimization Profile for one input tensor
                    dOP["name"] = tensor.name
                    dOP["bShapeTensor"] = tensor.is_shape_tensor
                    dOP["nDimension"] = len(tensor.shape)
                    dOP["min"] = list(shapeList[0])
                    dOP["opt"] = list(shapeList[1])
                    dOP["max"] = list(shapeList[2])
                    lOP.append(dOP)
            lAllOP.append(lOP)
    dBuilderConfig["lOptimizationProfile"] = lAllOP

    lOP = []  # List of whole sets of CalibrationProfile
    calibrationProfile = builderConfig.get_calibration_profile()
    if calibrationProfile is not None:
        for j in range(network.num_inputs):
            name = network.get_input(j).name
            shapeList = calibrationProfile.get_shape(name)
            dOP = {}  # Dictionary of CalibrationProfile for only one single input tensor
            dOP["name"] = name
            dOP["min"] = list(shapeList[0])
            dOP["opt"] = list(shapeList[1])
            dOP["max"] = list(shapeList[2])
            lOP.append(dOP)
    dBuilderConfig["lCalibrationProfile"] = lOP

    return dBuilderConfig

def extractNetwork(network):

    dNetwork = {}  # Dictionary of Network
    dNetwork["sName"] = network.name
    dNetwork["nLayer"] = network.num_layers
    dNetwork["nInput"] = network.num_inputs
    dNetwork["nOutput"] = network.num_outputs
    dNetwork["bImplicitBatchMode"] = network.has_implicit_batch_dimension
    dNetwork["bExplicitPrecision"] = network.has_explicit_precision

    lBinding = []  # List of Binding
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        dBinding = {}  # Dictionary of Binding
        dBinding["sName"] = tensor.name
        dBinding["bInput"] = tensor.is_network_input
        dBinding["bShapeTensor"] = tensor.is_shape_tensor
        lBinding.append(dBinding)
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        dBinding = {}
        dBinding["sName"] = tensor.name
        dBinding["bInput"] = tensor.is_network_input
        dBinding["bShapeTensor"] = tensor.is_shape_tensor
        lBinding.append(dBinding)
    dNetwork["Binding"] = lBinding

    return dNetwork

def exactTensor(tensor):

    dTensor = {}  # Dicitonary of TEnsor
    dTensor["lShape"] = list(tensor.shape)  # save as list
    dTensor["kLocation"] = int(tensor.location)  # save as int
    dTensor["bBroadcastAcrossBatch"] = tensor.broadcast_across_batch  # useless in Explicit Batch mode
    dTensor["kDataType"] = int(tensor.dtype)  # save as int
    dTensor["nAllowedFormat"] = tensor.allowed_formats  # save as bit mask
    dTensor["lDynamicRange"] = None if tensor.dynamic_range is None else list(tensor.dynamic_range)  # save as list
    dTensor["bExecutionTensor"] = tensor.is_execution_tensor
    dTensor["bShapeTensor"] = tensor.is_shape_tensor
    dTensor["bNetworkInput"] = tensor.is_network_input
    dTensor["bNetworkOutput"] = tensor.is_network_output

    return dTensor

def exactLayerAndTensor(network):

    lLayer = []  # List of Layer, layers are indexed by serial number
    dTensor = {}  # Dictionary of Tensor, tensors are indexed by name, not by serial number
    dIfCondition = {}  # Dictionary of IfCondition structure
    dLoop = {}  # Dictionary of Loop structure
    parameter = {}  # weight of each layer

    for i in range(network.num_layers):
        layer = network.get_layer(i)
        dLayer = {}  # Dictionary of one layer
        dLayer["kIndex"] = i
        dLayer["sName"] = layer.name
        dLayer["kType"] = int(layer.type)
        dLayer["nInput"] = layer.num_inputs
        dLayer["nOutput"] = layer.num_outputs
        dLayer["kPrecision"] = int(layer.precision)  # save as int
        dLayer["bPrecisionIsSet"] = layer.precision_is_set

        lInputTensor = []  # List of Input Tensor
        for j in range(layer.num_inputs):
            tensor = layer.get_input(j)
            if layer.type == trt.LayerType.FILL and j == 0 and tensor is None:  # for linspace fill mode of Fill layer, inputTensor 0 could be None
                lInputTensor.append(None)
            elif layer.type == trt.LayerType.SLICE and j < layer.num_inputs - 1 and tensor is None:  # for Slice layer, input tensors before the last one could be None
                lInputTensor.append(None)
            elif layer.type == trt.LayerType.RNN_V2 and j >= 1 and tensor == None:  # for RNNV2 layer, seq_lengths / hidden_state / cell_state tensor could be None
                lInputTensor.append(None)
            else:
                lInputTensor.append(tensor.name)
                if not tensor.name in dTensor.keys():
                    dTensor[tensor.name] = exactTensor(tensor)
        dLayer["lInputTensorName"] = lInputTensor

        lOutputTensor = []  # List of Output Tensor
        lOutputTensorDataType = []  # List of Output Tensor Data Type
        for j in range(layer.num_outputs):
            tensor = layer.get_output(j)
            lOutputTensor.append(tensor.name)
            lOutputTensorDataType.append(int(layer.get_output_type(j)))
            if not tensor.name in dTensor.keys():
                dTensor[tensor.name] = exactTensor(tensor)
        dLayer["lOutputTensorName"] = lOutputTensor
        dLayer["lOutputTensorDataType"] = lOutputTensorDataType

        # Specialization of each layer
        #  0 LayerType.CONVOLUTION
        #  1 LayerType.FULLY_CONNECTED
        #  2 LayerType.ACTIVATION
        #  3 LayerType.POOLING
        #  4 LayerType.LRN
        #  5 LayerType.SCALE
        #  6 LayerType.SOFTMAX
        #  7 LayerType.DECONVOLUTION
        #  8 LayerType.CONCATENATION
        #  9 LayerType.ELEMENTWISE
        # 10 LayerType.PLUGIN
        # 11 LayerType.UNARY
        # 12 LayerType.PADDING
        # 13 LayerType.SHUFFLE
        # 14 LayerType.REDUCE
        # 15 LayerType.TOPK
        # 16 LayerType.GATHER
        # 17 LayerType.MATRIX_MULTIPLY
        # 18 LayerType.RAGGED_SOFTMAX
        # 19 LayerType.CONSTANT
        # 20 LayerType.RNN_V2
        # 21 LayerType.IDENTITY
        # 22 LayerType.PLUGIN_V2
        # 23 LayerType.SLICE
        # 24 LayerType.SHAPE
        # 25 LayerType.PARAMETRIC_RELU
        # 26 LayerType.RESIZE
        # 27 LayerType.TRIP_LIMIT
        # 28 LayerType.RECURRENCE
        # 29 LayerType.ITERATOR
        # 30 LayerType.LOOP_OUTPUT
        # 31 LayerType.SELECT
        # 32 LayerType.FILL
        # 33 LayerType.QUANTIZE
        # 34 LayerType.DEQUANTIZE
        # 35 LayerType.CONDITION
        # 36 LayerType.CONDITIONAL_INPUT
        # 37 LayerType.CONDITIONAL_OUTPUT
        # 38 LayerType.SCATTER
        # 39 LayerType.EINSUM
        # 40 LayerType.ASSERTION

        if layer.type == trt.LayerType.CONVOLUTION:  # 0
            layer.__class__ = trt.IConvolutionLayer
            dLayer["kernel"] = layer.name + "-kernel"
            dLayer["lKernelShape"] = [layer.num_output_maps, layer.get_input(0).shape[1], *list(layer.kernel_size_nd)]
            parameter[layer.name + "-kernel"] = layer.kernel
            dLayer["bias"] = layer.name + "-bias"
            dLayer["lBiasShape"] = list(layer.bias.shape)
            parameter[layer.name + "-bias"] = layer.bias
            #dLayer["kernel_size"] = list(layer.kernel_size)  # deprecated since TensorRT 8
            dLayer["kernel_size_nd"] = list(layer.kernel_size_nd)  # save as list
            dLayer["num_output_maps"] = layer.num_output_maps
            #dLayer["stride"] = list(layer.stride)  # deprecated since TensorRT 8
            dLayer["stride_nd"] = list(layer.stride_nd)  # save as list
            #dLayer["dilation"] = list(layer.dilation)  # deprecated since TensorRT 8
            dLayer["dilation_nd"] = list(layer.dilation_nd)  # save as list
            dLayer["num_groups"] = layer.num_groups
            #dLayer["padding"] = layer.padding  # deprecated since TensorRT 8
            dLayer["padding_nd"] = list(layer.padding_nd)  # save as list
            dLayer["padding_mode"] = int(layer.padding_mode)  # save as int
            dLayer["pre_padding"] = list(layer.pre_padding)  # save as list
            dLayer["post_padding"] = list(layer.post_padding)  # save as list

        elif layer.type == trt.LayerType.FULLY_CONNECTED:  # 1
            layer.__class__ = trt.IFullyConnectedLayer
            dLayer["kernel"] = layer.name + "-kernel"
            dLayer["lKernelShape"] = [layer.num_output_channels, layer.kernel.shape[0] // layer.num_output_channels]
            parameter[layer.name + "-kernel"] = layer.kernel
            dLayer["bias"] = layer.name + "-bias"
            dLayer["lBiasShape"] = list(layer.bias.shape)
            parameter[layer.name + "-bias"] = layer.bias
            dLayer["num_output_channels"] = layer.num_output_channels

        elif layer.type == trt.LayerType.ACTIVATION:  # 2
            layer.__class__ = trt.IActivationLayer
            dLayer["alpha"] = layer.alpha
            dLayer["beta"] = layer.beta
            dLayer["type"] = int(layer.type)  # save as int

        elif layer.type == trt.LayerType.POOLING:  # 3
            layer.__class__ = trt.IPoolingLayer
            dLayer["average_count_excludes_padding"] = layer.average_count_excludes_padding
            dLayer["blend_factor"] = layer.blend_factor
            #dLayer["stride"] = list(layer.stride)  # deprecated since TensorRT 8
            dLayer["stride_nd"] = list(layer.stride_nd)  # save as list
            #dLayer["padding"] = layer.padding  # deprecated since TensorRT 8
            dLayer["padding_nd"] = list(layer.padding_nd)  # save as list
            dLayer["padding_mode"] = int(layer.padding_mode)  # save as int
            dLayer["pre_padding"] = list(layer.pre_padding)  # save as list
            dLayer["post_padding"] = list(layer.post_padding)  # save as list
            dLayer["type"] = int(layer.type)  # save as int
            #dLayer["window_size"] = list(layer.window_size)  # deprecated since TensorRT 8
            dLayer["window_size_nd"] = list(layer.window_size_nd)  # save as list

        elif layer.type == trt.LayerType.LRN:  # 4
            layer.__class__ = trt.ILRNLayer
            dLayer["alpha"] = layer.alpha
            dLayer["beta"] = layer.beta
            dLayer["k"] = layer.k
            dLayer["window_size"] = layer.window_size

        elif layer.type == trt.LayerType.SCALE:  # 5
            layer.__class__ = trt.IScaleLayer
            dLayer["channel_axis"] = layer.channel_axis
            dLayer["mode"] = int(layer.mode)  # save as int
            dLayer["scale"] = layer.name + "-scale"
            dLayer["lScaleShape"] = layer.scale.shape
            parameter[layer.name + "-scale"] = layer.scale
            dLayer["shift"] = layer.name + "-shift"
            dLayer["lShiftShape"] = layer.shift.shape
            parameter[layer.name + "-shift"] = layer.shift
            dLayer["power"] = layer.name + "-power"
            dLayer["lPowerShape"] = layer.power.shape
            parameter[layer.name + "-power"] = layer.power

        elif layer.type == trt.LayerType.SOFTMAX:  # 6
            layer.__class__ = trt.ISoftMaxLayer
            dLayer["axes"] = layer.axes

        elif layer.type == trt.LayerType.DECONVOLUTION:  # 7
            layer.__class__ = trt.IDeconvolutionLayer
            dLayer["kernel"] = layer.name + "-kernel"
            dLayer["lKernelShape"] = [layer.num_output_maps, layer.get_input(0).shape[1], *list(layer.kernel_size_nd)]
            parameter[layer.name + "-kernel"] = layer.kernel
            dLayer["bias"] = layer.name + "-bias"
            dLayer["lBiasShape"] = list(layer.bias.shape)
            parameter[layer.name + "-bias"] = layer.bias
            #dLayer["kernel_size"] = list(layer.kernel_size)  # deprecated since TensorRT 8
            dLayer["kernel_size_nd"] = list(layer.kernel_size_nd)  # save as list
            dLayer["num_output_maps"] = layer.num_output_maps
            #dLayer["stride"] = list(layer.stride)  # deprecated since TensorRT 8
            dLayer["stride_nd"] = list(layer.stride_nd)  # save as list
            #dLayer["dilation"] = list(layer.dilation)  # deprecated since TensorRT 8
            dLayer["dilation_nd"] = list(layer.dilation_nd)  # save as list
            dLayer["num_groups"] = layer.num_groups
            #dLayer["padding"] = list(layer.padding)  # deprecated since TensorRT 8
            dLayer["padding_nd"] = list(layer.padding_nd)  # save as list
            dLayer["padding_mode"] = int(layer.padding_mode)  # save as int
            dLayer["pre_padding"] = list(layer.pre_padding)  # save as list
            dLayer["post_padding"] = list(layer.post_padding)  # save as list

        elif layer.type == trt.LayerType.CONCATENATION:  # 8
            layer.__class__ = trt.IConcatenationLayer
            dLayer["axis"] = layer.axis

        elif layer.type == trt.LayerType.ELEMENTWISE:  # 9
            layer.__class__ = trt.IElementWiseLayer
            dLayer["op"] = int(layer.op)  # save as int

        elif layer.type == trt.LayerType.PLUGIN:  # 10
            print("IPlugin Layer not supported!")  # layer.__class__ = trt.IPluginLayer
            #break

        elif layer.type == trt.LayerType.UNARY:  # 11
            layer.__class__ = trt.IUnaryLayer
            dLayer["op"] = int(layer.op)  # save as int

        elif layer.type == trt.LayerType.PADDING:  # 12
            layer.__class__ = trt.IPaddingLayer
            #dLayer["pre_padding"] = list(layer.pre_padding)  # deprecated since TensorRT 8
            dLayer["pre_padding_nd"] = list(layer.pre_padding_nd)  # save as list, different from Convolution / Deconvolution Layer
            #dLayer["post_padding"] = layer.post_padding  # deprecated since TensorRT 8  # different from Convolution / Deconvolution Layer
            dLayer["post_padding_nd"] = list(layer.post_padding_nd)  # save as list

        elif layer.type == trt.LayerType.SHUFFLE:  # 13
            layer.__class__ = trt.IShuffleLayer

            if layer.num_inputs == 2:  # dynamic shuffle mode
                dLayer["bDynamicShuffle"] = True
            else:
                dLayer["bDynamicShuffle"] = False
                try:
                    dLayer["reshape_dims"] = list(layer.reshape_dims)  # save as list
                except ValueError:
                    dLayer["reshape_dims"] = None  # no reshape operation if ValueError raised
            dLayer["first_transpose"] = list(layer.first_transpose)  # save as list
            dLayer["second_transpose"] = list(layer.second_transpose)  # save as list
            dLayer["zero_is_placeholder"] = layer.zero_is_placeholder

        elif layer.type == trt.LayerType.REDUCE:  # 14
            layer.__class__ = trt.IReduceLayer
            dLayer["axes"] = layer.axes
            dLayer["op"] = int(layer.op)  # save as int
            dLayer["keep_dims"] = layer.keep_dims

        elif layer.type == trt.LayerType.TOPK:  # 15
            layer.__class__ = trt.ITopKLayer
            dLayer["axes"] = layer.axes
            dLayer["op"] = int(layer.op)  # save as int
            dLayer["k"] = layer.k

        elif layer.type == trt.LayerType.GATHER:  # 16
            layer.__class__ = trt.IGatherLayer
            dLayer["axis"] = layer.axis
            dLayer["mode"] = int(layer.mode)  # save as int

        elif layer.type == trt.LayerType.MATRIX_MULTIPLY:  # 17
            layer.__class__ = trt.IMatrixMultiplyLayer
            dLayer["op0"] = int(layer.op0)  # save as int
            dLayer["op1"] = int(layer.op1)  # save as int

        elif layer.type == trt.LayerType.RAGGED_SOFTMAX:  # 18
            layer.__class__ = trt.IRaggedSoftMaxLayer

        elif layer.type == trt.LayerType.CONSTANT:  # 19
            layer.__class__ = trt.IConstantLayer
            dLayer["weights"] = layer.name + "-weights"
            dLayer["lWeightShape"] = list(layer.shape)
            parameter[layer.name + "-weights"] = layer.weights
            dLayer["shape"] = list(layer.shape)

        elif layer.type == trt.LayerType.RNN_V2:  # 20
            layer.__class__ = trt.IRNNv2Layer
            dLayer["num_layers"] = layer.num_layers
            dLayer["hidden_size"] = layer.hidden_size
            dLayer["max_seq_length"] = layer.max_seq_length
            dLayer["data_length"] = layer.data_length
            dLayer["op"] = int(layer.op)  # save as int
            dLayer["input_mode"] = int(layer.input_mode)  # save as int
            dLayer["direction"] = int(layer.direction)  # save as int
            nRealLayer = layer.num_layers * (2 if layer.direction == trt.RNNDirection.BIDIRECTION else 1)
            if layer.op == trt.RNNOperation.RELU or layer.op == trt.RNNOperation.TANH:
                lGateKind = [trt.RNNGateType.INPUT]
            elif layer.op == trt.RNNOperation.LSTM:
                lGateKind = [trt.RNNGateType.INPUT, trt.RNNGateType.CELL, trt.RNNGateType.FORGET, trt.RNNGateType.OUTPUT]
            elif layer.op == trt.RNNOperation.GRU:
                lGateKind = [trt.RNNGateType.UPDATE, trt.RNNGateType.RESET]
            else:
                lGateKind = []
            for j in range(nRealLayer):
                for gateKind in lGateKind:
                    if layer.input_mode == trt.RNNInputMode.LINEAR:
                        parameter[layer.name + "-" + str(j) + "-" + str(int(gateKind)) + "-weightX"] = layer.get_weights_for_gate(j, gateKind, True)
                    parameter[layer.name + "-" + str(j) + "-" + str(int(gateKind)) + "-biasX"] = layer.get_bias_for_gate(j, gateKind, True)  # bias for X is always needed
                    parameter[layer.name + "-" + str(j) + "-weightH"] = layer.get_weights_for_gate(j, gateKind, False)
                    parameter[layer.name + "-" + str(j) + "-biasH"] = layer.get_bias_for_gate(j, gateKind, False)

        elif layer.type == trt.LayerType.IDENTITY:  # 21
            layer.__class__ = trt.IIdentityLayer

        elif layer.type == trt.LayerType.PLUGIN_V2:  # 22
            layer.__class__ = trt.IPluginV2Layer
            print("PluginV2 Layer not support!")
            #dLayer["plugin_namespace"] = layer.plugin_namespace
            #dLayer["plugin_type"] = layer.plugin_type
            #dLayer["plugin_version"] = layer.plugin_version
            #dLayer["tensorrt_version"] = layer.tensorrt_version

        elif layer.type == trt.LayerType.SLICE:  # 23
            layer.__class__ = trt.ISliceLayer
            dLayer["mode"] = int(layer.mode)  # save as int
            try:
                dLayer["start"] = list(layer.start)  # save as list
            except ValueError:
                dLayer["start"] = None
            try:
                dLayer["shape"] = list(layer.shape)  # save as list
            except ValueError:
                dLayer["shape"] = None
            try:
                dLayer["stride"] = list(layer.stride)  # save as list
            except ValueError:
                dLayer["stride"] = None
            if layer.mode == trt.SliceMode.FILL and layer.num_inputs == 5:
                dLayer["fill"] = True
            else:
                dLayer["fill"] = False

        elif layer.type == trt.LayerType.SHAPE:  # 24
            layer.__class__ = trt.IShapeLayer

        elif layer.type == trt.LayerType.PARAMETRIC_RELU:  # 25
            layer.__class__ = trt.IParametricReLULayer

        elif layer.type == trt.LayerType.RESIZE:  # 26
            layer.__class__ = trt.IResizeLayer
            if layer.num_inputs == 2:  # dynamic resize mode
                dLayer["bDynamicResize"] = True
            else:
                dLayer["bDynamicResize"] = False
                if layer.scales == []:  # static resize mode + use shape mode
                    dLayer["bShapeMode"] = True
                    dLayer["shape"] = list(layer.shape)  # save as list
                else:  # static resize mode + use scale mode, TODO: how to check layer.shape? such as "layer.shape == [0] == [0]"
                    dLayer["bShapeMode"] = False
                    dLayer["scales"] = list(layer.scales)  # save as list
            dLayer["resize_mode"] = int(layer.resize_mode)  # save as int
            if layer.resize_mode == trt.ResizeMode.LINEAR and layer.coordinate_transformation == trt.ResizeCoordinateTransformation.ASYMMETRIC:
                print("[Warning from NetworkInspector]: ResizeCoordinateTransformation of Resize Layer %s is set as HALF_PIXEL though default behaviour or your explicit set is ASYMMETRIC mode, please refer to the source code of NetworkInspector if you insist to use ASYMMETRIC mode!" % layer.name)
                layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
                #layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC # uncomment this line if you want to use ASYMMETRIC mode
            dLayer["coordinate_transformation"] = int(layer.coordinate_transformation)  # save as int
            dLayer["selector_for_single_pixel"] = int(layer.selector_for_single_pixel)  # save as int
            dLayer["nearest_rounding"] = int(layer.nearest_rounding)  # save as int

        elif layer.type == trt.LayerType.TRIP_LIMIT:  # 27
            layer.__class__ = trt.ITripLimitLayer
            if layer.loop.name not in dLoop.keys():  # search every time because the appearance order of layers in loop is uncertain
                dLoop[layer.loop.name] = {}
                dLoop[layer.loop.name]["RecurrenceLayerName"] = []
                dLoop[layer.loop.name]["LoopOutputLayerName"] = []
                dLoop[layer.loop.name]["IteratorLayerName"] = []
            dLoop[layer.loop.name]["TripLimitLayerName"] = layer.name
            dLayer["kind"] = int(layer.kind)  # save as int

        elif layer.type == trt.LayerType.RECURRENCE:  # 28
            layer.__class__ = trt.IRecurrenceLayer
            if layer.loop.name not in dLoop.keys():  # search every time because the appearance order of layers in loop is uncertain
                dLoop[layer.loop.name] = {}
                dLoop[layer.loop.name]["RecurrenceLayerName"] = []
                dLoop[layer.loop.name]["LoopOutputLayerName"] = []
                dLoop[layer.loop.name]["IteratorLayerName"] = []
            dLoop[layer.loop.name]["RecurrenceLayerName"].append(layer.name)  # a Loop structure could have more than one recurrence layer

        elif layer.type == trt.LayerType.ITERATOR:  # 29
            layer.__class__ = trt.IIteratorLayer
            if layer.loop.name not in dLoop.keys():  # search every time because the appearance order of layers in loop is uncertain
                dLoop[layer.loop.name] = {}
                dLoop[layer.loop.name]["RecurrenceLayerName"] = []
                dLoop[layer.loop.name]["LoopOutputLayerName"] = []
                dLoop[layer.loop.name]["IteratorLayerName"] = []
            dLoop[layer.loop.name]["IteratorLayerName"].append(layer.name)  # a Loop structure could have more than one iterator layer
            dLayer["axis"] = layer.axis
            dLayer["reverse"] = layer.reverse

        elif layer.type == trt.LayerType.LOOP_OUTPUT:  # 30
            layer.__class__ = trt.ILoopOutputLayer
            if layer.loop.name not in dLoop.keys():  # search every time because the appearance order of layers in loop is uncertain
                dLoop[layer.loop.name] = {}
                dLoop[layer.loop.name]["RecurrenceLayerName"] = []
                dLoop[layer.loop.name]["LoopOutputLayerName"] = []
                dLoop[layer.loop.name]["IteratorLayerName"] = []
            dLoop[layer.loop.name]["LoopOutputLayerName"].append(layer.name)  # a Loop structure could have more than one output layer
            dLayer["axis"] = layer.axis
            dLayer["kind"] = int(layer.kind)  # save as int

        elif layer.type == trt.LayerType.SELECT:  # 31
            layer.__class__ = trt.ISelectLayer

        elif layer.type == trt.LayerType.FILL:  # 32
            layer.__class__ = trt.IFillLayer
            dLayer["operation"] = int(layer.operation)  # save as int
            if layer.get_input(0) is not None:  # dynamic fill mode, the shape of output tensor depends on input tenor 0
                dLayer["bDynamicShapeFill"] = True
                dLayer["shape"] = None
            else:  # static fill mode, the shape of output tensor is given by input parameter
                dLayer["bDynamicShapeFill"] = False
                dLayer["shape"] = list(layer.shape)  # save as list

        elif layer.type == trt.LayerType.QUANTIZE:  # 33
            layer.__class__ = trt.IQuantizeLayer
            dLayer["axis"] = layer.axis

        elif layer.type == trt.LayerType.DEQUANTIZE:  # 34
            layer.__class__ = trt.IDequantizeLayer
            dLayer["axis"] = layer.axis

        elif layer.type == trt.LayerType.CONDITION:  # 35
            layer.__class__ = trt.IConditionLayer
            if layer.conditional.name not in dIfCondition.keys():  # search every time because the appearance order of layers in IfCondition is uncertain
                dIfCondition[layer.conditional.name] = {}
            dIfCondition[layer.conditional.name]["ConditionLayerIndex"] = i

        elif layer.type == trt.LayerType.CONDITIONAL_INPUT:  # 36
            layer.__class__ = trt.IIfConditionalInputLayer
            if layer.conditional.name not in dIfCondition.keys():  # search every time because the appearance order of layers in IfCondition is uncertain
                dIfCondition[layer.conditional.name] = {}
            dIfCondition[layer.conditional.name]["InputLayerIndex"] = i

        elif layer.type == trt.LayerType.CONDITIONAL_OUTPUT:  # 37
            layer.__class__ = trt.IIfConditionalOutputLayer
            if layer.conditional.name not in dIfCondition.keys():  # search every time because the appearance order of layers in IfCondition is uncertain
                dIfCondition[layer.conditional.name] = {}
            dIfCondition[layer.conditional.name]["OutputLayerIndex"] = i

        elif layer.type == trt.LayerType.SCATTER:  # 38
            layer.__class__ = trt.IScatterLayer
            dLayer["axis"] = layer.axis
            dLayer["mode"] = int(layer.mode)  # save as int

        elif layer.type == trt.LayerType.EINSUM:  # 39
            layer.__class__ = trt.IEinsumLayer
            dLayer["equation"] = layer.equation

        elif layer.type == trt.LayerType.ASSERTION:  # 40
            layer.__class__ = trt.IAssertionLayer
            dLayer["message"] = layer.message

        else:
            print("Layer not supported!")
            break

        lLayer.append(dLayer)

    return lLayer, dTensor, dIfCondition, dLoop, parameter

def inspectNetwork(builder, builderConfig, network, lOptimizationProfile=[], calibrationProfile=None, bPrintInformation=True, jsonFile="./model.json", paraFile="./model.npz"):

    # print network before parsing
    if bPrintInformation:
        print("\nOriginal network:")
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            print("%4d->%s,in=%d,out=%d,%s" % (i, str(layer.type)[10:], layer.num_inputs, layer.num_outputs, layer.name))
            for j in range(layer.num_inputs):
                tensor = layer.get_input(j)
                if tensor == None:
                    print("\tInput  %2d:" % j, "None")
                else:
                    print("\tInput  %2d:%s,%s,%s" % (j, tensor.shape, str(tensor.dtype)[9:], tensor.name))
            for j in range(layer.num_outputs):
                tensor = layer.get_output(j)
                if tensor == None:
                    print("\tOutput %2d:" % j, "None")
                else:
                    print("\tOutput %2d:%s,%s,%s" % (j, tensor.shape, str(tensor.dtype)[9:], tensor.name))

    bigDictionary = {}

    # Builder, almost useless especially since TensorRT 8
    bigDictionary["Builder"] = extractBuilder(builder, builderConfig, network)

    # BuilderConfig
    bigDictionary["BuilderConfig"] = extractBuilderConfig(builderConfig, network, lOptimizationProfile)

    # Network
    bigDictionary["Network"] = extractNetwork(network)

    # Layer and Tensor
    bigDictionary["Layer"], bigDictionary["Tensor"], bigDictionary["IfCondition"], bigDictionary["Loop"], parameter = exactLayerAndTensor(network)

    # Save result as file
    with open(jsonFile, "w") as f:
        f.write(json.dumps(bigDictionary))

    np.savez(paraFile, **parameter)

    return