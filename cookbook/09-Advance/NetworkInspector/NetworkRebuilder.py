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

from copy import deepcopy
import json
import numpy as np
import tensorrt as trt

def rebuildNetwork(logger, bPrintInformation=True, jsonFile="./model.json", paraFile="./model.npz"):

    with open(jsonFile, "r") as f:
        js = json.loads(f.read())

    if paraFile is not None:
        para = np.load(paraFile)
    else:
        print("Using fake weight!")
        np.random.seed(31193)
        para = None

    # Builder
    if bPrintInformation:
        print("\nJSON Information:")
        for key in js["Builder"].keys():
            print("js[\"Builder\"][\"%s\"] = %s" % (key, js["Builder"][key]))

    builder = trt.Builder(logger)
    builder.max_threads = js["Builder"]["nMaxThread"]

    if int(trt.__version__.split(".")[0]) < 8:  # deprecated since TensorRT 8
        builder.max_batch_size = js["builder"]["nMaxBatchSize"]
        builder.max_workspace_size = js["builder"]["nMaxWorkspaceSize"]

    # Network
    if bPrintInformation:
        for key in js["Network"].keys():
            print("js[\"Network\"][\"%s\"] = %s" % (key, js["Network"][key]))

    networkFlag = 0
    if not js["Network"]["bImplicitBatchMode"]:
        networkFlag |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if js["Network"]["bExplicitPrecision"]:  # deprecated since TensorRT 8.4
        networkFlag |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
    network = builder.create_network(networkFlag)

    network.name = js["Network"]["sName"]
    for i in range(js["Network"]["nInput"]):
        tensorName = js["Network"]["Binding"][i]["sName"]
        inputTensor = js["Tensor"][tensorName]
        network.add_input(tensorName, trt.DataType(inputTensor["kDataType"]), inputTensor["lShape"])

    dTensor = {}  # Dictionary of tensors in new network
    for i in range(js["Network"]["nInput"]):
        tensor = network.get_input(i)
        dTensor[tensor.name] = tensor

    dIfCondition = {}  # Dictionary of IfCondition structure in new network
    dLoop = {}  # Dictionary of Loop structure in new network

    dLateLayerTensor = {}  # In some cases, the shape tensor comsumed in early layer is produced in later layer, so we mark and set them later.

    # Constant layer for Range Node from ONNX file
    constantLayer0 = network.add_constant([], trt.Weights(np.ascontiguousarray(np.array([0], dtype=np.int32))))
    constantLayer0.name = "ConstantLayer0ForRangeNoe"
    constantLayer0.get_output(0).name = "ConstantTensor0ForRangeNoe"
    constantLayer1 = network.add_constant([1], trt.Weights(np.ascontiguousarray(np.array([1], dtype=np.int32))))
    constantLayer1.name = "ConstantLayer1ForRangeNoe"
    constantLayer1.get_output(0).name = "ConstantTensor1ForRangeNoe"

    # rebuild network layer by layer -------------------------------------------
    for i in range(js["Network"]["nLayer"]):
        layerInfo = js["Layer"][i]
        if bPrintInformation:
            print("%4d->%-15s,%s" % (i, str(trt.LayerType(layerInfo["kType"]))[10:], layerInfo["sName"]))
            for key in layerInfo.keys():
                print("      Layer[\"%s\"] = %s" % (key, layerInfo[key]))

        # Specialization part of each layer
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

        if layerInfo["kType"] == int(trt.LayerType.CONVOLUTION):  # 0
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            if para is not None:
                kernel = para[layerInfo["sName"] + "-kernel"]
                bias = para[layerInfo["sName"] + "-bias"]
            else:
                kernelShape = layerInfo["lKernelShape"]
                biasShape = layerInfo["lBiasShape"]
                kernel = np.random.rand(np.prod(kernelShape)).astype(np.float32).reshape(kernelShape) * 2 - 1
                bias = np.random.rand(np.prod(biasShape)).astype(np.float32).reshape(biasShape) * 2 - 1
            if np.prod(kernel.shape) != 0:  # Normal
                if np.prod(bias.shape) != 0:
                    layer = network.add_convolution_nd(inputTensor, layerInfo["num_output_maps"], layerInfo["kernel_size_nd"], trt.Weights(kernel), trt.Weights(bias))
                else:
                    layer = network.add_convolution_nd(inputTensor, layerInfo["num_output_maps"], layerInfo["kernel_size_nd"], trt.Weights(kernel))
            else:  # INT8-QDQ
                assert (layerInfo["nInput"] == 2)
                if np.prod(bias.shape) != 0:
                    layer = network.add_convolution_nd(inputTensor, layerInfo["num_output_maps"], layerInfo["kernel_size_nd"], trt.Weights(), trt.Weights(bias))
                else:
                    layer = network.add_convolution_nd(inputTensor, layerInfo["num_output_maps"], layerInfo["kernel_size_nd"], trt.Weights())
                tensorName1 = layerInfo["lInputTensorName"][1]
                inputTensor1 = dTensor[tensorName1]
                layer.set_input(1, inputTensor1)
            layer.kernel_size_nd = layerInfo["kernel_size_nd"]
            layer.num_output_maps = layerInfo["num_output_maps"]
            layer.stride_nd = layerInfo["stride_nd"]
            layer.dilation_nd = layerInfo["dilation_nd"]
            layer.num_groups = layerInfo["num_groups"]
            layer.padding_nd = layerInfo["padding_nd"]
            layer.padding_mode = trt.PaddingMode(layerInfo["padding_mode"])
            layer.pre_padding = layerInfo["pre_padding"]
            layer.post_padding = layerInfo["post_padding"]

        elif layerInfo["kType"] == int(trt.LayerType.FULLY_CONNECTED):  # 1
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            if para is not None:
                kernel = para[layerInfo["sName"] + "-kernel"]
                bias = para[layerInfo["sName"] + "-bias"]
            else:
                kernelShape = layerInfo["lKernelShape"]
                biasShape = layerInfo["lBiasShape"]
                kernel = np.random.rand(np.prod(kernelShape)).astype(np.float32).reshape(kernelShape) * 2 - 1
                bias = np.random.rand(np.prod(biasShape)).astype(np.float32).reshape(biasShape) * 2 - 1
            if np.prod(kernel.shape) != 0:  # Normal
                if np.prod(bias.shape) != 0:
                    layer = network.add_fully_connected(inputTensor, layerInfo["num_output_channels"], trt.Weights(kernel), trt.Weights(bias))
                else:
                    layer = network.add_fully_connected(inputTensor, layerInfo["num_output_channels"], trt.Weights(kernel))
            else:  # INT8-QDQ
                assert (layerInfo["nInput"] == 2)
                if np.prod(bias.shape) != 0:
                    layer = network.add_fully_connected(inputTensor, layerInfo["num_output_channels"], trt.Weights(), trt.Weights(bias))
                else:
                    layer = network.add_fully_connected(inputTensor, layerInfo["num_output_channels"], trt.Weights())
                tensorName = layerInfo["lInputTensorName"][1]
                inputTensor = dTensor[tensorName]
                layer.set_input(1, inputTensor)
            layer.num_output_channels = layerInfo["num_output_channels"]

        elif layerInfo["kType"] == int(trt.LayerType.ACTIVATION):  # 2
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = network.add_activation(inputTensor, trt.ActivationType.RELU)
            layer.alpha = layerInfo["alpha"]
            layer.beta = layerInfo["beta"]
            layer.type = trt.ActivationType(layerInfo["type"])

        elif layerInfo["kType"] == int(trt.LayerType.POOLING):  # 3
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = network.add_pooling_nd(inputTensor, trt.PoolingType.MAX, [1, 1])
            layer.average_count_excludes_padding = layerInfo["average_count_excludes_padding"]
            layer.blend_factor = layerInfo["blend_factor"]
            layer.stride_nd = layerInfo["stride_nd"]
            layer.padding_nd = layerInfo["padding_nd"]
            layer.padding_mode = trt.PaddingMode(layerInfo["padding_mode"])
            layer.pre_padding = layerInfo["pre_padding"]
            layer.post_padding = layerInfo["post_padding"]
            layer.type = trt.PoolingType(layerInfo["type"])
            layer.window_size_nd = layerInfo["window_size_nd"]

        elif layerInfo["kType"] == int(trt.LayerType.LRN):  # 4
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = network.add_lrn(inputTensor, 1, 0.0, 1.0, 1.0)
            layer.alpha = layerInfo["alpha"]
            layer.beta = layerInfo["beta"]
            layer.k = layerInfo["k"]
            layer.window_size = layerInfo["window_size"]

        elif layerInfo["kType"] == int(trt.LayerType.SCALE):  # 5
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            if para is not None:
                scale = para[layerInfo["sName"] + "-scale"]
                shift = para[layerInfo["sName"] + "-shift"]
                power = para[layerInfo["sName"] + "-power"]
            else:
                scaleShape = layerInfo["lScaleShape"]
                shiftShape = layerInfo["lShiftShape"]
                powerShape = layerInfo["lPowerShape"]
                scale = np.random.rand(np.prod(scaleShape)).astype(np.float32).reshape(scaleShape) * 2 - 1
                shift = np.random.rand(np.prod(shiftShape)).astype(np.float32).reshape(shiftShape) * 2 - 1
                power = np.ones(np.prod(powerShape)).astype(np.float32).reshape(powerShape)
            layer = network.add_scale_nd(inputTensor, trt.ScaleMode(layerInfo["mode"]), shift, scale, power, layerInfo["channel_axis"])
            layer.channel_axis = layerInfo["channel_axis"]
            layer.mode = trt.ScaleMode(layerInfo["mode"])
            layer.shift = shift
            layer.scale = scale
            layer.power = power

        elif layerInfo["kType"] == int(trt.LayerType.SOFTMAX):  # 6
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = network.add_softmax(inputTensor)
            layer.axes = layerInfo["axes"]

        elif layerInfo["kType"] == int(trt.LayerType.DECONVOLUTION):  # 7
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            if para is not None:
                kernel = para[layerInfo["sName"] + "-kernel"]
                bias = para[layerInfo["sName"] + "-bias"]
            else:
                kernelShape = layerInfo["lKernelShape"]
                biasShape = layerInfo["lBiasShape"]
                kernel = np.random.rand(np.prod(kernelShape)).astype(np.float32).reshape(kernelShape) * 2 - 1
                bias = np.random.rand(np.prod(biasShape)).astype(np.float32).reshape(biasShape) * 2 - 1
            if np.prod(kernel.shape) != 0:  # Normal
                if np.prod(bias.shape) != 0:
                    layer = network.add_deconvolution_nd(inputTensor, 1, [1, 1], trt.Weights(kernel), trt.Weights(bias))
                else:
                    layer = network.add_deconvolution_nd(inputTensor, 1, [1, 1], trt.Weights(kernel))
            else:  # INT8-QDQ
                assert (layerInfo["nInput"] == 2)
                if np.prod(bias.shape) != 0:
                    layer = network.add_deconvolution_nd(inputTensor, 1, [1, 1], trt.Weights(), trt.Weights(bias))
                else:
                    layer = network.add_deconvolution_nd(inputTensor, 1, [1, 1], trt.Weights())
            layer.kernel_size_nd = layerInfo["kernel_size_nd"]
            layer.num_output_maps = layerInfo["num_output_maps"]
            layer.stride_nd = layerInfo["stride_nd"]
            layer.dilation_nd = layerInfo["dilation_nd"]
            layer.num_groups = layerInfo["num_groups"]
            layer.padding_nd = layerInfo["padding_nd"]
            layer.padding_mode = trt.PaddingMode(layerInfo["padding_mode"])
            layer.pre_padding = layerInfo["pre_padding"]
            layer.post_padding = layerInfo["post_padding"]

        elif layerInfo["kType"] == int(trt.LayerType.CONCATENATION):  # 8
            inputTensorList = []
            for j in range(layerInfo["nInput"]):
                tensorName = layerInfo["lInputTensorName"][j]
                inputTensorList.append(dTensor[tensorName])
            layer = network.add_concatenation(inputTensorList)
            layer.axis = layerInfo["axis"]

        elif layerInfo["kType"] == int(trt.LayerType.ELEMENTWISE):  # 9
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor0 = dTensor[tensorName]
            tensorName = layerInfo["lInputTensorName"][1]
            inputTensor1 = dTensor[tensorName]
            layer = network.add_elementwise(inputTensor0, inputTensor1, trt.ElementWiseOperation.SUM)
            layer.op = trt.ElementWiseOperation(layerInfo["op"])

        elif layerInfo["kType"] == int(trt.LayerType.PLUGIN):  # 10
            print("IPlugin Layer not supported!")
            break

        elif layerInfo["kType"] == int(trt.LayerType.UNARY):  # 11
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = network.add_unary(inputTensor, trt.UnaryOperation.ABS)
            layer.op = trt.UnaryOperation(layerInfo["op"])

        elif layerInfo["kType"] == int(trt.LayerType.PADDING):  # 12
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = network.add_padding_nd(inputTensor, (0, 0), (0, 0))
            layer.pre_padding_nd = layerInfo["pre_padding_nd"]
            layer.post_padding_nd = layerInfo["post_padding_nd"]

        elif layerInfo["kType"] == int(trt.LayerType.SHUFFLE):  # 13
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = network.add_shuffle(inputTensor)
            if layerInfo["bDynamicShuffle"]:
                tensorName = layerInfo["lInputTensorName"][1]
                if tensorName in dTensor.keys():  # In some cases, the shape tensor comsumed in early layer is produced in later layer, so we mark and set them later.
                    inputTensor = dTensor[tensorName]
                    layer.set_input(1, inputTensor)
                else:
                    dLateLayerTensor[layerInfo["kIndex"]] = tensorName
            else:
                if layerInfo["reshape_dims"] is not None:
                    layer.reshape_dims = layerInfo["reshape_dims"]
            layer.first_transpose = layerInfo["first_transpose"]
            layer.second_transpose = layerInfo["second_transpose"]
            layer.zero_is_placeholder = layerInfo["zero_is_placeholder"]

        elif layerInfo["kType"] == int(trt.LayerType.REDUCE):  # 14
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = network.add_reduce(inputTensor, trt.ReduceOperation.SUM, 1 << 1, False)
            layer.axes = layerInfo["axes"]
            layer.op = trt.ReduceOperation(layerInfo["op"])
            layer.keep_dims = layerInfo["keep_dims"]

        elif layerInfo["kType"] == int(trt.LayerType.TOPK):  # 15
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = network.add_topk(inputTensor, trt.TopKOperation.MAX, 2, 1 << 1)
            layer.axes = layerInfo["axes"]
            layer.op = trt.TopKOperation(layerInfo["op"])
            layer.k = layerInfo["k"]

        elif layerInfo["kType"] == int(trt.LayerType.GATHER):  # 16
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor0 = dTensor[tensorName]
            tensorName = layerInfo["lInputTensorName"][1]
            inputTensor1 = dTensor[tensorName]
            layer = network.add_gather(inputTensor0, inputTensor1, 1)
            layer.axis = layerInfo["axis"]
            layer.mode = trt.GatherMode(layerInfo["mode"])

        elif layerInfo["kType"] == int(trt.LayerType.MATRIX_MULTIPLY):  # 17
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor0 = dTensor[tensorName]
            tensorName = layerInfo["lInputTensorName"][1]
            inputTensor1 = dTensor[tensorName]
            layer = network.add_matrix_multiply(inputTensor0, trt.MatrixOperation.NONE, inputTensor1, trt.MatrixOperation.NONE)
            layer.op0 = trt.MatrixOperation(layerInfo["op0"])
            layer.op1 = trt.MatrixOperation(layerInfo["op1"])

        elif layerInfo["kType"] == int(trt.LayerType.RAGGED_SOFTMAX):  # 18
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor0 = dTensor[tensorName]
            tensorName = layerInfo["lInputTensorName"][1]
            inputTensor1 = dTensor[tensorName]
            layer = network.add_ragged_softmax(inputTensor0, inputTensor1)

        elif layerInfo["kType"] == int(trt.LayerType.CONSTANT):  # 19
            weightShape = layerInfo["shape"]
            if weightShape == [0]:
                layer = network.add_constant([0], trt.Weights())
            else:
                if para is not None:
                    weight = para[layerInfo["sName"] + "-weights"]
                else:
                    weight = np.random.rand(np.prod(weightShape)).astype(np.float32).reshape(weightShape)
                layer = network.add_constant(weightShape, trt.Weights(weight))

        elif layerInfo["kType"] == int(trt.LayerType.RNN_V2):  # 20
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = network.add_rnn_v2(inputTensor, layerInfo["num_layers"], layerInfo["hidden_size"], layerInfo["max_seq_length"], trt.RNNOperation(layerInfo["op"]))
            #layer.num_layers = layerInfo["num_layers"]  # read only
            #layer.hidden_size = layerInfo["hidden_size"]  # read only
            #layer.max_seq_length = layerInfo["max_seq_length"]  # read only
            layer.op = trt.RNNOperation(layerInfo["op"])
            #layer.data_length = layerInfo["data_length"]  # read only
            layer.input_mode = trt.RNNInputMode(layerInfo["input_mode"])
            layer.direction = trt.RNNDirection(layerInfo["direction"])
            tensorName = layerInfo["lInputTensorName"][1]
            if tensorName is not None:
                inputTensor = dTensor[tensorName]
                layer.hidden_state = inputTensor
            tensorName = layerInfo["lInputTensorName"][2]
            if tensorName is not None:
                inputTensor = dTensor[tensorName]
                layer.cell_state = inputTensor
            tensorName = layerInfo["lInputTensorName"][3]
            if tensorName is not None:
                inputTensor = dTensor[tensorName]
                layer.seq_lengths = inputTensor
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
                        layer.set_weights_for_gate(j, gateKind, True, para[layerInfo["sName"] + "-" + str(j) + "-" + str(int(gateKind)) + "-weightX"])
                    layer.set_bias_for_gate(j, gateKind, True, para[layerInfo["sName"] + "-" + str(j) + "-" + str(int(gateKind)) + "-biasX"])
                    layer.set_weights_for_gate(j, gateKind, False, para[layerInfo["sName"] + "-" + str(j) + "-weightH"])
                    layer.set_bias_for_gate(j, gateKind, False, para[layerInfo["sName"] + "-" + str(j) + "-biasH"])

        elif layerInfo["kType"] == int(trt.LayerType.IDENTITY):  # 21
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = network.add_identity(inputTensor)

        elif layerInfo["kType"] == int(trt.LayerType.PLUGIN_V2):  # 22
            print("IPlugin Layer not supported!")
            break

        elif layerInfo["kType"] == int(trt.LayerType.SLICE):  # 23
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = network.add_slice(inputTensor, [0], [1], [1])
            layer.mode = trt.SliceMode(layerInfo["mode"])
            if layerInfo["start"] == None:
                tensorName = layerInfo["lInputTensorName"][1]
                inputTensor = dTensor[tensorName]
                layer.set_input(1, inputTensor)
            else:
                layer.start = layerInfo["start"]
            if layerInfo["shape"] == None:
                tensorName = layerInfo["lInputTensorName"][2]
                inputTensor = dTensor[tensorName]
                layer.set_input(2, inputTensor)
            else:
                layer.shape = layerInfo["shape"]
            if layerInfo["stride"] == None:
                tensorName = layerInfo["lInputTensorName"][3]
                inputTensor = dTensor[tensorName]
                layer.set_input(3, inputTensor)
            else:
                layer.stride = layerInfo["stride"]
            if trt.SliceMode(layerInfo["mode"]) == trt.SliceMode.FILL and layerInfo["fill"] == True:
                tensorName = layerInfo["lInputTensorName"][4]
                inputTensor = dTensor[tensorName]
                layer.set_input(4, inputTensor)

        elif layerInfo["kType"] == int(trt.LayerType.SHAPE):  # 24
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = network.add_shape(inputTensor)

        elif layerInfo["kType"] == int(trt.LayerType.PARAMETRIC_RELU):  # 25
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor0 = dTensor[tensorName]
            tensorName = layerInfo["lInputTensorName"][1]
            inputTensor1 = dTensor[tensorName]
            layer = network.add_parametric_relu(inputTensor0, inputTensor1)

        elif layerInfo["kType"] == int(trt.LayerType.RESIZE):  # 26
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = network.add_resize(inputTensor)
            if layerInfo["bDynamicResize"]:
                tensorName = layerInfo["lInputTensorName"][1]
                inputTensor = dTensor[tensorName]
                layer.set_input(1, inputTensor)
            elif layerInfo["bShapeMode"]:
                layer.shape = layerInfo["shape"]
            else:
                layer.scales = layerInfo["scales"]
            layer.resize_mode = trt.ResizeMode(layerInfo["resize_mode"])
            layer.coordinate_transformation = trt.ResizeCoordinateTransformation(layerInfo["coordinate_transformation"])
            layer.selector_for_single_pixel = trt.ResizeSelector(layerInfo["selector_for_single_pixel"])
            layer.nearest_rounding = trt.ResizeRoundMode(layerInfo["nearest_rounding"])

        elif layerInfo["kType"] == int(trt.LayerType.TRIP_LIMIT):  # 27
            bExist = False
            for key, value in dLoop.items():  # find if the Loop already exists in the new network
                if value["TripLimitLayerName"] == layerInfo["sName"]:
                    bExist = True
                    sLoopName = key
            if not bExist:  # not exist, add a new Loop structure
                for key, value in js["Loop"].items():
                    if value["TripLimitLayerName"] == layerInfo["sName"]:
                        dLoop[key] = {}
                        dLoop[key]["TripLimitLayerName"] = layerInfo["sName"]
                        dLoop[key]["RecurrenceLayerName"] = value["RecurrenceLayerName"]
                        dLoop[key]["LoopOutputLayerName"] = value["LoopOutputLayerName"]
                        dLoop[key]["IteratorLayerName"] = value["IteratorLayerName"]
                        dLoop[key]["RecurrenceLayer"] = []
                        dLoop[key]["LoopOutputLayer"] = []
                        dLoop[key]["IteratorLayer"] = []
                        dLoop[key]["Loop"] = network.add_loop()
                        sLoopName = key
                        break
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = dLoop[sLoopName]["Loop"].add_trip_limit(inputTensor, trt.TripLimit(layerInfo["kind"]))
            dLoop[sLoopName]["TripLimitLayer"] = layer

        elif layerInfo["kType"] == int(trt.LayerType.RECURRENCE):  # 28
            bExist = False
            for key, value in dLoop.items():  # find if the Loop already exists in the new network
                if layerInfo["sName"] in value["RecurrenceLayerName"]:
                    bExist = True
                    sLoopName = key
            if not bExist:  # not exist, add a new Loop structure
                for key, value in js["Loop"].items():
                    if layerInfo["sName"] in value["RecurrenceLayerName"]:
                        dLoop[key] = {}
                        dLoop[key]["TripLimitLayerName"] = value["TripLimitLayerName"]
                        dLoop[key]["RecurrenceLayerName"] = value["RecurrenceLayerName"]
                        dLoop[key]["LoopOutputLayerName"] = value["LoopOutputLayerName"]
                        dLoop[key]["IteratorLayerName"] = value["IteratorLayerName"]
                        dLoop[key]["RecurrenceLayer"] = []
                        dLoop[key]["LoopOutputLayer"] = []
                        dLoop[key]["IteratorLayer"] = []
                        dLoop[key]["Loop"] = network.add_loop()
                        sLoopName = key
                        break
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = dLoop[sLoopName]["Loop"].add_recurrence(inputTensor)  # the second input tensor is recycled in the later additional scan
            dLoop[sLoopName]["RecurrenceLayer"].append(layer)  # append rather than assign

        elif layerInfo["kType"] == int(trt.LayerType.ITERATOR):  # 29
            bExist = False
            for key, value in dLoop.items():  # find if the Loop already exists in the new network
                if layerInfo["sName"] in value["IteratorLayerName"]:
                    bExist = True
                    sLoopName = key
            if not bExist:  # not exist, add a new Loop structure
                for key, value in js["Loop"].items():
                    if layerInfo["sName"] in value["IteratorLayerName"]:
                        dLoop[key] = {}
                        dLoop[key]["TripLimitLayerName"] = value["TripLimitLayerName"]
                        dLoop[key]["RecurrenceLayerName"] = value["RecurrenceLayerName"]
                        dLoop[key]["LoopOutputLayerName"] = value["LoopOutputLayerName"]
                        dLoop[key]["IteratorLayerName"] = value["IteratorLayerName"]
                        dLoop[key]["RecurrenceLayer"] = []
                        dLoop[key]["LoopOutputLayer"] = []
                        dLoop[key]["IteratorLayer"] = []
                        dLoop[key]["Loop"] = network.add_loop()
                        sLoopName = key
                        break
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = dLoop[sLoopName]["Loop"].add_iterator(inputTensor, layerInfo["axis"], layerInfo["reverse"])
            layer.axis = layerInfo["axis"]
            #layer.reverse = layerInfo["reverse"]  # read only
            dLoop[sLoopName]["IteratorLayer"].append(layer)  # append rather than assign

        elif layerInfo["kType"] == int(trt.LayerType.LOOP_OUTPUT):  # 30
            bExist = False
            for key, value in dLoop.items():  # find if the Loop already exists in the new network
                if layerInfo["sName"] in value["LoopOutputLayerName"]:
                    bExist = True
                    sLoopName = key
            if not bExist:  # not exist, add a new Loop structure
                for key, value in js["Loop"].items():
                    if layerInfo["sName"] in value["LoopOutputLayerName"]:
                        dLoop[key] = {}
                        dLoop[key]["TripLimitLayerName"] = value["TripLimitLayerName"]
                        dLoop[key]["RecurrenceLayerName"] = value["RecurrenceLayerName"]
                        dLoop[key]["LoopOutputLayerName"] = value["LoopOutputLayerName"]
                        dLoop[key]["IteratorLayerName"] = value["IteratorLayerName"]
                        dLoop[key]["RecurrenceLayer"] = []
                        dLoop[key]["LoopOutputLayer"] = []
                        dLoop[key]["IteratorLayer"] = []
                        dLoop[key]["Loop"] = network.add_loop()
                        sLoopName = key
                        break
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = dLoop[sLoopName]["Loop"].add_loop_output(inputTensor, trt.LoopOutput(layerInfo["kind"]), layerInfo["axis"])  # the optinal second input tensor is recycled in the later additional scan
            layer.axis = layerInfo["axis"]
            #layer.kind = trt.LoopOutput(layerInfo["kind"])  # read only
            dLoop[sLoopName]["LoopOutputLayer"].append(layer)  # append rather than assign

        elif layerInfo["kType"] == int(trt.LayerType.SELECT):  # 31
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor0 = dTensor[tensorName]
            tensorName = layerInfo["lInputTensorName"][1]
            inputTensor1 = dTensor[tensorName]
            tensorName = layerInfo["lInputTensorName"][2]
            inputTensor2 = dTensor[tensorName]
            layer = network.add_select(inputTensor0, inputTensor1, inputTensor2)

        elif layerInfo["kType"] == int(trt.LayerType.FILL):  # 32
            layer = network.add_fill([1], trt.FillOperation(layerInfo["operation"]))
            layer.operation = trt.FillOperation(layerInfo["operation"])
            if layerInfo["bDynamicShapeFill"]:
                tensorName = layerInfo["lInputTensorName"][0]
                inputTensor = dTensor[tensorName]
                layer.set_input(0, inputTensor)
            else:
                layer.shape = layerInfo["shape"]
            if layerInfo["nInput"] >= 2:
                tensorName = layerInfo["lInputTensorName"][1]
                inputTensor = dTensor[tensorName]
                layer.set_input(1, inputTensor)
                tensorName = layerInfo["lInputTensorName"][2]
                inputTensor = dTensor[tensorName]
                layer.set_input(2, inputTensor)
            if "Range" in layerInfo["sName"]:  # The special case: parse Range node from ONNX
                layer.set_input(1, constantLayer0.get_output(0))
                layer.set_input(2, constantLayer1.get_output(0))

        elif layerInfo["kType"] == int(trt.LayerType.QUANTIZE):  # 33
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor0 = dTensor[tensorName]
            tensorName = layerInfo["lInputTensorName"][1]
            inputTensor1 = dTensor[tensorName]
            layer = network.add_quantize(inputTensor0, inputTensor1)
            if layerInfo["nInput"] == 3:
                tensorName = layerInfo["lInputTensorName"][2]
                inputTensor2 = dTensor[tensorName]
                layer.set_input(2, inputTensor2)
            #layer.axis = layerInfo["axis"]  # TODO: layerInfo["axis"] is always "-1", but not supported by TensorRT
            if layerInfo["axis"] != -1:
                layer.axis = layerInfo["axis"]
            else:
                layer.axis = 0  # change it into 0, per-tensor Quantization/Dequantization

        elif layerInfo["kType"] == int(trt.LayerType.DEQUANTIZE):  # 34
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor0 = dTensor[tensorName]
            tensorName = layerInfo["lInputTensorName"][1]
            inputTensor1 = dTensor[tensorName]
            layer = network.add_dequantize(inputTensor0, inputTensor1)
            #layer.axis = layerInfo["axis"]  # TODO: layerInfo["axis"] is always "-1", but not supported by TensorRT
            if layerInfo["axis"] != -1:
                layer.axis = layerInfo["axis"]
            else:
                layer.axis = 0  # change it into 0, per-tensor Quantization/Dequantization

        elif layerInfo["kType"] == int(trt.LayerType.CONDITION):  # 35
            bExist = False
            for key, value in dIfCondition.items():  # find if the IfCondition already exists in the new network
                if value["ConditionLayerIndex"] == i:
                    bExist = True
                    sIfConditionName = key
            if not bExist:  # not exist, add a new IfCondition structure
                for key, value in js["IfCondition"].items():
                    if value["InputLayerIndex"] == i:
                        dIfCondition[key] = {}
                        dIfCondition[key]["ConditionLayerIndex"] = i
                        dIfCondition[key]["InputLayerIndex"] = value["InputLayerIndex"]
                        dIfCondition[key]["OutputLayerIndex"] = value["OutputLayerIndex"]
                        dIfCondition[key]["IfCondition"] = network.add_if_conditional()
                        sIfConditionName = key
                        break
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = dIfCondition[sIfConditionName]["IfCondition"].set_condition(inputTensor)
            dIfCondition[sIfConditionName]["ConditionLayer"] = layer

        elif layerInfo["kType"] == int(trt.LayerType.CONDITIONAL_INPUT):  # 36
            bExist = False
            for key, value in dIfCondition.items():  # find if the IfCondition already exists in the new network
                if value["InputLayerIndex"] == i:
                    bExist = True
                    sIfConditionName = key
            if not bExist:  # not exist, add a new IfCondition structure
                for key, value in js["IfCondition"].items():
                    if value["InputLayerIndex"] == i:
                        dIfCondition[key] = {}
                        dIfCondition[key]["ConditionLayerIndex"] = value["ConditionLayerIndex"]
                        dIfCondition[key]["InputLayerIndex"] = i
                        dIfCondition[key]["OutputLayerIndex"] = value["OutputLayerIndex"]
                        dIfCondition[key]["IfCondition"] = network.add_if_conditional()
                        sIfConditionName = key
                        break
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = dIfCondition[sIfConditionName]["IfCondition"].add_input(inputTensor)
            dIfCondition[sIfConditionName]["InputLayer"] = layer

        elif layerInfo["kType"] == int(trt.LayerType.CONDITIONAL_OUTPUT):  # 37
            bExist = False
            for key, value in dIfCondition.items():  # find if the IfCondition already exists in the new network
                if value["OutputLayerIndex"] == i:
                    bExist = True
                    sIfConditionName = key
            if not bExist:  # not exist, add a new IfCondition structure
                for key, value in js["IfCondiftion"].items():
                    if value["InputLayerIndex"] == i:
                        dIfCondition[key] = {}
                        dIfCondition[key]["ConditionLayerIndex"] = value["ConditionLayerIndex"]
                        dIfCondition[key]["InputLayerIndex"] = value["InputLayerIndex"]
                        dIfCondition[key]["OutputLayerIndex"] = i
                        dIfCondition[key]["IfCondition"] = network.add_if_conditional()
                        sIfConditionName = key
                        break
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor0 = dTensor[tensorName]
            tensorName = layerInfo["lInputTensorName"][1]
            inputTensor1 = dTensor[tensorName]
            layer = dIfCondition[sIfConditionName]["IfCondition"].add_output(inputTensor0, inputTensor1)
            dIfCondition[sIfConditionName]["InputLayer"] = layer

        elif layerInfo["kType"] == int(trt.LayerType.SCATTER):  # 38
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor0 = dTensor[tensorName]
            tensorName = layerInfo["lInputTensorName"][1]
            inputTensor1 = dTensor[tensorName]
            tensorName = layerInfo["lInputTensorName"][2]
            inputTensor2 = dTensor[tensorName]
            layer = network.add_scatter(inputTensor0, inputTensor1, inputTensor2, trt.ScatterMode.ELEMENT)
            layer.axis = layerInfo["axis"]
            layer.mode = trt.ScatterMode(layerInfo["mode"])

        elif layerInfo["kType"] == int(trt.LayerType.EINSUM):  # 39
            inputTensorList = []
            for j in range(layerInfo["nInput"]):
                tensorName = layerInfo["lInputTensorName"][j]
                inputTensorList.append(dTensor[tensorName])
            layer = network.add_einsum(inputTensorList, layerInfo["equation"])
            layer.equation = layerInfo["equation"]

        elif layerInfo["kType"] == int(trt.LayerType.ASSERTION):  # 40
            tensorName = layerInfo["lInputTensorName"][0]
            inputTensor = dTensor[tensorName]
            layer = network.add_assertion(inputTensor, "Error message")
            layer.message = layerInfo["message"]

        # Common part of each layer
        layer.name = layerInfo["sName"]
        if layerInfo["bPrecisionIsSet"]:
            layer.precision = trt.DataType(layerInfo["kPrecision"])

        for j in range(layerInfo["nOutput"]):
            #layer.set_output_type(j, trt.DataType(layerInfo["lOutputTensorDataType"][j])) # remove the use of set_output_type
            outputTensor = layer.get_output(j)
            referenceTensor = js["Tensor"][layerInfo["lOutputTensorName"][j]]
            outputTensor.name = layerInfo["lOutputTensorName"][j]
            outputTensor.dtype = trt.DataType(referenceTensor["kDataType"])
            layer.set_output_type(j, trt.DataType(trt.DataType(referenceTensor["kDataType"])))  # Use data type of referenceTensor as standard
            outputTensor.location = trt.TensorLocation(referenceTensor["kLocation"])
            if outputTensor.dtype == trt.DataType.FLOAT and referenceTensor["nAllowedFormat"] != 4095:  # The default value of a tensor's allowed_formats is 4095, meaning no more constraint is set
                formatBitMask = referenceTensor["nAllowedFormat"] & \
                    (1 << int(trt.TensorFormat.LINEAR) | 1 << int(trt.TensorFormat.CHW32) | 1 << int(trt.TensorFormat.HWC))
            elif outputTensor.dtype == trt.DataType.HALF and referenceTensor["nAllowedFormat"] != 4095:
                formatBitMask = referenceTensor["nAllowedFormat"] & \
                    (1 << int(trt.TensorFormat.LINEAR) | 1 << int(trt.TensorFormat.CHW2) | 1 << int(trt.TensorFormat.HWC8) | \
                        1 << int(trt.TensorFormat.CHW4) | 1 << int(trt.TensorFormat.CHW16) | 1 << int(trt.TensorFormat.CHW32) | \
                        1 << int(trt.TensorFormat.DHWC8) | 1 << int(trt.TensorFormat.CDHW32) | 1 << int(trt.TensorFormat.HWC16))
            elif outputTensor.dtype == trt.DataType.INT32 and referenceTensor["nAllowedFormat"] != 4095:
                formatBitMask = referenceTensor["nAllowedFormat"] & \
                    (1 << int(trt.TensorFormat.LINEAR) | 1 << int(trt.TensorFormat.CHW32))
            elif outputTensor.dtype == trt.DataType.INT8 and referenceTensor["nAllowedFormat"] != 4095:
                formatBitMask = referenceTensor["nAllowedFormat"] & \
                    (1 << int(trt.TensorFormat.LINEAR) | 1 << int(trt.TensorFormat.CHW4) | 1 << int(trt.TensorFormat.CHW32) | 1 << int(trt.TensorFormat.CDHW32))
            else:  # bool
                formatBitMask = referenceTensor["nAllowedFormat"] & \
                    (1 << int(trt.TensorFormat.LINEAR))
            outputTensor.allowed_formats = formatBitMask
            if referenceTensor["lDynamicRange"] is not None:
                outputTensor.dynamic_range = referenceTensor["lDynamicRange"]
            dTensor[outputTensor.name] = outputTensor

    # Addition scan, recycle the second input tensor of Shuffle layer
    for key, value in dLateLayerTensor.items():
        layer = network.get_layer(i)
        if tensorName in dTensor.keys():
            inputTensor = dTensor[tensorName]
            layer.set_input(1, inputTensor)
        else:
            print("Error finding tensor %s" % tensorName)

    # Addition scan, recycle the second input tensor of recurrence layer or output lauer in Loop structure
    for key, value in dLoop.items():
        for recurrenceLayer in dLoop[key]["RecurrenceLayer"]:
            for i in range(js["Network"]["nLayer"]):
                if js["Layer"][i]["sName"] == recurrenceLayer.name:
                    tensorName = js["Layer"][i]["lInputTensorName"][1]
                    break
            inputTensor = dTensor[tensorName]
            recurrenceLayer.set_input(1, inputTensor)

        for outputLayer in dLoop[key]["LoopOutputLayer"]:
            if outputLayer.kind == trt.LoopOutput.LAST_VALUE:  # only CONCATENTE and REVERSE mode need the second input tensor
                continue
            for i in range(js["Network"]["nLayer"]):
                if js["Layer"][i]["sName"] == outputLayer.name:
                    tensorName = js["Layer"][i]["lInputTensorName"][1]
                    break
            inputTensor = dTensor[tensorName]
            outputLayer.set_input(1, inputTensor)

    # mark output tensor
    for i in range(js["Network"]["nInput"], js["Network"]["nInput"] + js["Network"]["nOutput"]):
        tensorName = js["Network"]["Binding"][i]["sName"]
        outputTensor = dTensor[tensorName]
        assert (outputTensor.name in js["Tensor"])
        network.mark_output(outputTensor)

    # BuilderConfig
    if bPrintInformation:
        for key in js["BuilderConfig"].keys():
            print("js[\"BuilderConfig\"][\"%s\"] = %s" % (key, js["BuilderConfig"][key]))

    config = builder.create_builder_config()
    if int(trt.__version__.split(".")[0]) < 8:  # deprecated since TensorRT 8
        config.max_workspace_size = js["BuilderConfig"]["nMaxWorkspaceSize"]
    else:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, js["BuilderConfig"]["nMemoryPoolLimit"]["WORKSPACE"])
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, js["BuilderConfig"]["nMemoryPoolLimit"]["DLA_MANAGED_SRAM"])
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, js["BuilderConfig"]["nMemoryPoolLimit"]["DLA_LOCAL_DRAM"])
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, js["BuilderConfig"]["nMemoryPoolLimit"]["DLA_GLOBAL_DRAM"])

    config.flags = js["BuilderConfig"]["nFlag"]
    config.quantization_flags = js["BuilderConfig"]["nQuantizationFlag"]
    config.engine_capability = trt.EngineCapability(js["BuilderConfig"]["kEngineCapability"])

    config.profile_stream = js["BuilderConfig"]["nProfileStream"]
    config.profiling_verbosity = trt.ProfilingVerbosity(js["BuilderConfig"]["kProfilingVerbosity"])
    config.avg_timing_iterations = js["BuilderConfig"]["nAverageTimingIteration"]
    config.set_tactic_sources(js["BuilderConfig"]["nTacticSource"])

    for i in range(js["BuilderConfig"]["nOptimizationProfile"]):
        op = js["BuilderConfig"]["lOptimizationProfile"][i]
        optimizationProfile = builder.create_optimization_profile()
        for j in range(network.num_inputs):
            if op[j] == {}:
                continue
            elif op[j]["bShapeTensor"]:
                optimizationProfile.set_shape_input(op[j]["name"], op[j]["min"], op[j]["opt"], op[j]["max"])
            else:
                optimizationProfile.set_shape(op[j]["name"], op[j]["min"], op[j]["opt"], op[j]["max"])
        config.add_optimization_profile(optimizationProfile)

    # print network before building
    if bPrintInformation:
        print("\nFinal network:")
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

    return builder.build_serialized_network(network, config)