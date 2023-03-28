/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "cnpy.h"
#include "cookbookHelper.cuh"
#include "json.hpp"
#include "json_fwd.hpp"

using json = nlohmann::json;

int extractBuilder(json &js, const IBuilder *builder, const IBuilderConfig *builderConfig, const INetworkDefinition *network)
{
    json dBuilder;
    dBuilder["nMaxThread"]           = builder->getMaxThreads();
    dBuilder["bPlatformHasTF32"]     = builder->platformHasTf32();
    dBuilder["bPlatformHasFastFP16"] = builder->platformHasFastFp16();
    dBuilder["bPlatformHasFastINT8"] = builder->platformHasFastInt8();
    dBuilder["nMaxDLABatchSize"]     = builder->getMaxDLABatchSize();
    dBuilder["nDLACore"]             = builder->getNbDLACores();
    dBuilder["bNetworkSupported"]    = builder->isNetworkSupported(*network, *builderConfig);

    if (int(NV_TENSORRT_MAJOR) < 8) // deprecated since TensorRT 8
    {
        dBuilder["nMaxBatchSize"] = builder->getMaxBatchSize();
        //dBuilder["nMaxWorkspaceSize"]=builder->getMaxWorkspaceSize(); // removed since TensorRT 8
    }

    js["Builder"] = dBuilder;

    return 0;
}

// BuilderConfig
int extractBuilderConfig(json &                                     js,
                         const IBuilderConfig *                     builderConfig,
                         const INetworkDefinition *                 network,
                         const std::vector<IOptimizationProfile *> &lOptimizationProfile)
{
    json dBuilderConfig;

    if (int(NV_TENSORRT_MAJOR) < 8) // deprecated since TensorRT 8
    {
        ;
        //dBuilder["nMaxWorkspaceSize"]=builder->getMaxWorkspaceSize(); // removed since TensorRT 8
    }
    else
    {
        json dMemory; // Dictionary of Memory management
        dMemory["WORKSPACE"]               = builderConfig->getMemoryPoolLimit(MemoryPoolType::kWORKSPACE);
        dMemory["DLA_MANAGED_SRAM"]        = builderConfig->getMemoryPoolLimit(MemoryPoolType::kDLA_MANAGED_SRAM);
        dMemory["DLA_LOCAL_DRAM"]          = builderConfig->getMemoryPoolLimit(MemoryPoolType::kDLA_LOCAL_DRAM);
        dMemory["DLA_GLOBAL_DRAM"]         = builderConfig->getMemoryPoolLimit(MemoryPoolType::kDLA_GLOBAL_DRAM);
        dBuilderConfig["nMemoryPoolLimit"] = dMemory;
    }

    dBuilderConfig["kDefaultDeviceType"] = int(builderConfig->getDefaultDeviceType());
    dBuilderConfig["nDLACore"]           = builderConfig->getDLACore();

    dBuilderConfig["kEngineCapability"]       = int(builderConfig->getEngineCapability());
    dBuilderConfig["nFlag"]                   = builderConfig->getFlags();
    dBuilderConfig["nQuantizationFlag"]       = int(builderConfig->getQuantizationFlags());
    dBuilderConfig["nOptimizationProfile"]    = builderConfig->getNbOptimizationProfiles();
    dBuilderConfig["nProfileStream"]          = (unsigned long)builderConfig->getProfileStream();
    dBuilderConfig["kProfilingVerbosity"]     = int(builderConfig->getProfilingVerbosity());
    dBuilderConfig["nAverageTimingIteration"] = builderConfig->getAvgTimingIterations();
    dBuilderConfig["nTacticSource"]           = builderConfig->getTacticSources();

    //dBuilderConfig["int8_calibrator"] = builderConfig.getInt8Calibrator; //TODO
    //dBuilderConfig["algorithmSelector"]  = builderConfig.getAlgorithmSelector; //TODO
    //dBuilderConfig["get_device_type"] = builderConfig.getDeviceType(); // TODO
    //dBuilderConfig["bCanRunOnDLA"]    = builderConfig.canRunOnDLA();   //TODO

    json lAllOP;
    if (lOptimizationProfile.size() > 0)
    {
        assert(lOptimizationProfile.size() == builderConfig->getNbOptimizationProfiles());
        for (int i = 0; i < builderConfig->getNbOptimizationProfiles(); ++i)
        {
            const IOptimizationProfile *optimizationProfile = lOptimizationProfile[i];

            json lOP;
            for (int j = 0; j < network->getNbInputs(); ++j)
            {
                ITensor *tensor = network->getInput(j);
                if (tensor->isShapeTensor())
                {
                    int32_t const *minShape = optimizationProfile->getShapeValues(tensor->getName(), OptProfileSelector::kMIN);
                    int32_t const *optShape = optimizationProfile->getShapeValues(tensor->getName(), OptProfileSelector::kOPT);
                    int32_t const *maxShape = optimizationProfile->getShapeValues(tensor->getName(), OptProfileSelector::kMAX);

                    if (optShape == nullptr)
                    {
                        printf("[INFO from NetowrkInspector]: No profile for input tensor %d, continue\n", j);
                        lOP += {}; // place holder for input tensor j
                    }
                    else
                    {
                        json dOP;
                        dOP["name"]         = tensor->getName();
                        dOP["bShapeTensor"] = tensor->isShapeTensor();
                        dOP["nDimension"]   = tensor->getDimensions().nbDims;
                        dOP["min"]          = {};
                        dOP["opt"]          = {};
                        dOP["max"]          = {};
                        for (int k = 0; k < tensor->getDimensions().nbDims; ++k)
                        {
                            dOP["min"].push_back(minShape[k]);
                            dOP["opt"].push_back(optShape[k]);
                            dOP["max"].push_back(maxShape[k]);
                        }
                        lOP += dOP;
                    }
                }
                else
                {
                    Dims minShape = optimizationProfile->getDimensions(tensor->getName(), OptProfileSelector::kMIN);
                    Dims optShape = optimizationProfile->getDimensions(tensor->getName(), OptProfileSelector::kOPT);
                    Dims maxShape = optimizationProfile->getDimensions(tensor->getName(), OptProfileSelector::kMAX);

                    if (optShape.nbDims == -1)
                    {
                        printf("[INFO from NetowrkInspector]: No profile for input tensor %d, continue\n", j);
                        lOP += {}; // place holder for input tensor j
                    }
                    else
                    {
                        json dOP;
                        dOP["name"]         = tensor->getName();
                        dOP["bShapeTensor"] = tensor->isShapeTensor();
                        dOP["nDimension"]   = tensor->getDimensions().nbDims;
                        dOP["min"]          = minShape.d;
                        dOP["opt"]          = optShape.d;
                        dOP["max"]          = maxShape.d;
                        lOP += dOP;
                    }
                }
            }

            lAllOP += lOP;
        }
    }
    dBuilderConfig["lOptimizationProfile"] = lAllOP;

    js["BuilderConfig"] = dBuilderConfig;
    return 0;
}

// Network
int extractNetwork(json &js, const INetworkDefinition *network)
{
    json dNetwork;
    dNetwork["sName"]              = network->getName();
    dNetwork["nLayer"]             = network->getNbLayers();
    dNetwork["nInput"]             = network->getNbInputs();
    dNetwork["nOutput"]            = network->getNbOutputs();
    dNetwork["bImplicitBatchMode"] = network->hasImplicitBatchDimension();
    dNetwork["bExplicitPrecision"] = network->hasExplicitPrecision();

    json lBinding;
    for (int i = 0; i < network->getNbInputs(); ++i)
    {
        ITensor *tensor          = network->getInput(i);
        json     dBinding        = {};
        dBinding["sName"]        = tensor->getName();
        dBinding["bInput"]       = tensor->isNetworkInput();
        dBinding["bShapeTensor"] = tensor->isShapeTensor();
        lBinding.push_back(dBinding);
    }
    for (int i = 0; i < network->getNbOutputs(); ++i)
    {
        ITensor *tensor          = network->getOutput(i);
        json     dBinding        = {};
        dBinding["sName"]        = tensor->getName();
        dBinding["bInput"]       = tensor->isNetworkInput();
        dBinding["bShapeTensor"] = tensor->isShapeTensor();
        lBinding.push_back(dBinding);
    }
    dNetwork["Binding"] = lBinding;

    js["Network"] = dNetwork;
    return 0;
}

// Layer and Tensor
int exactTensor(json &js, const ITensor *tensor)
{
    json dTensor;
    dTensor["nDimension"]            = tensor->getDimensions().nbDims;
    dTensor["lShape"]                = tensor->getDimensions().d;
    dTensor["kLocation"]             = int(tensor->getLocation());
    dTensor["bBroadcastAcrossBatch"] = tensor->getBroadcastAcrossBatch();
    dTensor["kDataType"]             = int(tensor->getType());
    dTensor["nAllowedFormat"]        = int(tensor->getAllowedFormats());
    dTensor["lDynamicRange"]         = {tensor->getDynamicRangeMin(), tensor->getDynamicRangeMax()};
    dTensor["bExecutionTensor"]      = tensor->isExecutionTensor();
    dTensor["bShapeTensor"]          = tensor->isShapeTensor();
    dTensor["bNetworkInput"]         = tensor->isNetworkInput();
    dTensor["bNetworkOutput"]        = tensor->isNetworkOutput();

    js[tensor->getName()] = dTensor;
    return 0;
}

int exactLayerAndTensor(json &js, const std::string &paraFile, const INetworkDefinition *network)
{
    json lLayer;
    json dTensor;
    json dIfCondition;
    json dLoop;
    //parameter

    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        ILayer *layer = network->getLayer(i);
        json    dLayer;
        dLayer["kIndex"]          = i;
        dLayer["sName"]           = layer->getName();
        dLayer["kType"]           = int(layer->getType());
        dLayer["nInput"]          = layer->getNbInputs();
        dLayer["nOutput"]         = layer->getNbOutputs();
        dLayer["kPrecision"]      = int(layer->getPrecision());
        dLayer["bPrecisionIsSet"] = layer->precisionIsSet();

        json lInputTensor;
        for (int j = 0; j < layer->getNbInputs(); ++j)
        {
            ITensor *tensor = layer->getInput(j);
            if (layer->getType() == LayerType::kFILL && j == 0 && tensor == nullptr)
            {
                lInputTensor.push_back({});
            }
            else if (layer->getType() == LayerType::kSLICE && j < layer->getNbInputs() - 1 && tensor == nullptr)
            {
                lInputTensor.push_back({});
            }
            else if (layer->getType() == LayerType::kRNN_V2 && j >= 1 && tensor == nullptr)
            {
                lInputTensor.push_back({});
            }
            else
            {
                lInputTensor.push_back(tensor->getName());
                if (!dTensor.contains(std::string("/") + std::string(tensor->getName())))
                {
                    exactTensor(dTensor, tensor);
                }
            }
            dLayer["lInputTensorName"] = lInputTensor;
        }

        json lOutputTensor;
        json lOutputTensorDataType;
        for (int j = 0; j < layer->getNbOutputs(); ++j)
        {
            ITensor *tensor = layer->getOutput(j);
            lOutputTensor.push_back(tensor->getName());
            lOutputTensorDataType.push_back(layer->getOutputType(j));
            if (!dTensor.contains(std::string("/") + std::string(tensor->getName())))
            {
                exactTensor(dTensor, tensor);
            }
            dLayer["lOutputTensorName"]     = lOutputTensor;
            dLayer["lOutputTensorDataType"] = lOutputTensorDataType;
        }

        // Specialization of each layer
        //  0 LayerType.CONVOLUTION
        //  1 LayerType.FULLY_CONNECTED
        //  2 LayerType.ACTIVATION
        //  3 LayerType.POOLING
        //  4 LayerType.LRN
        //  5 LayerType.SCALE
        //  6 LayerType.SOFTMAX
        //  7 LayerType.DECONVOLUTION
        //  8 LayerType.CONCATENATION
        //  9 LayerType.ELEMENTWISE
        // 10 LayerType.PLUGIN
        // 11 LayerType.UNARY
        // 12 LayerType.PADDING
        // 13 LayerType.SHUFFLE
        // 14 LayerType.REDUCE
        // 15 LayerType.TOPK
        // 16 LayerType.GATHER
        // 17 LayerType.MATRIX_MULTIPLY
        // 18 LayerType.RAGGED_SOFTMAX
        // 19 LayerType.CONSTANT
        // 20 LayerType.RNN_V2
        // 21 LayerType.IDENTITY
        // 22 LayerType.PLUGIN_V2
        // 23 LayerType.SLICE
        // 24 LayerType.SHAPE
        // 25 LayerType.PARAMETRIC_RELU
        // 26 LayerType.RESIZE
        // 27 LayerType.TRIP_LIMIT
        // 28 LayerType.RECURRENCE
        // 29 LayerType.ITERATOR
        // 30 LayerType.LOOP_OUTPUT
        // 31 LayerType.SELECT
        // 32 LayerType.FILL
        // 33 LayerType.QUANTIZE
        // 34 LayerType.DEQUANTIZE
        // 35 LayerType.CONDITION
        // 36 LayerType.CONDITIONAL_INPUT
        // 37 LayerType.CONDITIONAL_OUTPUT
        // 38 LayerType.SCATTER
        // 39 LayerType.EINSUM
        // 40 LayerType.ASSERTION

        switch (layer->getType())
        {
        case LayerType::kCONVOLUTION: // 0
            IConvolutionLayer *ll  = (IConvolutionLayer *)layer;
            dLayer["kernel"]       = ll->getName() + std::string("-kernel");
            dLayer["lKernelShape"] = json::array({ll->getNbOutputMaps(), ll->getInput(0)->getDimensions().d[1], ll->getKernelSizeNd().d[0], ll->getKernelSizeNd().d[1]});
            dLayer["bias"]         = ll->getName() + std::string("-bias");
            dLayer["lBiasShape"]   = json::array({ll->getNbOutputMaps()});
            /*
            dLayer["kernel_size_nd"]       = list(ll.kernel_size_nd);
            dLayer["num_output_maps"]      = ll.num_output_maps;
            dLayer["stride_nd"]            = list(ll.stride_nd);
            dLayer["dilation_nd"]          = list(ll.dilation_nd);
            dLayer["num_groups"]           = ll.num_groups;
            dLayer["padding_nd"]           = list(ll.padding_nd);
            dLayer["padding_mode"]         = int(ll.padding_mode);
            dLayer["pre_padding"]          = list(ll.pre_padding);
            dLayer["post_padding"]         = list(ll.post_padding);
            parameter[ll.name + "-kernel"] = ll.kernel;
            parameter[ll.name + "-bias"]   = ll.bias;
            //dLayer["kernel_size"] = list(ll.kernel_size);  // deprecated since TensorRT 8
            //dLayer["stride"] = list(ll.stride);  // deprecated since TensorRT 8
            //dLayer["dilation"] = list(ll.dilation);  // deprecated since TensorRT 8
            //dLayer["padding"] = ll.padding; // deprecated since TensorRT 8
            */
            break;
        case LayerType::kFULLY_CONNECTED: // 1
            /*
            layer.__class__ = trt.IFullyConnectedLayer
            dLayer["kernel"] = layer.name + "-kernel"
            dLayer["lKernelShape"] = [layer.num_output_channels, layer.kernel.shape[0] // layer.num_output_channels]
            parameter[layer.name + "-kernel"] = layer.kernel
            dLayer["bias"] = layer.name + "-bias"
            dLayer["lBiasShape"] = list(layer.bias.shape)
            parameter[layer.name + "-bias"] = layer.bias
            dLayer["num_output_channels"] = layer.num_output_channels
            */
            break;
        case LayerType::kACTIVATION: // 2
            break;
        case LayerType::kPOOLING: // 3
            break;
        case LayerType::kLRN: // 4
            break;
        case LayerType::kSCALE: // 5
            break;
        case LayerType::kSOFTMAX: // 6
            break;
        case LayerType::kDECONVOLUTION: // 7
            break;
        case LayerType::kCONCATENATION: // 8
            break;
        case LayerType::kELEMENTWISE: // 9
            break;
        case LayerType::kPLUGIN: // 10
            break;
        case LayerType::kUNARY: // 11
            break;
        case LayerType::kPADDING: // 12
            break;
        case LayerType::kSHUFFLE: // 13
            break;
        case LayerType::kREDUCE: // 14
            break;
        case LayerType::kTOPK: // 15
            break;
        case LayerType::kGATHER: // 16
            break;
        case LayerType::kMATRIX_MULTIPLY: // 17
            break;
        case LayerType::kRAGGED_SOFTMAX: // 18
            break;
        case LayerType::kCONSTANT: // 19
            break;
        case LayerType::kRNN_V2: // 20
            break;
        case LayerType::kIDENTITY: // 21
            break;
        case LayerType::kPLUGIN_V2: // 22
            break;
        case LayerType::kSLICE: // 23
            break;
        case LayerType::kSHAPE: // 24
            break;
        case LayerType::kPARAMETRIC_RELU: // 25
            break;
        case LayerType::kRESIZE: // 26
            break;
        case LayerType::kTRIP_LIMIT: // 27
            break;
        case LayerType::kRECURRENCE: // 28
            break;
        case LayerType::kITERATOR: // 29
            break;
        case LayerType::kLOOP_OUTPUT: // 30
            break;
        case LayerType::kSELECT: // 31
            break;
        case LayerType::kFILL: // 32
            break;
        case LayerType::kQUANTIZE: // 33
            break;
        case LayerType::kDEQUANTIZE: // 34
            break;
        case LayerType::kCONDITION: // 35
            break;
        case LayerType::kCONDITIONAL_INPUT: // 36
            break;
        case LayerType::kCONDITIONAL_OUTPUT: // 37
            break;
        case LayerType::kSCATTER: // 38
            break;
        case LayerType::kEINSUM: // 39
            break;
        case LayerType::kASSERTION: // 40
            break;
        default:
            printf("Layer not supported!\n");
        }

        lLayer.push_back(dLayer);

        js["Layer"]       = lLayer;
        js["Tensor"]      = dTensor;
        js["IfCondition"] = dIfCondition;
        js["Loop"]        = dLoop;
        return 0;
    }

    /*
        elif layer.type == trt.LayerType.ACTIVATION:  # 2
            layer.__class__ = trt.IActivationLayer
            dLayer["alpha"] = layer.alpha
            dLayer["beta"] = layer.beta
            dLayer["type"] = int(layer.type);

        elif layer.type == trt.LayerType.POOLING:  # 3
            layer.__class__ = trt.IPoolingLayer
            dLayer["average_count_excludes_padding"] = layer.average_count_excludes_padding
            dLayer["blend_factor"] = layer.blend_factor
            #dLayer["stride"] = list(layer.stride)  # deprecated since TensorRT 8
            dLayer["stride_nd"] = list(layer.stride_nd)  ;
            #dLayer["padding"] = layer.padding  # deprecated since TensorRT 8
            dLayer["padding_nd"] = list(layer.padding_nd)  ;
            dLayer["padding_mode"] = int(layer.padding_mode);
            dLayer["pre_padding"] = list(layer.pre_padding)  ;
            dLayer["post_padding"] = list(layer.post_padding)  ;
            dLayer["type"] = int(layer.type);
            #dLayer["window_size"] = list(layer.window_size)  # deprecated since TensorRT 8
            dLayer["window_size_nd"] = list(layer.window_size_nd)  ;

        elif layer.type == trt.LayerType.LRN:  # 4
            layer.__class__ = trt.ILRNLayer
            dLayer["alpha"] = layer.alpha
            dLayer["beta"] = layer.beta
            dLayer["k"] = layer.k
            dLayer["window_size"] = layer.window_size

        elif layer.type == trt.LayerType.SCALE:  # 5
            layer.__class__ = trt.IScaleLayer
            dLayer["channel_axis"] = layer.channel_axis
            dLayer["mode"] = int(layer.mode);
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
            dLayer["kernel_size_nd"] = list(layer.kernel_size_nd)  ;
            dLayer["num_output_maps"] = layer.num_output_maps
            #dLayer["stride"] = list(layer.stride)  # deprecated since TensorRT 8
            dLayer["stride_nd"] = list(layer.stride_nd)  ;
            #dLayer["dilation"] = list(layer.dilation)  # deprecated since TensorRT 8
            dLayer["dilation_nd"] = list(layer.dilation_nd)  ;
            dLayer["num_groups"] = layer.num_groups
            #dLayer["padding"] = list(layer.padding)  # deprecated since TensorRT 8
            dLayer["padding_nd"] = list(layer.padding_nd)  ;
            dLayer["padding_mode"] = int(layer.padding_mode);
            dLayer["pre_padding"] = list(layer.pre_padding)  ;
            dLayer["post_padding"] = list(layer.post_padding)  ;

        elif layer.type == trt.LayerType.CONCATENATION:  # 8
            layer.__class__ = trt.IConcatenationLayer
            dLayer["axis"] = layer.axis

        elif layer.type == trt.LayerType.ELEMENTWISE:  # 9
            layer.__class__ = trt.IElementWiseLayer
            dLayer["op"] = int(layer.op);

        elif layer.type == trt.LayerType.PLUGIN:  # 10
            print("IPlugin Layer not supported!")  # layer.__class__ = trt.IPluginLayer
            #break

        elif layer.type == trt.LayerType.UNARY:  # 11
            layer.__class__ = trt.IUnaryLayer
            dLayer["op"] = int(layer.op);

        elif layer.type == trt.LayerType.PADDING:  # 12
            layer.__class__ = trt.IPaddingLayer
            #dLayer["pre_padding"] = list(layer.pre_padding)  # deprecated since TensorRT 8
            dLayer["pre_padding_nd"] = list(layer.pre_padding_nd)  ;
            #dLayer["post_padding"] = layer.post_padding  # deprecated since TensorRT 8  # different from Convolution / Deconvolution Layer
            dLayer["post_padding_nd"] = list(layer.post_padding_nd)  ;

        elif layer.type == trt.LayerType.SHUFFLE:  # 13
            layer.__class__ = trt.IShuffleLayer

            if layer.num_inputs == 2:  # dynamic shuffle mode
                dLayer["bDynamicShuffle"] = True
            else:
                dLayer["bDynamicShuffle"] = False
                try:
                    dLayer["reshape_dims"] = list(layer.reshape_dims)  ;
                except ValueError:
                    dLayer["reshape_dims"] = None  # no reshape operation if ValueError raised
            dLayer["first_transpose"] = list(layer.first_transpose)  ;
            dLayer["second_transpose"] = list(layer.second_transpose)  ;
            dLayer["zero_is_placeholder"] = layer.zero_is_placeholder

        elif layer.type == trt.LayerType.REDUCE:  # 14
            layer.__class__ = trt.IReduceLayer
            dLayer["axes"] = layer.axes
            dLayer["op"] = int(layer.op);
            dLayer["keep_dims"] = layer.keep_dims

        elif layer.type == trt.LayerType.TOPK:  # 15
            layer.__class__ = trt.ITopKLayer
            dLayer["axes"] = layer.axes
            dLayer["op"] = int(layer.op);
            dLayer["k"] = layer.k

        elif layer.type == trt.LayerType.GATHER:  # 16
            layer.__class__ = trt.IGatherLayer
            dLayer["axis"] = layer.axis
            dLayer["mode"] = int(layer.mode);

        elif layer.type == trt.LayerType.MATRIX_MULTIPLY:  # 17
            layer.__class__ = trt.IMatrixMultiplyLayer
            dLayer["op0"] = int(layer.op0);
            dLayer["op1"] = int(layer.op1);

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
            dLayer["op"] = int(layer.op);
            dLayer["input_mode"] = int(layer.input_mode);
            dLayer["direction"] = int(layer.direction);
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
            dLayer["mode"] = int(layer.mode);
            try:
                dLayer["start"] = list(layer.start)  ;
            except ValueError:
                dLayer["start"] = None
            try:
                dLayer["shape"] = list(layer.shape)  ;
            except ValueError:
                dLayer["shape"] = None
            try:
                dLayer["stride"] = list(layer.stride)  ;
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
                    dLayer["shape"] = list(layer.shape)  ;
                else:  # static resize mode + use scale mode, TODO: how to check layer.shape? such as "layer.shape == [0] == [0]"
                    dLayer["bShapeMode"] = False
                    dLayer["scales"] = list(layer.scales)  ;
            dLayer["resize_mode"] = int(layer.resize_mode);
            if layer.resize_mode == trt.ResizeMode.LINEAR and layer.coordinate_transformation == trt.ResizeCoordinateTransformation.ASYMMETRIC:
                print("[Warning from NetworkInspector]: ResizeCoordinateTransformation of Resize Layer %s is set as HALF_PIXEL though default behaviour or your explicit set is ASYMMETRIC mode, please refer to the source code of NetworkInspector if you insist to use ASYMMETRIC mode!" % layer.name)
                layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
                #layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC # uncomment this line if you want to use ASYMMETRIC mode
            dLayer["coordinate_transformation"] = int(layer.coordinate_transformation);
            dLayer["selector_for_single_pixel"] = int(layer.selector_for_single_pixel);
            dLayer["nearest_rounding"] = int(layer.nearest_rounding);

        elif layer.type == trt.LayerType.TRIP_LIMIT:  # 27
            layer.__class__ = trt.ITripLimitLayer
            if layer.loop.name not in dLoop.keys():  # search every time because the appearance order of layers in loop is uncertain
                dLoop[layer.loop.name] = {}
                dLoop[layer.loop.name]["RecurrenceLayerName"] = []
                dLoop[layer.loop.name]["LoopOutputLayerName"] = []
                dLoop[layer.loop.name]["IteratorLayerName"] = []
            dLoop[layer.loop.name]["TripLimitLayerName"] = layer.name
            dLayer["kind"] = int(layer.kind);

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
            dLayer["kind"] = int(layer.kind);

        elif layer.type == trt.LayerType.SELECT:  # 31
            layer.__class__ = trt.ISelectLayer

        elif layer.type == trt.LayerType.FILL:  # 32
            layer.__class__ = trt.IFillLayer
            dLayer["operation"] = int(layer.operation);
            if layer.get_input(0) is not None:  # dynamic fill mode, the shape of output tensor depends on input tenor 0
                dLayer["bDynamicShapeFill"] = True
                dLayer["shape"] = None
            else:  # static fill mode, the shape of output tensor is given by input parameter
                dLayer["bDynamicShapeFill"] = False
                dLayer["shape"] = list(layer.shape)  ;

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
            dLayer["mode"] = int(layer.mode);

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
    */
    return 0;
}

int inspectNetwork(const IBuilder *                          builder,
                   const IBuilderConfig *                    builderConfig,
                   const INetworkDefinition *                network,
                   const std::vector<IOptimizationProfile *> lOptimizationProfile = {},
                   const IOptimizationProfile *              calibrationProfile   = {},
                   bool                                      bPrintInformation    = true,
                   const std::string                         jsonFile             = "./model.json",
                   const std::string                         paraFile             = "./model.npz")
{
    if (bPrintInformation)
    {
        printf("\nOriginal network:\n");
        for (int i = 0; i < network->getNbLayers(); ++i)
        {
            ILayer *layer = network->getLayer(i);
            std::cout << std::setw(4) << i << std::string("->") << layerTypeToString(layer->getType()) << std::string(",in=") << layer->getNbInputs() << std::string(",out=") << layer->getNbOutputs() << std::string(",") << std::string(layer->getName()) << std::endl;
            for (int j = 0; j < layer->getNbInputs(); ++j)
            {
                ITensor *tensor = layer->getInput(j);
                std::cout << std::string("\tInput  ") << std::setw(2) << j << std::string(":") << shapeToString(tensor->getDimensions()) << std::string(",") << dataTypeToString(tensor->getType()) << std::string(",") << std ::string(tensor->getName()) << std::endl;
            }
            for (int j = 0; j < layer->getNbOutputs(); ++j)
            {
                ITensor *tensor = layer->getOutput(j);
                std::cout << std::string("\tOutput ") << std::setw(2) << j << std::string(":") << shapeToString(tensor->getDimensions()) << std::string(",") << dataTypeToString(tensor->getType()) << std::string(",") << std ::string(tensor->getName()) << std::endl;
            }
        }
    }

    json js;
    int  state {0};

    // Builder, almost useless especially since TensorRT 8
    state |= extractBuilder(js, builder, builderConfig, network);

    // BuilderConfig
    state |= extractBuilderConfig(js, builderConfig, network, lOptimizationProfile);

    // Network
    state |= extractNetwork(js, network);

    // Layer and Tensor
    state |= exactLayerAndTensor(js, paraFile, network);

    // Save result as file
    std::fstream fout;
    fout.open(jsonFile, std::ofstream::out);
    if (!fout)
    {
        std::cout << "Failed opening" << jsonFile << " to write!" << std::endl;
        return 1;
    }
    fout << js;
    fout.close();

    return state;
}
