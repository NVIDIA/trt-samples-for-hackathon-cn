from __future__ import annotations
import ctypes as ctypes
import os as os
import sys as sys
from tensorrt.tensorrt import APILanguage
from tensorrt.tensorrt import ActivationType
from tensorrt.tensorrt import AllocatorFlag
from tensorrt.tensorrt import AttentionNormalizationOp
from tensorrt.tensorrt import BoundingBoxFormat
from tensorrt.tensorrt import Builder
from tensorrt.tensorrt import BuilderFlag
from tensorrt.tensorrt import CalibrationAlgoType
from tensorrt.tensorrt import CollectiveOperation
from tensorrt.tensorrt import CumulativeOperation
from tensorrt.tensorrt import DataType
from tensorrt.tensorrt import DeviceType
from tensorrt.tensorrt import DimensionOperation
from tensorrt.tensorrt import Dims
from tensorrt.tensorrt import Dims2
from tensorrt.tensorrt import Dims3
from tensorrt.tensorrt import Dims4
from tensorrt.tensorrt import DimsExprs
from tensorrt.tensorrt import DimsHW
from tensorrt.tensorrt import DynamicPluginTensorDesc
from tensorrt.tensorrt import ElementWiseOperation
from tensorrt.tensorrt import EngineCapability
from tensorrt.tensorrt import EngineInspector
from tensorrt.tensorrt import EngineStat
from tensorrt.tensorrt import ErrorCode
from tensorrt.tensorrt import ErrorCodeTRT
from tensorrt.tensorrt import ExecutionContextAllocationStrategy
from tensorrt.tensorrt import FallbackString
from tensorrt.tensorrt import FillOperation
from tensorrt.tensorrt import GatherMode
from tensorrt.tensorrt import HardwareCompatibilityLevel
from tensorrt.tensorrt import IActivationLayer
from tensorrt.tensorrt import IAlgorithm
from tensorrt.tensorrt import IAlgorithmContext
from tensorrt.tensorrt import IAlgorithmIOInfo
from tensorrt.tensorrt import IAlgorithmSelector
from tensorrt.tensorrt import IAlgorithmVariant
from tensorrt.tensorrt import IAssertionLayer
from tensorrt.tensorrt import IAttention
from tensorrt.tensorrt import IAttentionBoundaryLayer
from tensorrt.tensorrt import IAttentionInputLayer
from tensorrt.tensorrt import IAttentionOutputLayer
from tensorrt.tensorrt import IBuilderConfig
from tensorrt.tensorrt import ICastLayer
from tensorrt.tensorrt import IConcatenationLayer
from tensorrt.tensorrt import IConditionLayer
from tensorrt.tensorrt import IConstantLayer
from tensorrt.tensorrt import IConvolutionLayer
from tensorrt.tensorrt import ICudaEngine
from tensorrt.tensorrt import ICumulativeLayer
from tensorrt.tensorrt import IDebugListener
from tensorrt.tensorrt import IDeconvolutionLayer
from tensorrt.tensorrt import IDequantizeLayer
from tensorrt.tensorrt import IDimensionExpr
from tensorrt.tensorrt import IDistCollectiveLayer
from tensorrt.tensorrt import IDynamicQuantizeLayer
from tensorrt.tensorrt import IEinsumLayer
from tensorrt.tensorrt import IElementWiseLayer
from tensorrt.tensorrt import IErrorRecorder
from tensorrt.tensorrt import IExecutionContext
from tensorrt.tensorrt import IExprBuilder
from tensorrt.tensorrt import IFillLayer
from tensorrt.tensorrt import IGatherLayer
from tensorrt.tensorrt import IGpuAllocator
from tensorrt.tensorrt import IGpuAsyncAllocator
from tensorrt.tensorrt import IGridSampleLayer
from tensorrt.tensorrt import IHostMemory
from tensorrt.tensorrt import IIdentityLayer
from tensorrt.tensorrt import IIfConditional
from tensorrt.tensorrt import IIfConditionalBoundaryLayer
from tensorrt.tensorrt import IIfConditionalInputLayer
from tensorrt.tensorrt import IIfConditionalOutputLayer
from tensorrt.tensorrt import IInt8Calibrator
from tensorrt.tensorrt import IInt8EntropyCalibrator
from tensorrt.tensorrt import IInt8EntropyCalibrator2
from tensorrt.tensorrt import IInt8LegacyCalibrator
from tensorrt.tensorrt import IInt8MinMaxCalibrator
from tensorrt.tensorrt import IIteratorLayer
from tensorrt.tensorrt import IKVCacheUpdateLayer
from tensorrt.tensorrt import ILRNLayer
from tensorrt.tensorrt import ILayer
from tensorrt.tensorrt import ILogger
from tensorrt.tensorrt import ILoop
from tensorrt.tensorrt import ILoopBoundaryLayer
from tensorrt.tensorrt import ILoopOutputLayer
from tensorrt.tensorrt import IMatrixMultiplyLayer
from tensorrt.tensorrt import IMoELayer
from tensorrt.tensorrt import INMSLayer
from tensorrt.tensorrt import INetworkDefinition
from tensorrt.tensorrt import INonZeroLayer
from tensorrt.tensorrt import INormalizationLayer
from tensorrt.tensorrt import IOneHotLayer
from tensorrt.tensorrt import IOptimizationProfile
from tensorrt.tensorrt import IOutputAllocator
from tensorrt.tensorrt import IPaddingLayer
from tensorrt.tensorrt import IParametricReLULayer
from tensorrt.tensorrt import IPluginCapability
from tensorrt.tensorrt import IPluginCreator
from tensorrt.tensorrt import IPluginCreatorInterface
from tensorrt.tensorrt import IPluginCreatorV3One
from tensorrt.tensorrt import IPluginCreatorV3Quick
from tensorrt.tensorrt import IPluginRegistry
from tensorrt.tensorrt import IPluginResource
from tensorrt.tensorrt import IPluginResourceContext
from tensorrt.tensorrt import IPluginV2
from tensorrt.tensorrt import IPluginV2DynamicExt
from tensorrt.tensorrt import IPluginV2DynamicExtBase
from tensorrt.tensorrt import IPluginV2Ext
from tensorrt.tensorrt import IPluginV2Layer
from tensorrt.tensorrt import IPluginV3
from tensorrt.tensorrt import IPluginV3Layer
from tensorrt.tensorrt import IPluginV3OneBuild
from tensorrt.tensorrt import IPluginV3OneBuildV2
from tensorrt.tensorrt import IPluginV3OneCore
from tensorrt.tensorrt import IPluginV3OneRuntime
from tensorrt.tensorrt import IPluginV3QuickAOTBuild
from tensorrt.tensorrt import IPluginV3QuickBuild
from tensorrt.tensorrt import IPluginV3QuickCore
from tensorrt.tensorrt import IPluginV3QuickRuntime
from tensorrt.tensorrt import IPoolingLayer
from tensorrt.tensorrt import IProfiler
from tensorrt.tensorrt import IProgressMonitor
from tensorrt.tensorrt import IQuantizeLayer
from tensorrt.tensorrt import IRaggedSoftMaxLayer
from tensorrt.tensorrt import IRecurrenceLayer
from tensorrt.tensorrt import IReduceLayer
from tensorrt.tensorrt import IResizeLayer
from tensorrt.tensorrt import IReverseSequenceLayer
from tensorrt.tensorrt import IRotaryEmbeddingLayer
from tensorrt.tensorrt import IRuntimeConfig
from tensorrt.tensorrt import IScaleLayer
from tensorrt.tensorrt import IScatterLayer
from tensorrt.tensorrt import ISelectLayer
from tensorrt.tensorrt import ISerializationConfig
from tensorrt.tensorrt import IShapeLayer
from tensorrt.tensorrt import IShuffleLayer
from tensorrt.tensorrt import ISliceLayer
from tensorrt.tensorrt import ISoftMaxLayer
from tensorrt.tensorrt import ISqueezeLayer
from tensorrt.tensorrt import IStreamReader
from tensorrt.tensorrt import IStreamReaderV2
from tensorrt.tensorrt import IStreamWriter
from tensorrt.tensorrt import ISymExpr
from tensorrt.tensorrt import ISymExprs
from tensorrt.tensorrt import ITensor
from tensorrt.tensorrt import ITimingCache
from tensorrt.tensorrt import ITopKLayer
from tensorrt.tensorrt import ITripLimitLayer
from tensorrt.tensorrt import IUnaryLayer
from tensorrt.tensorrt import IUnsqueezeLayer
from tensorrt.tensorrt import IVersionedInterface
from tensorrt.tensorrt import InterfaceInfo
from tensorrt.tensorrt import InterpolationMode
from tensorrt.tensorrt import KVCacheMode
from tensorrt.tensorrt import KernelLaunchParams
from tensorrt.tensorrt import LayerInformationFormat
from tensorrt.tensorrt import LayerType
from tensorrt.tensorrt import Logger
from tensorrt.tensorrt import LoopOutput
from tensorrt.tensorrt import MatrixOperation
from tensorrt.tensorrt import MemoryPoolType
from tensorrt.tensorrt import MoEActType
from tensorrt.tensorrt import NetworkDefinitionCreationFlag
from tensorrt.tensorrt import NodeIndices
from tensorrt.tensorrt import OnnxParser
from tensorrt.tensorrt import OnnxParserFlag
from tensorrt.tensorrt import OnnxParserRefitter
from tensorrt.tensorrt import PaddingMode
from tensorrt.tensorrt import ParserError
from tensorrt.tensorrt import Permutation
from tensorrt.tensorrt import PluginArgDataType
from tensorrt.tensorrt import PluginArgType
from tensorrt.tensorrt import PluginCapabilityType
from tensorrt.tensorrt import PluginCreatorVersion
from tensorrt.tensorrt import PluginField
from tensorrt.tensorrt import PluginFieldCollection
from tensorrt.tensorrt import PluginFieldCollection_
from tensorrt.tensorrt import PluginFieldType
from tensorrt.tensorrt import PluginTensorDesc
from tensorrt.tensorrt import PoolingType
from tensorrt.tensorrt import PreviewFeature
from tensorrt.tensorrt import Profiler
from tensorrt.tensorrt import ProfilingVerbosity
from tensorrt.tensorrt import QuantizationFlag
from tensorrt.tensorrt import QuickPluginCreationRequest
from tensorrt.tensorrt import ReduceOperation
from tensorrt.tensorrt import Refitter
from tensorrt.tensorrt import ResizeCoordinateTransformation
from tensorrt.tensorrt import ResizeRoundMode
from tensorrt.tensorrt import ResizeSelector
from tensorrt.tensorrt import Runtime
from tensorrt.tensorrt import RuntimePlatform
from tensorrt.tensorrt import SampleMode
from tensorrt.tensorrt import ScaleMode
from tensorrt.tensorrt import ScatterMode
from tensorrt.tensorrt import SeekPosition
from tensorrt.tensorrt import SerializationFlag
from tensorrt.tensorrt import SubGraphCollection
from tensorrt.tensorrt import TacticSource
from tensorrt.tensorrt import TempfileControlFlag
from tensorrt.tensorrt import TensorFormat
from tensorrt.tensorrt import TensorIOMode
from tensorrt.tensorrt import TensorLocation
from tensorrt.tensorrt import TensorRTPhase
from tensorrt.tensorrt import TilingOptimizationLevel
from tensorrt.tensorrt import TimingCacheKey
from tensorrt.tensorrt import TimingCacheValue
from tensorrt.tensorrt import TopKOperation
from tensorrt.tensorrt import TripLimit
from tensorrt.tensorrt import UnaryOperation
from tensorrt.tensorrt import Weights
from tensorrt.tensorrt import WeightsRole
from tensorrt.tensorrt import get_builder_plugin_registry
from tensorrt.tensorrt import get_nv_onnx_parser_version
from tensorrt.tensorrt import get_plugin_registry
from tensorrt.tensorrt import init_libnvinfer_plugins
import warnings as warnings
from . import tensorrt
__all__: list[str] = ['APILanguage', 'ActivationType', 'AllocatorFlag', 'AttentionNormalizationOp', 'BoundingBoxFormat', 'Builder', 'BuilderFlag', 'CalibrationAlgoType', 'CollectiveOperation', 'CumulativeOperation', 'DataType', 'DeviceType', 'DimensionOperation', 'Dims', 'Dims2', 'Dims3', 'Dims4', 'DimsExprs', 'DimsHW', 'DynamicPluginTensorDesc', 'ElementWiseOperation', 'EngineCapability', 'EngineInspector', 'EngineStat', 'ErrorCode', 'ErrorCodeTRT', 'ExecutionContextAllocationStrategy', 'FallbackString', 'FillOperation', 'GatherMode', 'HardwareCompatibilityLevel', 'IActivationLayer', 'IAlgorithm', 'IAlgorithmContext', 'IAlgorithmIOInfo', 'IAlgorithmSelector', 'IAlgorithmVariant', 'IAssertionLayer', 'IAttention', 'IAttentionBoundaryLayer', 'IAttentionInputLayer', 'IAttentionOutputLayer', 'IBuilderConfig', 'ICastLayer', 'IConcatenationLayer', 'IConditionLayer', 'IConstantLayer', 'IConvolutionLayer', 'ICudaEngine', 'ICumulativeLayer', 'IDebugListener', 'IDeconvolutionLayer', 'IDequantizeLayer', 'IDimensionExpr', 'IDistCollectiveLayer', 'IDynamicQuantizeLayer', 'IEinsumLayer', 'IElementWiseLayer', 'IErrorRecorder', 'IExecutionContext', 'IExprBuilder', 'IFillLayer', 'IGatherLayer', 'IGpuAllocator', 'IGpuAsyncAllocator', 'IGridSampleLayer', 'IHostMemory', 'IIdentityLayer', 'IIfConditional', 'IIfConditionalBoundaryLayer', 'IIfConditionalInputLayer', 'IIfConditionalOutputLayer', 'IInt8Calibrator', 'IInt8EntropyCalibrator', 'IInt8EntropyCalibrator2', 'IInt8LegacyCalibrator', 'IInt8MinMaxCalibrator', 'IIteratorLayer', 'IKVCacheUpdateLayer', 'ILRNLayer', 'ILayer', 'ILogger', 'ILoop', 'ILoopBoundaryLayer', 'ILoopOutputLayer', 'IMatrixMultiplyLayer', 'IMoELayer', 'INMSLayer', 'INetworkDefinition', 'INonZeroLayer', 'INormalizationLayer', 'IOneHotLayer', 'IOptimizationProfile', 'IOutputAllocator', 'IPaddingLayer', 'IParametricReLULayer', 'IPluginCapability', 'IPluginCreator', 'IPluginCreatorInterface', 'IPluginCreatorV3One', 'IPluginCreatorV3Quick', 'IPluginRegistry', 'IPluginResource', 'IPluginResourceContext', 'IPluginV2', 'IPluginV2DynamicExt', 'IPluginV2DynamicExtBase', 'IPluginV2Ext', 'IPluginV2Layer', 'IPluginV3', 'IPluginV3Layer', 'IPluginV3OneBuild', 'IPluginV3OneBuildV2', 'IPluginV3OneCore', 'IPluginV3OneRuntime', 'IPluginV3QuickAOTBuild', 'IPluginV3QuickBuild', 'IPluginV3QuickCore', 'IPluginV3QuickRuntime', 'IPoolingLayer', 'IProfiler', 'IProgressMonitor', 'IQuantizeLayer', 'IRaggedSoftMaxLayer', 'IRecurrenceLayer', 'IReduceLayer', 'IResizeLayer', 'IReverseSequenceLayer', 'IRotaryEmbeddingLayer', 'IRuntimeConfig', 'IScaleLayer', 'IScatterLayer', 'ISelectLayer', 'ISerializationConfig', 'IShapeLayer', 'IShuffleLayer', 'ISliceLayer', 'ISoftMaxLayer', 'ISqueezeLayer', 'IStreamReader', 'IStreamReaderV2', 'IStreamWriter', 'ISymExpr', 'ISymExprs', 'ITensor', 'ITimingCache', 'ITopKLayer', 'ITripLimitLayer', 'IUnaryLayer', 'IUnsqueezeLayer', 'IVersionedInterface', 'InterfaceInfo', 'InterpolationMode', 'KVCacheMode', 'KernelLaunchParams', 'LayerInformationFormat', 'LayerType', 'Logger', 'LoopOutput', 'MatrixOperation', 'MemoryPoolType', 'MoEActType', 'NetworkDefinitionCreationFlag', 'NodeIndices', 'OnnxParser', 'OnnxParserFlag', 'OnnxParserRefitter', 'PaddingMode', 'ParserError', 'Permutation', 'PluginArgDataType', 'PluginArgType', 'PluginCapabilityType', 'PluginCreatorVersion', 'PluginField', 'PluginFieldCollection', 'PluginFieldCollection_', 'PluginFieldType', 'PluginTensorDesc', 'PoolingType', 'PreviewFeature', 'Profiler', 'ProfilingVerbosity', 'QuantizationFlag', 'QuickPluginCreationRequest', 'ReduceOperation', 'Refitter', 'ResizeCoordinateTransformation', 'ResizeRoundMode', 'ResizeSelector', 'Runtime', 'RuntimePlatform', 'SampleMode', 'ScaleMode', 'ScatterMode', 'SeekPosition', 'SerializationFlag', 'SubGraphCollection', 'TacticSource', 'TempfileControlFlag', 'TensorFormat', 'TensorIOMode', 'TensorLocation', 'TensorRTPhase', 'TilingOptimizationLevel', 'TimingCacheKey', 'TimingCacheValue', 'TopKOperation', 'TripLimit', 'UnaryOperation', 'Weights', 'WeightsRole', 'attr', 'bfloat16', 'bool', 'common_enter', 'common_exit', 'ctypes', 'e8m0', 'float16', 'float32', 'fp4', 'fp8', 'get_builder_plugin_registry', 'get_nv_onnx_parser_version', 'get_plugin_registry', 'init_libnvinfer_plugins', 'int32', 'int4', 'int64', 'int8', 'nptype', 'os', 'sys', 'tensorrt', 'uint8', 'value', 'volume', 'warnings']
def _itemsize(trt_type):
    """

        Returns the size in bytes of this :class:`DataType`.
        The returned size is a rational number, possibly a `Real` denoting a fraction of a byte.

        :arg trt_type: The TensorRT data type.

        :returns: The size of the type.

    """
def common_enter(this):
    ...
def common_exit(this, exc_type, exc_value, traceback):
    """

        Context managers are deprecated and have no effect. Objects are automatically freed when
        the reference count reaches 0.

    """
def nptype(trt_type):
    """

        Returns the numpy-equivalent of a TensorRT :class:`DataType` .

        :arg trt_type: The TensorRT data type to convert.

        :returns: The equivalent numpy type.

    """
def volume(iterable):
    """

        Computes the volume of an iterable.

        :arg iterable: Any python iterable, including a :class:`Dims` object.

        :returns: The volume of the iterable. This will return 1 for empty iterables, as a scalar has an empty shape and the volume of a tensor with empty shape is 1.

    """
__version__: str = '10.16.0.72'
attr: str = 'VERBOSE'
bfloat16: DataType  # value = <DataType.BF16: 7>
bool: DataType  # value = <DataType.BOOL: 4>
e8m0: DataType  # value = <DataType.E8M0: 11>
float16: DataType  # value = <DataType.HALF: 1>
float32: DataType  # value = <DataType.FLOAT: 0>
fp4: DataType  # value = <DataType.FP4: 10>
fp8: DataType  # value = <DataType.FP8: 6>
int32: DataType  # value = <DataType.INT32: 3>
int4: DataType  # value = <DataType.INT4: 9>
int64: DataType  # value = <DataType.INT64: 8>
int8: DataType  # value = <DataType.INT8: 2>
uint8: DataType  # value = <DataType.UINT8: 5>
value: ILogger.Severity  # value = <Severity.VERBOSE: 4>
