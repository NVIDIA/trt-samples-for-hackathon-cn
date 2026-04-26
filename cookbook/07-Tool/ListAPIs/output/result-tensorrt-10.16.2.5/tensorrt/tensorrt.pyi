from __future__ import annotations

import collections.abc
import typing

import numpy
import typing_extensions

__all__: list[str] = ['APILanguage', 'ActivationType', 'AllocatorFlag', 'AttentionNormalizationOp', 'BoundingBoxFormat', 'Builder', 'BuilderFlag', 'CalibrationAlgoType', 'CollectiveOperation', 'CumulativeOperation', 'DataType', 'DeviceType', 'DimensionOperation', 'Dims', 'Dims2', 'Dims3', 'Dims4', 'DimsExprs', 'DimsHW', 'DynamicPluginTensorDesc', 'ElementWiseOperation', 'EngineCapability', 'EngineInspector', 'EngineStat', 'ErrorCode', 'ErrorCodeTRT', 'ExecutionContextAllocationStrategy', 'FallbackString', 'FillOperation', 'GatherMode', 'HardwareCompatibilityLevel', 'IActivationLayer', 'IAlgorithm', 'IAlgorithmContext', 'IAlgorithmIOInfo', 'IAlgorithmSelector', 'IAlgorithmVariant', 'IAssertionLayer', 'IAttention', 'IAttentionBoundaryLayer', 'IAttentionInputLayer', 'IAttentionOutputLayer', 'IBuilderConfig', 'ICastLayer', 'IConcatenationLayer', 'IConditionLayer', 'IConstantLayer', 'IConvolutionLayer', 'ICudaEngine', 'ICumulativeLayer', 'IDebugListener', 'IDeconvolutionLayer', 'IDequantizeLayer', 'IDimensionExpr', 'IDistCollectiveLayer', 'IDynamicQuantizeLayer', 'IEinsumLayer', 'IElementWiseLayer', 'IErrorRecorder', 'IExecutionContext', 'IExprBuilder', 'IFillLayer', 'IGatherLayer', 'IGpuAllocator', 'IGpuAsyncAllocator', 'IGridSampleLayer', 'IHostMemory', 'IIdentityLayer', 'IIfConditional', 'IIfConditionalBoundaryLayer', 'IIfConditionalInputLayer', 'IIfConditionalOutputLayer', 'IInt8Calibrator', 'IInt8EntropyCalibrator', 'IInt8EntropyCalibrator2', 'IInt8LegacyCalibrator', 'IInt8MinMaxCalibrator', 'IIteratorLayer', 'IKVCacheUpdateLayer', 'ILRNLayer', 'ILayer', 'ILogger', 'ILoop', 'ILoopBoundaryLayer', 'ILoopOutputLayer', 'IMatrixMultiplyLayer', 'IMoELayer', 'INMSLayer', 'INetworkDefinition', 'INonZeroLayer', 'INormalizationLayer', 'IOneHotLayer', 'IOptimizationProfile', 'IOutputAllocator', 'IPaddingLayer', 'IParametricReLULayer', 'IPluginCapability', 'IPluginCreator', 'IPluginCreatorInterface', 'IPluginCreatorV3One', 'IPluginCreatorV3Quick', 'IPluginRegistry', 'IPluginResource', 'IPluginResourceContext', 'IPluginV2', 'IPluginV2DynamicExt', 'IPluginV2DynamicExtBase', 'IPluginV2Ext', 'IPluginV2Layer', 'IPluginV3', 'IPluginV3Layer', 'IPluginV3OneBuild', 'IPluginV3OneBuildV2', 'IPluginV3OneCore', 'IPluginV3OneRuntime', 'IPluginV3QuickAOTBuild', 'IPluginV3QuickBuild', 'IPluginV3QuickCore', 'IPluginV3QuickRuntime', 'IPoolingLayer', 'IProfiler', 'IProgressMonitor', 'IQuantizeLayer', 'IRaggedSoftMaxLayer', 'IRecurrenceLayer', 'IReduceLayer', 'IResizeLayer', 'IReverseSequenceLayer', 'IRotaryEmbeddingLayer', 'IRuntimeConfig', 'IScaleLayer', 'IScatterLayer', 'ISelectLayer', 'ISerializationConfig', 'IShapeLayer', 'IShuffleLayer', 'ISliceLayer', 'ISoftMaxLayer', 'ISqueezeLayer', 'IStreamReader', 'IStreamReaderV2', 'IStreamWriter', 'ISymExpr', 'ISymExprs', 'ITensor', 'ITimingCache', 'ITopKLayer', 'ITripLimitLayer', 'IUnaryLayer', 'IUnsqueezeLayer', 'IVersionedInterface', 'InterfaceInfo', 'InterpolationMode', 'KVCacheMode', 'KernelLaunchParams', 'LayerInformationFormat', 'LayerType', 'Logger', 'LoopOutput', 'MatrixOperation', 'MemoryPoolType', 'MoEActType', 'NetworkDefinitionCreationFlag', 'NodeIndices', 'OnnxParser', 'OnnxParserFlag', 'OnnxParserRefitter', 'PaddingMode', 'ParserError', 'Permutation', 'PluginArgDataType', 'PluginArgType', 'PluginCapabilityType', 'PluginCreatorVersion', 'PluginField', 'PluginFieldCollection', 'PluginFieldCollection_', 'PluginFieldType', 'PluginTensorDesc', 'PoolingType', 'PreviewFeature', 'Profiler', 'ProfilingVerbosity', 'QuantizationFlag', 'QuickPluginCreationRequest', 'ReduceOperation', 'Refitter', 'ResizeCoordinateTransformation', 'ResizeRoundMode', 'ResizeSelector', 'Runtime', 'RuntimePlatform', 'SampleMode', 'ScaleMode', 'ScatterMode', 'SeekPosition', 'SerializationFlag', 'SubGraphCollection', 'TacticSource', 'TempfileControlFlag', 'TensorFormat', 'TensorIOMode', 'TensorLocation', 'TensorRTPhase', 'TilingOptimizationLevel', 'TimingCacheKey', 'TimingCacheValue', 'TopKOperation', 'TripLimit', 'UnaryOperation', 'Weights', 'WeightsRole', 'bfloat16', 'bool', 'e8m0', 'float16', 'float32', 'fp4', 'fp8', 'get_builder_plugin_registry', 'get_nv_onnx_parser_version', 'get_plugin_registry', 'init_libnvinfer_plugins', 'int32', 'int4', 'int64', 'int8', 'uint8']
class APILanguage:
    """

        The language used in the implementation of a TensorRT interface.


    Members:

      CPP

      PYTHON
    """
    CPP: typing.ClassVar[APILanguage]  # value = <APILanguage.CPP: 0>
    PYTHON: typing.ClassVar[APILanguage]  # value = <APILanguage.PYTHON: 1>
    __members__: typing.ClassVar[dict[str, APILanguage]]  # value = {'CPP': <APILanguage.CPP: 0>, 'PYTHON': <APILanguage.PYTHON: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ActivationType:
    """
    The type of activation to perform.

    Members:

      RELU : Rectified Linear activation

      SIGMOID : Sigmoid activation

      TANH : Hyperbolic Tangent activation

      LEAKY_RELU : Leaky Relu activation: f(x) = x if x >= 0, f(x) = alpha * x if x < 0

      ELU : Elu activation: f(x) = x if x >= 0, f(x) = alpha * (exp(x) - 1) if x < 0

      SELU : Selu activation: f(x) = beta * x if x > 0, f(x) = beta * (alpha * exp(x) - alpha) if x <= 0

      SOFTSIGN : Softsign activation: f(x) = x / (1 + abs(x))

      SOFTPLUS : Softplus activation: f(x) = alpha * log(exp(beta * x) + 1)

      CLIP : Clip activation: f(x) = max(alpha, min(beta, x))

      HARD_SIGMOID : Hard sigmoid activation: f(x) = max(0, min(1, alpha * x + beta))

      SCALED_TANH : Scaled Tanh activation: f(x) = alpha * tanh(beta * x)

      THRESHOLDED_RELU : Thresholded Relu activation: f(x) = x if x > alpha, f(x) = 0 if x <= alpha

      GELU_ERF : GELU erf activation: 0.5 * x * (1 + erf(sqrt(0.5) * x))

      GELU_TANH : GELU tanh activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (0.044715F * pow(x, 3) + x)))
    """
    CLIP: typing.ClassVar[ActivationType]  # value = <ActivationType.CLIP: 8>
    ELU: typing.ClassVar[ActivationType]  # value = <ActivationType.ELU: 4>
    GELU_ERF: typing.ClassVar[ActivationType]  # value = <ActivationType.GELU_ERF: 12>
    GELU_TANH: typing.ClassVar[ActivationType]  # value = <ActivationType.GELU_TANH: 13>
    HARD_SIGMOID: typing.ClassVar[ActivationType]  # value = <ActivationType.HARD_SIGMOID: 9>
    LEAKY_RELU: typing.ClassVar[ActivationType]  # value = <ActivationType.LEAKY_RELU: 3>
    RELU: typing.ClassVar[ActivationType]  # value = <ActivationType.RELU: 0>
    SCALED_TANH: typing.ClassVar[ActivationType]  # value = <ActivationType.SCALED_TANH: 10>
    SELU: typing.ClassVar[ActivationType]  # value = <ActivationType.SELU: 5>
    SIGMOID: typing.ClassVar[ActivationType]  # value = <ActivationType.SIGMOID: 1>
    SOFTPLUS: typing.ClassVar[ActivationType]  # value = <ActivationType.SOFTPLUS: 7>
    SOFTSIGN: typing.ClassVar[ActivationType]  # value = <ActivationType.SOFTSIGN: 6>
    TANH: typing.ClassVar[ActivationType]  # value = <ActivationType.TANH: 2>
    THRESHOLDED_RELU: typing.ClassVar[ActivationType]  # value = <ActivationType.THRESHOLDED_RELU: 11>
    __members__: typing.ClassVar[dict[str, ActivationType]]  # value = {'RELU': <ActivationType.RELU: 0>, 'SIGMOID': <ActivationType.SIGMOID: 1>, 'TANH': <ActivationType.TANH: 2>, 'LEAKY_RELU': <ActivationType.LEAKY_RELU: 3>, 'ELU': <ActivationType.ELU: 4>, 'SELU': <ActivationType.SELU: 5>, 'SOFTSIGN': <ActivationType.SOFTSIGN: 6>, 'SOFTPLUS': <ActivationType.SOFTPLUS: 7>, 'CLIP': <ActivationType.CLIP: 8>, 'HARD_SIGMOID': <ActivationType.HARD_SIGMOID: 9>, 'SCALED_TANH': <ActivationType.SCALED_TANH: 10>, 'THRESHOLDED_RELU': <ActivationType.THRESHOLDED_RELU: 11>, 'GELU_ERF': <ActivationType.GELU_ERF: 12>, 'GELU_TANH': <ActivationType.GELU_TANH: 13>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class AllocatorFlag:
    """


    Members:

      RESIZABLE : TensorRT may call realloc() on this allocation
    """
    RESIZABLE: typing.ClassVar[AllocatorFlag]  # value = <AllocatorFlag.RESIZABLE: 0>
    __members__: typing.ClassVar[dict[str, AllocatorFlag]]  # value = {'RESIZABLE': <AllocatorFlag.RESIZABLE: 0>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class AttentionNormalizationOp:
    """
    The normalization operations that may be performed by an Attention layer

    Members:

      NONE :

      SOFTMAX :
    """
    NONE: typing.ClassVar[AttentionNormalizationOp]  # value = <AttentionNormalizationOp.NONE: 0>
    SOFTMAX: typing.ClassVar[AttentionNormalizationOp]  # value = <AttentionNormalizationOp.SOFTMAX: 1>
    __members__: typing.ClassVar[dict[str, AttentionNormalizationOp]]  # value = {'NONE': <AttentionNormalizationOp.NONE: 0>, 'SOFTMAX': <AttentionNormalizationOp.SOFTMAX: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class BoundingBoxFormat:
    """

        Enumerates bounding box data formats used for the Boxes input tensor in the NMS layer.


    Members:

      CORNER_PAIRS : (x1, y1, x2, y2) where (x1, y1) and (x2, y2) are any pair of diagonal corners

      CENTER_SIZES : (x_center, y_center, width, height) where (x_center, y_center) is the center point of the box
    """
    CENTER_SIZES: typing.ClassVar[BoundingBoxFormat]  # value = <BoundingBoxFormat.CENTER_SIZES: 1>
    CORNER_PAIRS: typing.ClassVar[BoundingBoxFormat]  # value = <BoundingBoxFormat.CORNER_PAIRS: 0>
    __members__: typing.ClassVar[dict[str, BoundingBoxFormat]]  # value = {'CORNER_PAIRS': <BoundingBoxFormat.CORNER_PAIRS: 0>, 'CENTER_SIZES': <BoundingBoxFormat.CENTER_SIZES: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Builder:
    """

        Builds an :class:`ICudaEngine` from a :class:`INetworkDefinition` .

        :ivar platform_has_tf32: :class:`bool` Whether the platform has tf32 support.
        :ivar platform_has_fast_fp16: :class:`bool` Whether the platform has fast native fp16.
        :ivar platform_has_fast_int8: :class:`bool` Whether the platform has fast native int8.
        :ivar max_DLA_batch_size: :class:`int` The maximum batch size DLA can support. For any tensor the total volume of index dimensions combined(dimensions other than CHW) with the requested batch size should not exceed the value returned by this function.
        :ivar num_DLA_cores: :class:`int` The number of DLA engines available to this builder.
        :ivar error_recorder: :class:`IErrorRecorder` Application-implemented error reporting interface for TensorRT objects.
        :ivar gpu_allocator: :class:`IGpuAllocator` The GPU allocator to be used by the :class:`Builder` . All GPU
            memory acquired will use this allocator. If set to ``None``, the default allocator will be used.
        :ivar logger: :class:`ILogger` The logger provided when creating the refitter.
        :ivar max_threads: :class:`int` The maximum thread that can be used by the :class:`Builder`.
    """
    error_recorder: IErrorRecorder
    @staticmethod
    def __enter__(this):
        ...
    @staticmethod
    def __exit__(this, exc_type, exc_value, traceback):
        """

            Context managers are deprecated and have no effect. Objects are automatically freed when
            the reference count reaches 0.

        """
    def __del__(self) -> None:
        ...
    def __init__(self, logger: ILogger) -> None:
        """
            :arg logger: The logger to use.
        """
    def build_engine_with_config(self, network: INetworkDefinition, config: IBuilderConfig) -> ICudaEngine:
        """
            Builds a network for the given :class:`INetworkDefinition` and :class:`IBuilderConfig` .

            This function allows building a network and creating an engine.

            :arg network: Network definition.
            :arg config: Builder configuration.

            :returns: A pointer to a :class:`ICudaEngine` object that contains a built engine.
        """
    @typing.overload
    def build_serialized_network(self, network: INetworkDefinition, config: IBuilderConfig) -> IHostMemory:
        """
            Builds and serializes a network for the given :class:`INetworkDefinition` and :class:`IBuilderConfig` .

            This function allows building and serialization of a network without creating an engine.

            :arg network: Network definition.
            :arg config: Builder configuration.

            :returns: A pointer to a :class:`IHostMemory` object that contains a serialized network.
        """
    @typing.overload
    def build_serialized_network(self, network: INetworkDefinition, config: IBuilderConfig, kernel_text: typing.Any) -> IHostMemory:
        """
            Builds and serializes a network for the given :class:`INetworkDefinition` and :class:`IBuilderConfig` .

            This function allows building and serialization of a network without creating an engine.

            :arg network: Network definition.
            :arg config: Builder configuration.

            :returns: A pointer to a :class:`IHostMemory` object that contains a serialized network.
        """
    def build_serialized_network_to_stream(self, network: INetworkDefinition, config: IBuilderConfig, writer: IStreamWriter) -> bool:
        """
            Builds and serializes a network for the given :class:`INetworkDefinition` and :class:`IBuilderConfig` and write the serialized network to a writer stream.

            This function allows building and serialization of a network without creating an engine.

            :arg network: Network definition.
            :arg config: Builder configuration.
            :arg writer: Output stream writer.

            :returns: True if build succeed, otherwise False.
        """
    def create_builder_config(self) -> IBuilderConfig:
        """
            Create a builder configuration object.

            See :class:`IBuilderConfig`
        """
    def create_network(self, flags: typing.SupportsInt = 0) -> INetworkDefinition:
        """
            Create a :class:`INetworkDefinition` object.

            :arg flags: :class:`NetworkDefinitionCreationFlag` s combined using bitwise OR.

            :returns: An empty TensorRT :class:`INetworkDefinition` .
        """
    def create_optimization_profile(self) -> IOptimizationProfile:
        """
            Create a new optimization profile.

            If the network has any dynamic input tensors, the appropriate calls to :func:`IOptimizationProfile.set_shape` must be made. Likewise, if there are any shape input tensors, the appropriate calls to :func:`IOptimizationProfile.set_shape_input` are required.

            See :class:`IOptimizationProfile`
        """
    def get_plugin_registry(self) -> IPluginRegistry:
        """
            Get the local plugin registry that can be used by the builder.

            :returns: The local plugin registry that can be used by the builder.
        """
    def is_network_supported(self, network: INetworkDefinition, config: IBuilderConfig) -> bool:
        """
            Checks that a network is within the scope of the :class:`IBuilderConfig` settings.

            :arg network: The network definition to check for configuration compliance.
            :arg config: The configuration of the builder to use when checking the network.

            Given an :class:`INetworkDefinition` and an :class:`IBuilderConfig` , check if
            the network falls within the constraints of the builder configuration based on the
            :class:`EngineCapability` , :class:`BuilderFlag` , and :class:`DeviceType` .

            :returns: ``True`` if network is within the scope of the restrictions specified by the builder config, ``False`` otherwise.
                This function reports the conditions that are violated to the registered :class:`ErrorRecorder` .

            NOTE: A ``True`` return value does not guarantee that engine building will succeed, as backends may reject it for
                reasons not detectable with this fast validation. To definitively check whether a network can be built with a given config,
                use :func:`build_engine_with_config` or :func:`build_serialized_network` (depending on the engine capability).

            NOTE: This function will synchronize the cuda stream returned by ``config.profile_stream`` before returning.
        """
    def reset(self) -> None:
        """
            Resets the builder state to default values.
        """
    @property
    def logger(self) -> ILogger:
        ...
    @property
    def max_DLA_batch_size(self) -> int:
        ...
    @property
    def max_threads(self) -> int:
        ...
    @max_threads.setter
    def max_threads(self, arg1: typing.SupportsInt) -> bool:
        ...
    @property
    def num_DLA_cores(self) -> int:
        ...
    @property
    def platform_has_fast_fp16(self) -> bool:
        ...
    @property
    def platform_has_fast_int8(self) -> bool:
        ...
    @property
    def platform_has_tf32(self) -> bool:
        ...
class BuilderFlag:
    """
    Valid modes that the builder can enable when creating an engine from a network definition.

    Members:

      FP16 : Enable FP16 layer selection.
                    [DEPRECATED] Deprecated in TensorRT 10.12. Superseded by strong typing.

      BF16 : Enable BF16 layer selection.
                   [DEPRECATED] Deprecated in TensorRT 10.12. Superseded by strong typing.

      INT8 : Enable Int8 layer selection.
                   [DEPRECATED] Deprecated in TensorRT 10.12. Superseded by strong typing.

      DEBUG : Enable debugging of layers via synchronizing after every layer

      GPU_FALLBACK : Enable layers marked to execute on GPU if layer cannot execute on DLA

      REFIT : Enable building a refittable engine

      DISABLE_TIMING_CACHE : Disable reuse of timing information across identical layers.

      EDITABLE_TIMING_CACHE : Enable the editable timing cache.

      TF32 : Allow (but not require) computations on tensors of type DataType.FLOAT to use TF32. TF32 computes inner products by rounding the inputs to 10-bit mantissas before multiplying, but accumulates the sum using 23-bit mantissas. Enabled by default.

      SPARSE_WEIGHTS : Allow the builder to examine weights and use optimized functions when weights have suitable sparsity.

      SAFETY_SCOPE : Change the allowed parameters in the EngineCapability.STANDARD flow to match the restrictions that EngineCapability.SAFETY check against for DeviceType.GPU and EngineCapability.DLA_STANDALONE check against the DeviceType.DLA case. This flag is forced to true if EngineCapability.SAFETY at build time if it is unset.
                   [DEPRECATED] Deprecated in TensorRT 10.16. In EngineCapability.STANDARD flow, safety restrictions are no longer supported. In EngineCapability.SAFETY and EngineCapability.DLA_STANDALONE flows, restrictions are enforced natively. This flag is retained for API compatibility but is ignored.

      OBEY_PRECISION_CONSTRAINTS : Require that layers execute in specified precisions. Build fails otherwise.
                   [DEPRECATED] Deprecated in TensorRT 10.12. Superseded by strong typing.

      PREFER_PRECISION_CONSTRAINTS : Prefer that layers execute in specified precisions. Fall back (with warning) to another precision if build would otherwise fail.
                   [DEPRECATED] Deprecated in TensorRT 10.12. Superseded by strong typing.

      DIRECT_IO : Require that no reformats be inserted between a layer and a network I/O tensor for which ``ITensor.allowed_formats`` was set. Build fails if a reformat is required for functional correctness.
                   [DEPRECATED] Deprecated in TensorRT 10.7.)

      REJECT_EMPTY_ALGORITHMS : [DEPRECATED] Deprecated in TensorRT 10.8. Please use editable mode in ITimingCache instead. Fail if IAlgorithmSelector.select_algorithms returns an empty set of algorithms.

      VERSION_COMPATIBLE : Restrict to lean runtime operators to provide version forward compatibility for the plan files.

      EXCLUDE_LEAN_RUNTIME : Exclude lean runtime from the plan.

      FP8 : Enable plugins with FP8 input/output
        [DEPRECATED] Deprecated in TensorRT 10.12. Superseded by strong typing.

      ERROR_ON_TIMING_CACHE_MISS : Emit error when a tactic being timed is not present in the timing cache.

      DISABLE_COMPILATION_CACHE : Disable caching JIT compilation results during engine build.

      WEIGHTLESS : Strip the perf-irrelevant weights from the plan file, update them later using refitting for better file size.
                   [DEPRECATED] Deprecated in TensorRT 10.0.


      STRIP_PLAN : Strip the refittable weights from the engine plan file.

      REFIT_IDENTICAL : Create a refittable engine using identical weights. Different weights during refits yield unpredictable behavior.

      WEIGHT_STREAMING : Enable building with the ability to stream varying amounts of weights during Runtime. This decreases GPU memory of TRT at the expense of performance.

      INT4 : Enable plugins with INT4 input/output
                   [DEPRECATED] Deprecated in TensorRT 10.12. Superseded by strong typing.

      REFIT_INDIVIDUAL : Create a refittable engine and allows the users to specify which weights are refittable and which are not.

      STRICT_NANS : Disable floating-point optimizations: 0*x => 0, x-x => 0, or x/x => 1. These identities are not true when x is a NaN or Inf, and thus might hide propagation or generation of NaNs.

      MONITOR_MEMORY : Enable memory monitor during build time.

      FP4 : Enable plugins with FP4 input/output
                   [DEPRECATED] Deprecated in TensorRT 10.12. Superseded by strong typing.

      DISTRIBUTIVE_INDEPENDENCE : Enable distributive independence.
    """
    BF16: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.BF16: 17>
    DEBUG: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.DEBUG: 2>
    DIRECT_IO: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.DIRECT_IO: 11>
    DISABLE_COMPILATION_CACHE: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.DISABLE_COMPILATION_CACHE: 18>
    DISABLE_TIMING_CACHE: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.DISABLE_TIMING_CACHE: 5>
    DISTRIBUTIVE_INDEPENDENCE: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.DISTRIBUTIVE_INDEPENDENCE: 28>
    EDITABLE_TIMING_CACHE: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.EDITABLE_TIMING_CACHE: 27>
    ERROR_ON_TIMING_CACHE_MISS: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.ERROR_ON_TIMING_CACHE_MISS: 16>
    EXCLUDE_LEAN_RUNTIME: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.EXCLUDE_LEAN_RUNTIME: 14>
    FP16: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.FP16: 0>
    FP4: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.FP4: 26>
    FP8: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.FP8: 15>
    GPU_FALLBACK: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.GPU_FALLBACK: 3>
    INT4: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.INT4: 22>
    INT8: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.INT8: 1>
    MONITOR_MEMORY: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.MONITOR_MEMORY: 25>
    OBEY_PRECISION_CONSTRAINTS: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.OBEY_PRECISION_CONSTRAINTS: 9>
    PREFER_PRECISION_CONSTRAINTS: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.PREFER_PRECISION_CONSTRAINTS: 10>
    REFIT: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.REFIT: 4>
    REFIT_IDENTICAL: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.REFIT_IDENTICAL: 20>
    REFIT_INDIVIDUAL: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.REFIT_INDIVIDUAL: 23>
    REJECT_EMPTY_ALGORITHMS: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.REJECT_EMPTY_ALGORITHMS: 12>
    SAFETY_SCOPE: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.SAFETY_SCOPE: 8>
    SPARSE_WEIGHTS: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.SPARSE_WEIGHTS: 7>
    STRICT_NANS: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.STRICT_NANS: 24>
    STRIP_PLAN: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.WEIGHTLESS: 19>
    TF32: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.TF32: 6>
    VERSION_COMPATIBLE: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.VERSION_COMPATIBLE: 13>
    WEIGHTLESS: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.WEIGHTLESS: 19>
    WEIGHT_STREAMING: typing.ClassVar[BuilderFlag]  # value = <BuilderFlag.WEIGHT_STREAMING: 21>
    __members__: typing.ClassVar[dict[str, BuilderFlag]]  # value = {'FP16': <BuilderFlag.FP16: 0>, 'BF16': <BuilderFlag.BF16: 17>, 'INT8': <BuilderFlag.INT8: 1>, 'DEBUG': <BuilderFlag.DEBUG: 2>, 'GPU_FALLBACK': <BuilderFlag.GPU_FALLBACK: 3>, 'REFIT': <BuilderFlag.REFIT: 4>, 'DISABLE_TIMING_CACHE': <BuilderFlag.DISABLE_TIMING_CACHE: 5>, 'EDITABLE_TIMING_CACHE': <BuilderFlag.EDITABLE_TIMING_CACHE: 27>, 'TF32': <BuilderFlag.TF32: 6>, 'SPARSE_WEIGHTS': <BuilderFlag.SPARSE_WEIGHTS: 7>, 'SAFETY_SCOPE': <BuilderFlag.SAFETY_SCOPE: 8>, 'OBEY_PRECISION_CONSTRAINTS': <BuilderFlag.OBEY_PRECISION_CONSTRAINTS: 9>, 'PREFER_PRECISION_CONSTRAINTS': <BuilderFlag.PREFER_PRECISION_CONSTRAINTS: 10>, 'DIRECT_IO': <BuilderFlag.DIRECT_IO: 11>, 'REJECT_EMPTY_ALGORITHMS': <BuilderFlag.REJECT_EMPTY_ALGORITHMS: 12>, 'VERSION_COMPATIBLE': <BuilderFlag.VERSION_COMPATIBLE: 13>, 'EXCLUDE_LEAN_RUNTIME': <BuilderFlag.EXCLUDE_LEAN_RUNTIME: 14>, 'FP8': <BuilderFlag.FP8: 15>, 'ERROR_ON_TIMING_CACHE_MISS': <BuilderFlag.ERROR_ON_TIMING_CACHE_MISS: 16>, 'DISABLE_COMPILATION_CACHE': <BuilderFlag.DISABLE_COMPILATION_CACHE: 18>, 'WEIGHTLESS': <BuilderFlag.WEIGHTLESS: 19>, 'STRIP_PLAN': <BuilderFlag.WEIGHTLESS: 19>, 'REFIT_IDENTICAL': <BuilderFlag.REFIT_IDENTICAL: 20>, 'WEIGHT_STREAMING': <BuilderFlag.WEIGHT_STREAMING: 21>, 'INT4': <BuilderFlag.INT4: 22>, 'REFIT_INDIVIDUAL': <BuilderFlag.REFIT_INDIVIDUAL: 23>, 'STRICT_NANS': <BuilderFlag.STRICT_NANS: 24>, 'MONITOR_MEMORY': <BuilderFlag.MONITOR_MEMORY: 25>, 'FP4': <BuilderFlag.FP4: 26>, 'DISTRIBUTIVE_INDEPENDENCE': <BuilderFlag.DISTRIBUTIVE_INDEPENDENCE: 28>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class CalibrationAlgoType:
    """

        [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization.

        Version of calibration algorithm to use.


    Members:

      LEGACY_CALIBRATION

      ENTROPY_CALIBRATION

      ENTROPY_CALIBRATION_2

      MINMAX_CALIBRATION
    """
    ENTROPY_CALIBRATION: typing.ClassVar[CalibrationAlgoType]  # value = <CalibrationAlgoType.ENTROPY_CALIBRATION: 1>
    ENTROPY_CALIBRATION_2: typing.ClassVar[CalibrationAlgoType]  # value = <CalibrationAlgoType.ENTROPY_CALIBRATION_2: 2>
    LEGACY_CALIBRATION: typing.ClassVar[CalibrationAlgoType]  # value = <CalibrationAlgoType.LEGACY_CALIBRATION: 0>
    MINMAX_CALIBRATION: typing.ClassVar[CalibrationAlgoType]  # value = <CalibrationAlgoType.MINMAX_CALIBRATION: 3>
    __members__: typing.ClassVar[dict[str, CalibrationAlgoType]]  # value = {'LEGACY_CALIBRATION': <CalibrationAlgoType.LEGACY_CALIBRATION: 0>, 'ENTROPY_CALIBRATION': <CalibrationAlgoType.ENTROPY_CALIBRATION: 1>, 'ENTROPY_CALIBRATION_2': <CalibrationAlgoType.ENTROPY_CALIBRATION_2: 2>, 'MINMAX_CALIBRATION': <CalibrationAlgoType.MINMAX_CALIBRATION: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class CollectiveOperation:
    """
    The collective operations that may be performed by a DistCollective layer

    Members:

      ALL_REDUCE : All reduce collective operation

      ALL_GATHER : All gather collective operation

      BROADCAST : Broadcast collective operation

      REDUCE : Reduce collective operation

      REDUCE_SCATTER : Reduce scatter collective operation
    """
    ALL_GATHER: typing.ClassVar[CollectiveOperation]  # value = <CollectiveOperation.ALL_GATHER: 1>
    ALL_REDUCE: typing.ClassVar[CollectiveOperation]  # value = <CollectiveOperation.ALL_REDUCE: 0>
    BROADCAST: typing.ClassVar[CollectiveOperation]  # value = <CollectiveOperation.BROADCAST: 2>
    REDUCE: typing.ClassVar[CollectiveOperation]  # value = <CollectiveOperation.REDUCE: 3>
    REDUCE_SCATTER: typing.ClassVar[CollectiveOperation]  # value = <CollectiveOperation.REDUCE_SCATTER: 4>
    __members__: typing.ClassVar[dict[str, CollectiveOperation]]  # value = {'ALL_REDUCE': <CollectiveOperation.ALL_REDUCE: 0>, 'ALL_GATHER': <CollectiveOperation.ALL_GATHER: 1>, 'BROADCAST': <CollectiveOperation.BROADCAST: 2>, 'REDUCE': <CollectiveOperation.REDUCE: 3>, 'REDUCE_SCATTER': <CollectiveOperation.REDUCE_SCATTER: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class CumulativeOperation:
    """
    The cumulative operations that may be performed by a Cumulative layer

    Members:

      SUM :
    """
    SUM: typing.ClassVar[CumulativeOperation]  # value = <CumulativeOperation.SUM: 0>
    __members__: typing.ClassVar[dict[str, CumulativeOperation]]  # value = {'SUM': <CumulativeOperation.SUM: 0>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class DataType:
    """

        Represents data types.

        :ivar itemsize: :class:`int` The size in bytes of this :class:`DataType` .


    Members:

      FLOAT : 32-bit floating point format.

      HALF : IEEE 16-bit floating-point format.

      BF16 : Brain float -- has an 8 bit exponent and 8 bit significand

      INT8 : Signed 8-bit integer representing a quantized floating-point value.

      INT32 : Signed 32-bit integer format.

      INT64 : Signed 64-bit integer format.

      BOOL : 8-bit boolean. 0 = false, 1 = true, other values undefined.

      UINT8 :
        Unsigned 8-bit integer format.
        Cannot be used to represent quantized floating-point values.
        Use the IdentityLayer to convert ``uint8`` network-level inputs to {``float32``, ``float16``} prior
        to use with other TensorRT layers, or to convert intermediate output
        before ``uint8`` network-level outputs from {``float32``, ``float16``} to ``uint8``.
        ``uint8`` conversions are only supported for {``float32``, ``float16``}.
        ``uint8`` to {``float32``, ``float16``} conversion will convert the integer values
        to equivalent floating point values.
        {``float32``, ``float16``} to ``uint8`` conversion will convert the floating point values
        to integer values by truncating towards zero. This conversion has undefined behavior for
        floating point values outside the range [0.0f, 256.0) after truncation.
        ``uint8`` conversions are not supported for {``int8``, ``int32``, ``bool``}.


      FP8 :
        Signed 8-bit floating point with 1 sign bit, 4 exponent bits, 3 mantissa
        bits, and exponent-bias 7.


      INT4 : Signed 4-bit integer representing a quantized floating-point value.

      FP4 : Signed 4-bit floating point with 1 sign bit, 2 exponent bits and 1 mantissa bits.

      E8M0 : Unsigned 8-bit exponent-only floating point.
    """
    BF16: typing.ClassVar[DataType]  # value = <DataType.BF16: 7>
    BOOL: typing.ClassVar[DataType]  # value = <DataType.BOOL: 4>
    E8M0: typing.ClassVar[DataType]  # value = <DataType.E8M0: 11>
    FLOAT: typing.ClassVar[DataType]  # value = <DataType.FLOAT: 0>
    FP4: typing.ClassVar[DataType]  # value = <DataType.FP4: 10>
    FP8: typing.ClassVar[DataType]  # value = <DataType.FP8: 6>
    HALF: typing.ClassVar[DataType]  # value = <DataType.HALF: 1>
    INT32: typing.ClassVar[DataType]  # value = <DataType.INT32: 3>
    INT4: typing.ClassVar[DataType]  # value = <DataType.INT4: 9>
    INT64: typing.ClassVar[DataType]  # value = <DataType.INT64: 8>
    INT8: typing.ClassVar[DataType]  # value = <DataType.INT8: 2>
    UINT8: typing.ClassVar[DataType]  # value = <DataType.UINT8: 5>
    __members__: typing.ClassVar[dict[str, DataType]]  # value = {'FLOAT': <DataType.FLOAT: 0>, 'HALF': <DataType.HALF: 1>, 'BF16': <DataType.BF16: 7>, 'INT8': <DataType.INT8: 2>, 'INT32': <DataType.INT32: 3>, 'INT64': <DataType.INT64: 8>, 'BOOL': <DataType.BOOL: 4>, 'UINT8': <DataType.UINT8: 5>, 'FP8': <DataType.FP8: 6>, 'INT4': <DataType.INT4: 9>, 'FP4': <DataType.FP4: 10>, 'E8M0': <DataType.E8M0: 11>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def itemsize(self):
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class DeviceType:
    """
    Device types that TensorRT can execute on

    Members:

      GPU : GPU device

      DLA : DLA core
    """
    DLA: typing.ClassVar[DeviceType]  # value = <DeviceType.DLA: 1>
    GPU: typing.ClassVar[DeviceType]  # value = <DeviceType.GPU: 0>
    __members__: typing.ClassVar[dict[str, DeviceType]]  # value = {'GPU': <DeviceType.GPU: 0>, 'DLA': <DeviceType.DLA: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class DimensionOperation:
    """

        An operation on two IDimensionExprs, which represent integer expressions used in dimension computations.

        For example, given two IDimensionExprs x and y and an IExprBuilder eb, eb.operation(DimensionOperation.SUM, x, y) creates a representation of x + y.


    Members:

      SUM

      PROD

      MAX

      MIN

      SUB

      EQUAL

      LESS

      FLOOR_DIV

      CEIL_DIV
    """
    CEIL_DIV: typing.ClassVar[DimensionOperation]  # value = <DimensionOperation.CEIL_DIV: 8>
    EQUAL: typing.ClassVar[DimensionOperation]  # value = <DimensionOperation.EQUAL: 5>
    FLOOR_DIV: typing.ClassVar[DimensionOperation]  # value = <DimensionOperation.FLOOR_DIV: 7>
    LESS: typing.ClassVar[DimensionOperation]  # value = <DimensionOperation.LESS: 6>
    MAX: typing.ClassVar[DimensionOperation]  # value = <DimensionOperation.MAX: 2>
    MIN: typing.ClassVar[DimensionOperation]  # value = <DimensionOperation.MIN: 3>
    PROD: typing.ClassVar[DimensionOperation]  # value = <DimensionOperation.PROD: 1>
    SUB: typing.ClassVar[DimensionOperation]  # value = <DimensionOperation.SUB: 4>
    SUM: typing.ClassVar[DimensionOperation]  # value = <DimensionOperation.SUM: 0>
    __members__: typing.ClassVar[dict[str, DimensionOperation]]  # value = {'SUM': <DimensionOperation.SUM: 0>, 'PROD': <DimensionOperation.PROD: 1>, 'MAX': <DimensionOperation.MAX: 2>, 'MIN': <DimensionOperation.MIN: 3>, 'SUB': <DimensionOperation.SUB: 4>, 'EQUAL': <DimensionOperation.EQUAL: 5>, 'LESS': <DimensionOperation.LESS: 6>, 'FLOOR_DIV': <DimensionOperation.FLOOR_DIV: 7>, 'CEIL_DIV': <DimensionOperation.CEIL_DIV: 8>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Dims:
    """

        Structure to define the dimensions of a tensor. :class:`Dims` and all derived classes behave like Python :class:`tuple` s. Furthermore, the TensorRT API can implicitly convert Python iterables to :class:`Dims` objects, so :class:`tuple` or :class:`list` can be used in place of this class.
    """
    MAX_DIMS: typing.ClassVar[int] = 8
    __hash__: typing.ClassVar[None] = None
    @typing.overload
    def __eq__(self, arg0: list) -> bool:
        ...
    @typing.overload
    def __eq__(self, arg0: tuple) -> bool:
        ...
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> int:
        ...
    @typing.overload
    def __getitem__(self, arg0: slice) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, shape: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: Dims) -> None:
        ...
    def __str__(self) -> str:
        ...
class Dims2(Dims):
    """

        Structure to define 2D shape.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, dim0: typing.SupportsInt, dim1: typing.SupportsInt) -> None:
        ...
    @typing.overload
    def __init__(self, shape: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
class Dims3(Dims):
    """

        Structure to define 3D shape.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, dim0: typing.SupportsInt, dim1: typing.SupportsInt, dim2: typing.SupportsInt) -> None:
        ...
    @typing.overload
    def __init__(self, shape: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
class Dims4(Dims):
    """

        Structure to define 4D tensor.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, dim0: typing.SupportsInt, dim1: typing.SupportsInt, dim2: typing.SupportsInt, dim3: typing.SupportsInt) -> None:
        ...
    @typing.overload
    def __init__(self, shape: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
class DimsExprs:
    """

        Analog of class `Dims` with expressions (`IDimensionExpr`) instead of constants for the dimensions.

        Behaves like a Python iterable and lists or tuples of `IDimensionExpr` can be used to construct it.
    """
    def __getitem__(self, arg0: typing.SupportsInt) -> IDimensionExpr:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: collections.abc.Sequence[IDimensionExpr]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: typing.SupportsInt) -> None:
        ...
    def __len__(self) -> int:
        ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: IDimensionExpr) -> None:
        ...
class DimsHW(Dims2):
    """

        Structure to define 2D shape with height and width.

        :ivar h: :class:`int` The first dimension (height).
        :ivar w: :class:`int` The second dimension (width).
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, h: typing.SupportsInt, w: typing.SupportsInt) -> None:
        ...
    @typing.overload
    def __init__(self, shape: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    @property
    def h(self) -> int:
        ...
    @h.setter
    def h(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def w(self) -> int:
        ...
    @w.setter
    def w(self, arg1: typing.SupportsInt) -> None:
        ...
class DynamicPluginTensorDesc:
    """

        Summarizes tensors that a plugin might see for an input or output.

        :ivar desc: :class:`PluginTensorDesc` Information required to interpret a pointer to tensor data, except that desc.dims has -1 in place of any runtime dimension..
        :ivar min: :class:`Dims` 	Lower bounds on tensor's dimensions.
        :ivar max: :class:`Dims` 	Upper bounds on tensor's dimensions.
    """
    desc: PluginTensorDesc
    max: Dims
    min: Dims
    opt: Dims
    def __init__(self) -> None:
        ...
class ElementWiseOperation:
    """
    The binary operations that may be performed by an ElementWise layer.

    Members:

      SUM : Sum of the two elements

      PROD : Product of the two elements

      MAX : Max of the two elements

      MIN : Min of the two elements

      SUB : Subtract the second element from the first

      DIV : Divide the first element by the second

      POW : The first element to the power of the second element

      FLOOR_DIV : Floor division of the first element by the second

      AND : Logical AND of two elements

      OR : Logical OR of two elements

      XOR : Logical XOR of two elements

      EQUAL : Check if two elements are equal

      GREATER : Check if element in first tensor is greater than corresponding element in second tensor

      LESS : Check if element in first tensor is less than corresponding element in second tensor
    """
    AND: typing.ClassVar[ElementWiseOperation]  # value = <ElementWiseOperation.AND: 8>
    DIV: typing.ClassVar[ElementWiseOperation]  # value = <ElementWiseOperation.DIV: 5>
    EQUAL: typing.ClassVar[ElementWiseOperation]  # value = <ElementWiseOperation.EQUAL: 11>
    FLOOR_DIV: typing.ClassVar[ElementWiseOperation]  # value = <ElementWiseOperation.FLOOR_DIV: 7>
    GREATER: typing.ClassVar[ElementWiseOperation]  # value = <ElementWiseOperation.GREATER: 12>
    LESS: typing.ClassVar[ElementWiseOperation]  # value = <ElementWiseOperation.LESS: 13>
    MAX: typing.ClassVar[ElementWiseOperation]  # value = <ElementWiseOperation.MAX: 2>
    MIN: typing.ClassVar[ElementWiseOperation]  # value = <ElementWiseOperation.MIN: 3>
    OR: typing.ClassVar[ElementWiseOperation]  # value = <ElementWiseOperation.OR: 9>
    POW: typing.ClassVar[ElementWiseOperation]  # value = <ElementWiseOperation.POW: 6>
    PROD: typing.ClassVar[ElementWiseOperation]  # value = <ElementWiseOperation.PROD: 1>
    SUB: typing.ClassVar[ElementWiseOperation]  # value = <ElementWiseOperation.SUB: 4>
    SUM: typing.ClassVar[ElementWiseOperation]  # value = <ElementWiseOperation.SUM: 0>
    XOR: typing.ClassVar[ElementWiseOperation]  # value = <ElementWiseOperation.XOR: 10>
    __members__: typing.ClassVar[dict[str, ElementWiseOperation]]  # value = {'SUM': <ElementWiseOperation.SUM: 0>, 'PROD': <ElementWiseOperation.PROD: 1>, 'MAX': <ElementWiseOperation.MAX: 2>, 'MIN': <ElementWiseOperation.MIN: 3>, 'SUB': <ElementWiseOperation.SUB: 4>, 'DIV': <ElementWiseOperation.DIV: 5>, 'POW': <ElementWiseOperation.POW: 6>, 'FLOOR_DIV': <ElementWiseOperation.FLOOR_DIV: 7>, 'AND': <ElementWiseOperation.AND: 8>, 'OR': <ElementWiseOperation.OR: 9>, 'XOR': <ElementWiseOperation.XOR: 10>, 'EQUAL': <ElementWiseOperation.EQUAL: 11>, 'GREATER': <ElementWiseOperation.GREATER: 12>, 'LESS': <ElementWiseOperation.LESS: 13>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class EngineCapability:
    """

        List of supported engine capability flows.
        The EngineCapability determines the restrictions of a network during build time and what runtime
        it targets. EngineCapability.STANDARD does not provide any restrictions on functionality and the resulting
        serialized engine can be executed with TensorRT's standard runtime APIs in the nvinfer1 namespace.
        EngineCapability.SAFETY provides a restricted subset of network operations that are safety certified and
        the resulting serialized engine can be executed with TensorRT's safe runtime APIs in the `nvinfer2::safe` namespace.
        EngineCapability.DLA_STANDALONE provides a restricted subset of network operations that are DLA compatible and
        the resulting serialized engine can be executed using standalone DLA runtime APIs. See sampleCudla for an
        example of integrating cuDLA APIs with TensorRT APIs.

    Members:

      STANDARD : Standard: TensorRT flow without targeting the standard runtime. This flow supports both DeviceType::kGPU and DeviceType::kDLA.

      SAFETY : Safety: TensorRT flow with restrictions targeting the safety runtime. See safety documentation for list of supported layers and formats. This flow supports only DeviceType::kGPU.

      DLA_STANDALONE : DLA Standalone: TensorRT flow with restrictions targeting external, to TensorRT, DLA runtimes. See DLA documentation for list of supported layers and formats. This flow supports only DeviceType::kDLA.
    """
    DLA_STANDALONE: typing.ClassVar[EngineCapability]  # value = <EngineCapability.DLA_STANDALONE: 2>
    SAFETY: typing.ClassVar[EngineCapability]  # value = <EngineCapability.SAFETY: 1>
    STANDARD: typing.ClassVar[EngineCapability]  # value = <EngineCapability.STANDARD: 0>
    __members__: typing.ClassVar[dict[str, EngineCapability]]  # value = {'STANDARD': <EngineCapability.STANDARD: 0>, 'SAFETY': <EngineCapability.SAFETY: 1>, 'DLA_STANDALONE': <EngineCapability.DLA_STANDALONE: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class EngineInspector:
    """

        An engine inspector which prints out the layer information of an engine or an execution context.

        The amount of printed information depends on the profiling verbosity setting of the builder config when the engine is built.
        By default, the profiling verbosity is set to ProfilingVerbosity.LAYER_NAMES_ONLY, and only layer names will be printed.
        If the profiling verbosity is set to ProfilingVerbosity.DETAILED, layer names and layer parameters will be printed.
        If the profiling verbosity is set to ProfilingVerbosity.NONE, no layer information will be printed.

        :ivar execution_context: :class:`IExecutionContext` Set or get context currently being inspected.
        :ivar error_recorder: :class:`IErrorRecorder` Application-implemented error reporting interface for TensorRT objects.
    """
    error_recorder: IErrorRecorder
    execution_context: IExecutionContext
    def get_engine_information(self, format: LayerInformationFormat) -> str:
        """
            Get a string describing the information about all the layers in the current engine or the execution context.

            :arg format: :class:`LayerInformationFormat` The format the layer information should be printed in.

            :returns: A string describing the information about all the layers in the current engine or the execution context.
        """
    def get_layer_information(self, layer_index: typing.SupportsInt, format: LayerInformationFormat) -> str:
        """
            Get a string describing the information about a specific layer in the current engine or the execution context.

            :arg layer_index: The index of the layer. It must lie in [0, engine.num_layers].
            :arg format: :class:`LayerInformationFormat` The format the layer information should be printed in.

            :returns: A string describing the information about a specific layer in the current engine or the execution context.
        """
class EngineStat:
    """
    The kind of engine statistics that queried from the ICudaEngine.

    Members:

      TOTAL_WEIGHTS_SIZE : The total weights size in bytes in the engine.

      STRIPPED_WEIGHTS_SIZE : The stripped weight size in bytes for engines built with BuilderFlag::kSTRIP_PLAN.
    """
    STRIPPED_WEIGHTS_SIZE: typing.ClassVar[EngineStat]  # value = <EngineStat.STRIPPED_WEIGHTS_SIZE: 1>
    TOTAL_WEIGHTS_SIZE: typing.ClassVar[EngineStat]  # value = <EngineStat.TOTAL_WEIGHTS_SIZE: 0>
    __members__: typing.ClassVar[dict[str, EngineStat]]  # value = {'TOTAL_WEIGHTS_SIZE': <EngineStat.TOTAL_WEIGHTS_SIZE: 0>, 'STRIPPED_WEIGHTS_SIZE': <EngineStat.STRIPPED_WEIGHTS_SIZE: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ErrorCode:
    """

        The type of parser error


    Members:

      SUCCESS

      INTERNAL_ERROR

      MEM_ALLOC_FAILED

      MODEL_DESERIALIZE_FAILED

      INVALID_VALUE

      INVALID_GRAPH

      INVALID_NODE

      UNSUPPORTED_GRAPH

      UNSUPPORTED_NODE

      UNSUPPORTED_NODE_ATTR

      UNSUPPORTED_NODE_INPUT

      UNSUPPORTED_NODE_DATATYPE

      UNSUPPORTED_NODE_DYNAMIC

      UNSUPPORTED_NODE_SHAPE

      REFIT_FAILED
    """
    INTERNAL_ERROR: typing.ClassVar[ErrorCode]  # value = <ErrorCode.INTERNAL_ERROR: 1>
    INVALID_GRAPH: typing.ClassVar[ErrorCode]  # value = <ErrorCode.INVALID_GRAPH: 5>
    INVALID_NODE: typing.ClassVar[ErrorCode]  # value = <ErrorCode.INVALID_NODE: 6>
    INVALID_VALUE: typing.ClassVar[ErrorCode]  # value = <ErrorCode.INVALID_VALUE: 4>
    MEM_ALLOC_FAILED: typing.ClassVar[ErrorCode]  # value = <ErrorCode.MEM_ALLOC_FAILED: 2>
    MODEL_DESERIALIZE_FAILED: typing.ClassVar[ErrorCode]  # value = <ErrorCode.MODEL_DESERIALIZE_FAILED: 3>
    REFIT_FAILED: typing.ClassVar[ErrorCode]  # value = <ErrorCode.REFIT_FAILED: 14>
    SUCCESS: typing.ClassVar[ErrorCode]  # value = <ErrorCode.SUCCESS: 0>
    UNSUPPORTED_GRAPH: typing.ClassVar[ErrorCode]  # value = <ErrorCode.UNSUPPORTED_GRAPH: 7>
    UNSUPPORTED_NODE: typing.ClassVar[ErrorCode]  # value = <ErrorCode.UNSUPPORTED_NODE: 8>
    UNSUPPORTED_NODE_ATTR: typing.ClassVar[ErrorCode]  # value = <ErrorCode.UNSUPPORTED_NODE_ATTR: 9>
    UNSUPPORTED_NODE_DATATYPE: typing.ClassVar[ErrorCode]  # value = <ErrorCode.UNSUPPORTED_NODE_DATATYPE: 11>
    UNSUPPORTED_NODE_DYNAMIC: typing.ClassVar[ErrorCode]  # value = <ErrorCode.UNSUPPORTED_NODE_DYNAMIC: 12>
    UNSUPPORTED_NODE_INPUT: typing.ClassVar[ErrorCode]  # value = <ErrorCode.UNSUPPORTED_NODE_INPUT: 10>
    UNSUPPORTED_NODE_SHAPE: typing.ClassVar[ErrorCode]  # value = <ErrorCode.UNSUPPORTED_NODE_SHAPE: 13>
    __members__: typing.ClassVar[dict[str, ErrorCode]]  # value = {'SUCCESS': <ErrorCode.SUCCESS: 0>, 'INTERNAL_ERROR': <ErrorCode.INTERNAL_ERROR: 1>, 'MEM_ALLOC_FAILED': <ErrorCode.MEM_ALLOC_FAILED: 2>, 'MODEL_DESERIALIZE_FAILED': <ErrorCode.MODEL_DESERIALIZE_FAILED: 3>, 'INVALID_VALUE': <ErrorCode.INVALID_VALUE: 4>, 'INVALID_GRAPH': <ErrorCode.INVALID_GRAPH: 5>, 'INVALID_NODE': <ErrorCode.INVALID_NODE: 6>, 'UNSUPPORTED_GRAPH': <ErrorCode.UNSUPPORTED_GRAPH: 7>, 'UNSUPPORTED_NODE': <ErrorCode.UNSUPPORTED_NODE: 8>, 'UNSUPPORTED_NODE_ATTR': <ErrorCode.UNSUPPORTED_NODE_ATTR: 9>, 'UNSUPPORTED_NODE_INPUT': <ErrorCode.UNSUPPORTED_NODE_INPUT: 10>, 'UNSUPPORTED_NODE_DATATYPE': <ErrorCode.UNSUPPORTED_NODE_DATATYPE: 11>, 'UNSUPPORTED_NODE_DYNAMIC': <ErrorCode.UNSUPPORTED_NODE_DYNAMIC: 12>, 'UNSUPPORTED_NODE_SHAPE': <ErrorCode.UNSUPPORTED_NODE_SHAPE: 13>, 'REFIT_FAILED': <ErrorCode.REFIT_FAILED: 14>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    @typing.overload
    def __repr__(self) -> str:
        ...
    @typing.overload
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    @typing.overload
    def __str__(self) -> str:
        ...
    @typing.overload
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ErrorCodeTRT:
    """
    Error codes that can be returned by TensorRT during execution.

    Members:

      SUCCESS : Execution completed successfully.

      UNSPECIFIED_ERROR :
        An error that does not fall into any other category. This error is included for forward compatibility.


      INTERNAL_ERROR : A non-recoverable TensorRT error occurred.

      INVALID_ARGUMENT :
        An argument passed to the function is invalid in isolation. This is a violation of the API contract.


      INVALID_CONFIG :
        An error occurred when comparing the state of an argument relative to other arguments. For example, the
        dimensions for concat differ between two tensors outside of the channel dimension. This error is triggered
        when an argument is correct in isolation, but not relative to other arguments. This is to help to distinguish
        from the simple errors from the more complex errors.
        This is a violation of the API contract.


      FAILED_ALLOCATION :
        An error occurred when performing an allocation of memory on the host or the device.
        A memory allocation error is normally fatal, but in the case where the application provided its own memory
        allocation routine, it is possible to increase the pool of available memory and resume execution.


      FAILED_INITIALIZATION :
        One, or more, of the components that TensorRT relies on did not initialize correctly.
        This is a system setup issue.


      FAILED_EXECUTION :
        An error occurred during execution that caused TensorRT to end prematurely, either an asynchronous error,
        user cancellation, or other execution errors reported by CUDA/DLA. In a dynamic system, the
        data can be thrown away and the next frame can be processed or execution can be retried.
        This is either an execution error or a memory error.


      FAILED_COMPUTATION :
        An error occurred during execution that caused the data to become corrupted, but execution finished. Examples
        of this error are NaN squashing or integer overflow. In a dynamic system, the data can be thrown away and the
        next frame can be processed or execution can be retried.
        This is either a data corruption error, an input error, or a range error.


      INVALID_STATE :
        TensorRT was put into a bad state by incorrect sequence of function calls. An example of an invalid state is
        specifying a layer to be DLA only without GPU fallback, and that layer is not supported by DLA. This can occur
        in situations where a service is optimistically executing networks for multiple different configurations
        without checking proper error configurations, and instead throwing away bad configurations caught by TensorRT.
        This is a violation of the API contract, but can be recoverable.

        Example of a recovery:
        GPU fallback is disabled and conv layer with large filter(63x63) is specified to run on DLA. This will fail due
        to DLA not supporting the large kernel size. This can be recovered by either turning on GPU fallback
        or setting the layer to run on the GPU.


      UNSUPPORTED_STATE :
        An error occurred due to the network not being supported on the device due to constraints of the hardware or
        system. An example is running a unsafe layer in a safety certified context, or a resource requirement for the
        current network is greater than the capabilities of the target device. The network is otherwise correct, but
        the network and hardware combination is problematic. This can be recoverable.
        Examples:
        * Scratch space requests larger than available device memory and can be recovered by increasing allowed workspace size.
        * Tensor size exceeds the maximum element count and can be recovered by reducing the maximum batch size.
    """
    FAILED_ALLOCATION: typing.ClassVar[ErrorCodeTRT]  # value = <ErrorCodeTRT.FAILED_ALLOCATION: 5>
    FAILED_COMPUTATION: typing.ClassVar[ErrorCodeTRT]  # value = <ErrorCodeTRT.FAILED_COMPUTATION: 8>
    FAILED_EXECUTION: typing.ClassVar[ErrorCodeTRT]  # value = <ErrorCodeTRT.FAILED_EXECUTION: 7>
    FAILED_INITIALIZATION: typing.ClassVar[ErrorCodeTRT]  # value = <ErrorCodeTRT.FAILED_INITIALIZATION: 6>
    INTERNAL_ERROR: typing.ClassVar[ErrorCodeTRT]  # value = <ErrorCodeTRT.INTERNAL_ERROR: 2>
    INVALID_ARGUMENT: typing.ClassVar[ErrorCodeTRT]  # value = <ErrorCodeTRT.INVALID_ARGUMENT: 3>
    INVALID_CONFIG: typing.ClassVar[ErrorCodeTRT]  # value = <ErrorCodeTRT.INVALID_CONFIG: 4>
    INVALID_STATE: typing.ClassVar[ErrorCodeTRT]  # value = <ErrorCodeTRT.INVALID_STATE: 9>
    SUCCESS: typing.ClassVar[ErrorCodeTRT]  # value = <ErrorCodeTRT.SUCCESS: 0>
    UNSPECIFIED_ERROR: typing.ClassVar[ErrorCodeTRT]  # value = <ErrorCodeTRT.UNSPECIFIED_ERROR: 1>
    UNSUPPORTED_STATE: typing.ClassVar[ErrorCodeTRT]  # value = <ErrorCodeTRT.UNSUPPORTED_STATE: 10>
    __members__: typing.ClassVar[dict[str, ErrorCodeTRT]]  # value = {'SUCCESS': <ErrorCodeTRT.SUCCESS: 0>, 'UNSPECIFIED_ERROR': <ErrorCodeTRT.UNSPECIFIED_ERROR: 1>, 'INTERNAL_ERROR': <ErrorCodeTRT.INTERNAL_ERROR: 2>, 'INVALID_ARGUMENT': <ErrorCodeTRT.INVALID_ARGUMENT: 3>, 'INVALID_CONFIG': <ErrorCodeTRT.INVALID_CONFIG: 4>, 'FAILED_ALLOCATION': <ErrorCodeTRT.FAILED_ALLOCATION: 5>, 'FAILED_INITIALIZATION': <ErrorCodeTRT.FAILED_INITIALIZATION: 6>, 'FAILED_EXECUTION': <ErrorCodeTRT.FAILED_EXECUTION: 7>, 'FAILED_COMPUTATION': <ErrorCodeTRT.FAILED_COMPUTATION: 8>, 'INVALID_STATE': <ErrorCodeTRT.INVALID_STATE: 9>, 'UNSUPPORTED_STATE': <ErrorCodeTRT.UNSUPPORTED_STATE: 10>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ExecutionContextAllocationStrategy:
    """
    Different memory allocation behaviors for IExecutionContext.

    Members:

      STATIC : Default static allocation with the maximum size across all profiles.

      ON_PROFILE_CHANGE : Reallocate for a profile when it's selected.

      USER_MANAGED : The user supplies custom allocation to the execution context.
    """
    ON_PROFILE_CHANGE: typing.ClassVar[ExecutionContextAllocationStrategy]  # value = <ExecutionContextAllocationStrategy.ON_PROFILE_CHANGE: 1>
    STATIC: typing.ClassVar[ExecutionContextAllocationStrategy]  # value = <ExecutionContextAllocationStrategy.STATIC: 0>
    USER_MANAGED: typing.ClassVar[ExecutionContextAllocationStrategy]  # value = <ExecutionContextAllocationStrategy.USER_MANAGED: 2>
    __members__: typing.ClassVar[dict[str, ExecutionContextAllocationStrategy]]  # value = {'STATIC': <ExecutionContextAllocationStrategy.STATIC: 0>, 'ON_PROFILE_CHANGE': <ExecutionContextAllocationStrategy.ON_PROFILE_CHANGE: 1>, 'USER_MANAGED': <ExecutionContextAllocationStrategy.USER_MANAGED: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class FallbackString:
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
class FillOperation:
    """
    The tensor fill operations that may performed by an Fill layer.

    Members:

      LINSPACE : Generate evenly spaced numbers over a specified interval

      RANDOM_UNIFORM : Generate a tensor with random values drawn from a uniform distribution

      RANDOM_NORMAL : Generate a tensor with random values drawn from a normal distribution
    """
    LINSPACE: typing.ClassVar[FillOperation]  # value = <FillOperation.LINSPACE: 0>
    RANDOM_NORMAL: typing.ClassVar[FillOperation]  # value = <FillOperation.RANDOM_NORMAL: 2>
    RANDOM_UNIFORM: typing.ClassVar[FillOperation]  # value = <FillOperation.RANDOM_UNIFORM: 1>
    __members__: typing.ClassVar[dict[str, FillOperation]]  # value = {'LINSPACE': <FillOperation.LINSPACE: 0>, 'RANDOM_UNIFORM': <FillOperation.RANDOM_UNIFORM: 1>, 'RANDOM_NORMAL': <FillOperation.RANDOM_NORMAL: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class GatherMode:
    """
    Controls how IGatherLayer gathers data

    Members:

      DEFAULT : Similar to ONNX Gather. This is the default.

      ELEMENT : Similar to ONNX GatherElements.

      ND : Similar to ONNX GatherND.
    """
    DEFAULT: typing.ClassVar[GatherMode]  # value = <GatherMode.DEFAULT: 0>
    ELEMENT: typing.ClassVar[GatherMode]  # value = <GatherMode.ELEMENT: 1>
    ND: typing.ClassVar[GatherMode]  # value = <GatherMode.ND: 2>
    __members__: typing.ClassVar[dict[str, GatherMode]]  # value = {'DEFAULT': <GatherMode.DEFAULT: 0>, 'ELEMENT': <GatherMode.ELEMENT: 1>, 'ND': <GatherMode.ND: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class HardwareCompatibilityLevel:
    """

        Describes requirements of compatibility with GPU architectures other than that of the GPU on which the engine was built.
        Note that compatibility with future hardware depends on CUDA forward compatibility support.


    Members:

      NONE :
        Do not require hardware compatibility with GPU architectures other than that of the GPU on which the engine was built.


      AMPERE_PLUS :
        Require that the engine is compatible with Ampere and newer GPUs. This will limit the combined usage of driver reserved and backend kernel max shared memory to
        48KiB, may reduce the number of available tactics for each layer, and may prevent some fusions from occurring.
        Thus this can decrease the performance, especially for tf32 models.
        This option will disable cuDNN, cuBLAS, and cuBLASLt as tactic sources.
        This option is only supported for engines built on NVIDIA Ampere and later GPUs.


      SAME_COMPUTE_CAPABILITY :
        Require that the engine is compatible with GPUs that have the same Compute Capability version as the one it was built on.
        This may decrease the performance compared to an engine with no compatibility.
        This option will disable cuDNN, cuBLAS, and cuBLASLt as tactic sources.
        This option is only supported for engines built on NVIDIA Turing and later GPUs.
    """
    AMPERE_PLUS: typing.ClassVar[HardwareCompatibilityLevel]  # value = <HardwareCompatibilityLevel.AMPERE_PLUS: 1>
    NONE: typing.ClassVar[HardwareCompatibilityLevel]  # value = <HardwareCompatibilityLevel.NONE: 0>
    SAME_COMPUTE_CAPABILITY: typing.ClassVar[HardwareCompatibilityLevel]  # value = <HardwareCompatibilityLevel.SAME_COMPUTE_CAPABILITY: 2>
    __members__: typing.ClassVar[dict[str, HardwareCompatibilityLevel]]  # value = {'NONE': <HardwareCompatibilityLevel.NONE: 0>, 'AMPERE_PLUS': <HardwareCompatibilityLevel.AMPERE_PLUS: 1>, 'SAME_COMPUTE_CAPABILITY': <HardwareCompatibilityLevel.SAME_COMPUTE_CAPABILITY: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class IActivationLayer(ILayer):
    """

        An Activation layer in an :class:`INetworkDefinition` . This layer applies a per-element activation function to its input. The output has the same shape as the input.

        :ivar type: :class:`ActivationType` The type of activation to be performed.
        :ivar alpha: :class:`float` The alpha parameter that is used by some parametric activations (LEAKY_RELU, ELU, SELU, SOFTPLUS, CLIP, HARD_SIGMOID, SCALED_TANH). Other activations ignore this parameter.
        :ivar beta: :class:`float` The beta parameter that is used by some parametric activations (SELU, SOFTPLUS, CLIP, HARD_SIGMOID, SCALED_TANH). Other activations ignore this parameter.
    """
    type: ActivationType
    @property
    def alpha(self) -> float:
        ...
    @alpha.setter
    def alpha(self, arg1: typing.SupportsFloat) -> None:
        ...
    @property
    def beta(self) -> float:
        ...
    @beta.setter
    def beta(self, arg1: typing.SupportsFloat) -> None:
        ...
class IAlgorithm:
    """

            [DEPRECATED] Deprecated in TensorRT 10.8. Please use editable mode in ITimingCache instead.
            Application-implemented interface for selecting and reporting the tactic selection of a layer.
            Tactic Selection is a step performed by the builder for deciding best algorithms for a layer.

        :ivar algorithm_variant: :class:`IAlgorithmVariant&`  the algorithm variant.
        :ivar timing_msec: :class:`float` The time in milliseconds to execute the algorithm.
        :ivar workspace_size: :class:`int` The size of the GPU temporary memory in bytes which the algorithm uses at execution time.
    """
    def get_algorithm_io_info(self, index: typing.SupportsInt) -> IAlgorithmIOInfo:
        """
            [DEPRECATED] Deprecated in TensorRT 10.8. Please use editable mode in ITimingCache instead.
            A single call for both inputs and outputs. Incremental numbers assigned to indices of inputs and the outputs.

            :arg index: Index of the input or output of the algorithm. Incremental numbers assigned to indices of inputs and the outputs.

            :returns: A :class:`IAlgorithmIOInfo&`
        """
    @property
    def algorithm_variant(self) -> IAlgorithmVariant:
        ...
    @property
    def timing_msec(self) -> float:
        ...
    @property
    def workspace_size(self) -> int:
        ...
class IAlgorithmContext:
    """

        [DEPRECATED] Deprecated in TensorRT 10.8. Please use editable mode in ITimingCache instead.
        Describes the context and requirements, that could be fulfilled by one or
        more instances of IAlgorithm.
        see IAlgorithm

        :ivar name: :class:`str` name of the algorithm node.
        :ivar num_inputs: :class:`int`  number of inputs of the algorithm.
        :ivar num_outputs: :class:`int` number of outputs of the algorithm.
    """
    def get_shape(self, index: typing.SupportsInt) -> list[Dims]:
        """
            [DEPRECATED] Deprecated in TensorRT 10.8. Please use editable mode in ITimingCache instead.
            Get the minimum / optimum / maximum dimensions for a dynamic input tensor.

            :arg index: Index of the input or output of the algorithm. Incremental numbers assigned to indices of inputs and the outputs.

            :returns: A `List[Dims]` of length 3, containing the minimum, optimum, and maximum shapes, in that order. If the shapes have not been set yet, an empty list is returned.`
        """
    @property
    def name(self) -> str:
        ...
    @property
    def num_inputs(self) -> int:
        ...
    @property
    def num_outputs(self) -> int:
        ...
class IAlgorithmIOInfo:
    """

        [DEPRECATED] Deprecated in TensorRT 10.8. Please use editable mode in ITimingCache instead.
        This class carries information about input or output of the algorithm.
        IAlgorithmIOInfo for all the input and output along with IAlgorithmVariant denotes the variation of algorithm
        and can be used to select or reproduce an algorithm using IAlgorithmSelector.select_algorithms().

        :ivar dtype: :class:`DataType`  DataType of the input/output of algorithm.
        :ivar strides: :class:`Dims` strides of the input/output tensor of algorithm.
        :ivar vectorized_dim: :class:`int` the index of the vectorized dimension or -1 for non-vectorized formats.
        :ivar components_per_element: :class:`int` the number of components per element. This is always 1 for non-vectorized formats.
    """
    @property
    def components_per_element(self) -> int:
        ...
    @property
    def dtype(self) -> DataType:
        ...
    @property
    def strides(self) -> Dims:
        ...
    @property
    def vectorized_dim(self) -> int:
        ...
class IAlgorithmSelector:
    """

        [DEPRECATED] Deprecated in TensorRT 10.8. Please use editable mode in ITimingCache instead.
        Interface implemented by application for selecting and reporting algorithms of a layer provided by the
        builder.
        note A layer in context of algorithm selection may be different from ILayer in INetworkDefiniton.
        For example, an algorithm might be implementing a conglomeration of multiple ILayers in INetworkDefinition.

        To implement a custom algorithm selector, ensure that you explicitly instantiate the base class in :func:`__init__` :
        ::

            class MyAlgoSelector(trt.IAlgorithmSelector):
                def __init__(self):
                    trt.IAlgorithmSelector.__init__(self)

    """
    def __init__(self) -> None:
        ...
    def report_algorithms(self, contexts: collections.abc.Sequence[IAlgorithmContext], choices: collections.abc.Sequence[IAlgorithm]) -> None:
        """
            [DEPRECATED] Deprecated in TensorRT 10.8. Please use editable mode in ITimingCache instead.
            Called by TensorRT to report choices it made.

            Note: For a given optimization profile, this call comes after all calls to select_algorithms.
            choices[i] is the choice that TensorRT made for algoContexts[i], for i in [0, num_algorithms-1]

            For example, a possible implementation may look like this:
            ::

                def report_algorithms(self, contexts, choices):
                    # Prints the time of the chosen algorithm by TRT from the
                    # selection list passed in by select_algorithms
                    for choice in choices:
                        print(choice.timing_msec)

            :arg contexts: The list of all algorithm contexts.
            :arg choices: The list of algorithm choices made by TensorRT corresponding to each context.
        """
    def select_algorithms(self, context: IAlgorithmContext, choices: collections.abc.Sequence[IAlgorithm]) -> list[int]:
        """
            [DEPRECATED] Deprecated in TensorRT 10.8. Please use editable mode in ITimingCache instead.
            Select Algorithms for a layer from the given list of algorithm choices.

            Note: TRT uses its default algorithm selection to choose from the list returned by the user.
            If the returned list is empty, TRT’s default algorithm selection is used unless strict type constraints are set.
            The list of choices is valid only for this specific algorithm context.

            For example, the simplest implementation looks like this:
            ::

                def select_algorithms(self, context, choices):
                    assert len(choices) > 0
                    return list(range(len(choices)))

            :arg context: The context for which the algorithm choices are valid.
            :arg choices: The list of algorithm choices to select for implementation of this layer.

            :returns: A :class:`List[int]` indicating the indices from the choices vector that TensorRT should choose from.
        """
class IAlgorithmVariant:
    """

        [DEPRECATED] Deprecated in TensorRT 10.8. Please use editable mode in ITimingCache instead.
        provides a unique 128-bit identifier, which along with the input and output information
        denotes the variation of algorithm and can be used to select or reproduce an algorithm,
        using IAlgorithmSelector.select_algorithms()
        see IAlgorithmIOInfo, IAlgorithm, IAlgorithmSelector.select_algorithms()
        note A single implementation can have multiple tactics.

        :ivar implementation: :class:`int` implementation of the algorithm.
        :ivar tactic: :class:`int`  tactic of the algorithm.
    """
    @property
    def implementation(self) -> int:
        ...
    @property
    def tactic(self) -> int:
        ...
class IAssertionLayer(ILayer):
    """

        An assertion layer in an :class:`INetworkDefinition` .

        This layer implements assertions. The input must be a boolean shape tensor. If any element of it is ``False``, a build-time or run-time error occurs. Asserting equality of input dimensions may help the optimizer.

        :ivar message: :class:`string` Message to print if the assertion fails.
    """
    message: str
class IAttention:
    """

        An attention in a :class:`INetworkDefinition` .

        :ivar mask: :class:`ITensor` The mask tensor for attention. Cannot be set together with causal attention.
        :ivar norm_op: :class:`AttentionNormalizationOp` The normalization operation for the attention layer. Default to AttentionNormalizationOp::kSOFTMAX.
        :ivar decomposable: :class:`bool` Specifies whether decomposition into primitive ops is allowed when no attention fusion is supported. Default to False.
        :ivar causal: :class:`bool` Specifies whether the attention will run a causal inference. Cannot be used together with mask.
        :ivar name: :class:`str` The name of the attention.
        :ivar metadata: :class:`str` The metadata of the attention.
        :ivar normalization_quantize_scale: :class:`ITensor` The quantization scale for the attention normalization output.
        :ivar normalization_quantize_to_type: :class:`DataType` The datatype the attention normalization is quantized to.
        :ivar num_inputs: :class:`int` The number of inputs of the attention.
        :ivar num_outputs: :class:`int` The number of outputs of the attention.
        :ivar num_ranks: :class:`int` The number of ranks for multi-device attention execution (default: 1).
    """
    causal: bool
    decomposable: bool
    mask: ITensor
    metadata: str
    name: str
    norm_op: AttentionNormalizationOp
    normalization_quantize_scale: ITensor
    normalization_quantize_to_type: DataType
    def get_input(self, index: typing.SupportsInt) -> ITensor:
        """
            Get the input tensor specified by the given index.

            :arg index: The index of the input tensor.

            :returns: The tensor, or :class:`None` if it is out of range.
        """
    def get_output(self, index: typing.SupportsInt) -> ITensor:
        """
            Get the output tensor specified by the given index.

            :arg index: The index of the output tensor.

            :returns: The tensor, or :class:`None` if it is out of range.
        """
    def set_input(self, index: typing.SupportsInt, tensor: ITensor) -> bool:
        """
            Set the input tensor specified by the given index.

            The indices are as follows:

            =====   ==================================================================================
            Index   Description
            =====   ==================================================================================
                0     query.
                1     key.
                2     value.
            =====   ==================================================================================

            :arg index: The index of the input tensor. query:0, key:1, value:2
            :arg tensor: The input tensor.
        """
    @property
    def num_inputs(self) -> int:
        ...
    @property
    def num_outputs(self) -> int:
        ...
    @property
    def num_ranks(self) -> int:
        """
            :class:`int` The number of ranks for multi-device attention execution.

            When num_ranks > 1, this hints attention to perform multi-device attention.

            Default value is 1.
        """
    @num_ranks.setter
    def num_ranks(self, arg1: typing.SupportsInt) -> None:
        ...
class IAttentionBoundaryLayer(ILayer):
    """

        :ivar attention: :class:`IAttention` associated with this boundary layer.
    """
    @property
    def attention(self) -> IAttention:
        ...
class IAttentionInputLayer(IAttentionBoundaryLayer):
    """

        Marks input boundary to an :class:`IAttention` scope
    """
class IAttentionOutputLayer(IAttentionBoundaryLayer):
    """

        Marks output boundary to an :class:`IAttention` scope
    """
class IBuilderConfig:
    """


            :ivar avg_timing_iterations: :class:`int` The number of averaging iterations used when timing layers. When timing layers, the builder minimizes over a set of average times for layer execution. This parameter controls the number of iterations used in averaging. By default the number of averaging iterations is 1.
            :ivar int8_calibrator: :class:`IInt8Calibrator` [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization. Int8 Calibration interface. The calibrator is to minimize the information loss during the INT8 quantization process.
            :ivar flags: :class:`int` The build mode flags to turn on builder options for this network. The flags are listed in the BuilderFlags enum. The flags set configuration options to build the network. This should be in integer consisting of one or more :class:`BuilderFlag` s, combined via binary OR. For example, ``1 << BuilderFlag.FP16 | 1 << BuilderFlag.DEBUG``.
            :ivar profile_stream: :class:`int` The handle for the CUDA stream that is used to profile this network.
            :ivar num_optimization_profiles: :class:`int` The number of optimization profiles.
            :ivar default_device_type: :class:`tensorrt.DeviceType` The default DeviceType to be used by the Builder.
            :ivar DLA_core: :class:`int` The DLA core that the engine executes on. Must be between 0 and N-1 where N is the number of available DLA cores.
            :ivar profiling_verbosity: Profiling verbosity in NVTX annotations.
            :ivar engine_capability: The desired engine capability. See :class:`EngineCapability` for details.
            :ivar algorithm_selector: [DEPRECATED] Deprecated in TensorRT 10.8. Please use editable mode in ITimingCache instead. The :class:`IAlgorithmSelector` to use.
            :ivar builder_optimization_level: The builder optimization level which TensorRT should build the engine at. Setting a higher optimization level allows TensorRT to spend longer engine building time searching for more optimization options. The resulting engine may have better performance compared to an engine built with a lower optimization level. The default optimization level is 3. Valid values include integers from 0 to the maximum optimization level, which is currently 5. Setting it to be greater than the maximum level results in identical behavior to the maximum level.
            :ivar max_num_tactics: The maximum number of tactics to time when there is a choice of tactics. Setting a larger number allows TensorRT to spend longer engine building time searching for more optimization options. The resulting engine may have better performance compared to an engine built with a smaller number of tactics. Valid values include integers from -1 to the maximum 32-bit integer. Default value -1 indicates that TensorRT can decide the number of tactics based on its own heuristic.
            :ivar hardware_compatibility_level: Hardware compatibility allows an engine compatible with GPU architectures other than that of the GPU on which the engine was built.
            :ivar plugins_to_serialize: The plugin libraries to be serialized with forward-compatible engines.
            :ivar max_aux_streams: The maximum number of auxiliary streams that TRT is allowed to use. If the network contains operators that can run in parallel, TRT can execute them using auxiliary streams in addition to the one provided to the IExecutionContext::enqueueV3() call. The default maximum number of auxiliary streams is determined by the heuristics in TensorRT on whether enabling multi-stream would improve the performance. This behavior can be overridden by calling this API to set the maximum number of auxiliary streams explicitly. Set this to 0 to enforce single-stream inference. The resulting engine may use fewer auxiliary streams than the maximum if the network does not contain enough parallelism or if TensorRT determines that using more auxiliary streams does not help improve the performance. Allowing more auxiliary streams does not always give better performance since there will be synchronizations overhead between streams. Using CUDA graphs at runtime can help reduce the overhead caused by cross-stream synchronizations. Using more auxiliary leads to more memory usage at runtime since some activation memory blocks will not be able to be reused.
            :ivar progress_monitor: The :class:`IProgressMonitor` to use.
            :ivar tiling_optimization_level: The optimization level of tiling strategies. A Higher level allows TensorRT to spend more time searching for better optimization strategy.
            :ivar l2_limit_for_tiling: The target L2 cache usage for tiling optimization.
            :ivar remote_auto_tuning_config: The config string to be used during remote auto-tuning. Remote auto-tuning is only enabled for engines built with EngineCapability.SAFETY.

            Below are the descriptions about each builder optimization level:

            - Level 0: This enables the fastest compilation by disabling dynamic kernel generation and selecting the first tactic that succeeds in execution. This will also not respect a timing cache.
            - Level 1: Available tactics are sorted by heuristics, but only the top are tested to select the best. If a dynamic kernel is generated its compile optimization is low.
            - Level 2: Available tactics are sorted by heuristics, but only the fastest tactics are tested to select the best.
            - Level 3: Apply heuristics to see if a static precompiled kernel is applicable or if a new one has to be compiled dynamically.
            - Level 4: Always compiles a dynamic kernel.
            - Level 5: Always compiles a dynamic kernel and compares it to static kernels.

    """
    algorithm_selector: IAlgorithmSelector
    default_device_type: DeviceType
    engine_capability: EngineCapability
    hardware_compatibility_level: HardwareCompatibilityLevel
    int8_calibrator: IInt8Calibrator
    profiling_verbosity: ProfilingVerbosity
    progress_monitor: IProgressMonitor
    remote_auto_tuning_config: str
    runtime_platform: RuntimePlatform
    tiling_optimization_level: TilingOptimizationLevel
    @staticmethod
    def __enter__(this):
        ...
    @staticmethod
    def __exit__(this, exc_type, exc_value, traceback):
        """

            Context managers are deprecated and have no effect. Objects are automatically freed when
            the reference count reaches 0.

        """
    def __del__(self) -> None:
        ...
    def add_optimization_profile(self, profile: IOptimizationProfile) -> int:
        """
            Add an optimization profile.

            This function must be called at least once if the network has dynamic or shape input tensors.

            :arg profile: The new optimization profile, which must satisfy ``bool(profile) == True``

            :returns: The index of the optimization profile (starting from 0) if the input is valid, or -1 if the input is
                        not valid.
        """
    def can_run_on_DLA(self, layer: ILayer) -> bool:
        """
            Check if the layer can run on DLA.

            :arg layer: The layer to check

            :returns: A `bool` indicating whether the layer can run on DLA
        """
    def clear_flag(self, flag: BuilderFlag) -> None:
        """
                Clears the builder mode flag from the enabled flags.

                :arg flag: The flag to clear.
        """
    def clear_quantization_flag(self, flag: QuantizationFlag) -> None:
        """
                Clears the quantization flag from the enabled quantization flags.

                :arg flag: The flag to clear.
        """
    def create_timing_cache(self, serialized_timing_cache: collections.abc.Buffer) -> ITimingCache:
        """
            Create timing cache

            Create :class:`ITimingCache` instance from serialized raw data. The created timing cache doesn't belong to
            a specific builder config. It can be shared by multiple builder instances

            :arg serialized_timing_cache: The serialized timing cache. If an empty cache is provided (i.e. ``b""``),  a new cache will be created.

            :returns: The created :class:`ITimingCache` object.
        """
    def get_calibration_profile(self) -> IOptimizationProfile:
        """
            [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization.

            Get the current calibration profile.

            :returns: The current calibration profile or None if calibrartion profile is unset.
        """
    def get_device_type(self, layer: ILayer) -> DeviceType:
        """
            Get the device that the layer executes on.

            :arg layer: The layer to get the DeviceType for

            :returns: The DeviceType of the layer
        """
    def get_flag(self, flag: BuilderFlag) -> bool:
        """
                Check if a build mode flag is set.

                :arg flag: The flag to check.

                :returns: A `bool` indicating whether the flag is set.
        """
    def get_memory_pool_limit(self, pool: MemoryPoolType) -> int:
        """
                Retrieve the memory size limit of the corresponding pool in bytes.
                If :func:`set_memory_pool_limit` for the pool has not been called, this returns the default value used by TensorRT.
                This default value is not necessarily the maximum possible value for that configuration.

                :arg pool: The memory pool to get the limit for.

                :returns: The size of the memory limit, in bytes, for the corresponding pool.
        """
    def get_preview_feature(self, feature: PreviewFeature) -> bool:
        """
            Check if a preview feature is enabled.

            :arg feature: the feature to query

            :returns: true if the feature is enabled, false otherwise
        """
    def get_quantization_flag(self, flag: QuantizationFlag) -> bool:
        """
                Check if a quantization flag is set.

                :arg flag: The flag to check.

                :returns: A `bool` indicating whether the flag is set.
        """
    def get_tactic_sources(self) -> int:
        """
            Get the tactic sources currently set in the engine build configuration.
        """
    def get_timing_cache(self) -> ITimingCache:
        """
            Get the timing cache from current IBuilderConfig

            :returns: The timing cache used in current IBuilderConfig, or `None` if no timing cache is set.
        """
    def is_device_type_set(self, layer: ILayer) -> bool:
        """
            Check if the DeviceType for a layer is explicitly set.

            :arg layer: The layer to check for DeviceType

            :returns: True if DeviceType is not default, False otherwise
        """
    def reset(self) -> None:
        """
                Resets the builder configuration to defaults. When initializing a builder config object, we can call this function.
        """
    def reset_device_type(self, layer: ILayer) -> None:
        """
            Reset the DeviceType for the given layer.

            :arg layer: The layer to reset the DeviceType for
        """
    def set_calibration_profile(self, profile: IOptimizationProfile) -> bool:
        """
            [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization.

            Set a calibration profile.

            Calibration optimization profile must be set if int8 calibration is used to set scales for a network with runtime dimensions.

            :arg profile: The new calibration profile, which must satisfy ``bool(profile) == True`` or be None. MIN and MAX values will be overwritten by OPT.

            :returns: True if the calibration profile was set correctly.
        """
    def set_device_type(self, layer: ILayer, device_type: DeviceType) -> None:
        """
            Set the device that this layer must execute on. If DeviceType is not set or is reset, TensorRT will use the
            default DeviceType set in the builder.

            The DeviceType for a layer must be compatible with the safety flow (if specified). For example a layer
            cannot be marked for DLA execution while the builder is configured for SAFE_GPU.


            :arg layer: The layer to set the DeviceType of
            :arg device_type: The DeviceType the layer must execute on
        """
    def set_flag(self, flag: BuilderFlag) -> None:
        """
                Add the input builder mode flag to the already enabled flags.

                :arg flag: The flag to set.
        """
    def set_memory_pool_limit(self, pool: MemoryPoolType, pool_size: typing.SupportsInt) -> None:
        """
                Set the memory size for the memory pool.

                TensorRT layers access different memory pools depending on the operation.
                This function sets in the :class:`IBuilderConfig` the size limit, specified by pool_size, for the corresponding memory pool, specified by pool.
                TensorRT will build a plan file that is constrained by these limits or report which constraint caused the failure.

                If the size of the pool, specified by pool_size, fails to meet the size requirements for the pool,
                this function does nothing and emits the recoverable error, ErrorCode.INVALID_ARGUMENT, to the registered :class:`IErrorRecorder` .

                If the size of the pool is larger than the maximum possible value for the configuration,
                this function does nothing and emits ErrorCode.UNSUPPORTED_STATE.

                If the pool does not exist on the requested device type when building the network,
                a warning is emitted to the logger, and the memory pool value is ignored.

                Refer to MemoryPoolType to see the size requirements for each pool.

                :arg pool: The memory pool to limit the available memory for.
                :arg pool_size: The size of the pool in bytes.
        """
    def set_preview_feature(self, feature: PreviewFeature, enable: bool) -> None:
        """
            Enable or disable a specific preview feature.

            Allows enabling or disabling experimental features, which are not enabled by default in the current release.
            Preview Features have been fully tested but are not yet as stable as other features in TensorRT.
            They are provided as opt-in features for at least one release.

            Refer to PreviewFeature for additional information, and a list of the available features.

            :arg feature: the feature to enable
            :arg enable: whether to enable or disable
        """
    def set_quantization_flag(self, flag: QuantizationFlag) -> None:
        """
                Add the input quantization flag to the already enabled quantization flags.

                :arg flag: The flag to set.
        """
    def set_tactic_sources(self, tactic_sources: typing.SupportsInt) -> bool:
        """
            Set tactic sources.

            This bitset controls which tactic sources TensorRT is allowed to use for tactic selection.

            Multiple tactic sources may be combined with a bitwise OR operation. For example,
            to enable cublas and cublasLt as tactic sources, use a value of:
            ``1 << int(trt.TacticSource.CUBLAS) | 1 << int(trt.TacticSource.CUBLAS_LT)``

            :arg tactic_sources: The tactic sources to set

            :returns: A `bool` indicating whether the tactic sources in the build configuration were updated. The tactic sources in the build configuration will not be updated if the provided value is invalid.
        """
    def set_timing_cache(self, cache: ITimingCache, ignore_mismatch: bool) -> bool:
        """
            Attach a timing cache to IBuilderConfig

            The timing cache has verification header to make sure the provided cache can be used in current environment.
            A failure will be reported if the CUDA device property in the provided cache is different from current environment.
            ``bool(ignore_mismatch) == True`` skips strict verification and allows loading cache created from a different device.
            The cache must not be destroyed until after the engine is built.

            :arg cache: The timing cache to be used
            :arg ignore_mismatch: Whether or not allow using a cache that contains different CUDA device property

            :returns: A `BOOL` indicating whether the operation is done successfully.
        """
    @property
    def DLA_core(self) -> int:
        ...
    @DLA_core.setter
    def DLA_core(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def avg_timing_iterations(self) -> int:
        ...
    @avg_timing_iterations.setter
    def avg_timing_iterations(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def builder_optimization_level(self) -> int:
        ...
    @builder_optimization_level.setter
    def builder_optimization_level(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def flags(self) -> int:
        ...
    @flags.setter
    def flags(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def l2_limit_for_tiling(self) -> int:
        ...
    @l2_limit_for_tiling.setter
    def l2_limit_for_tiling(self, arg1: typing.SupportsInt) -> bool:
        ...
    @property
    def max_aux_streams(self) -> int:
        ...
    @max_aux_streams.setter
    def max_aux_streams(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def max_num_tactics(self) -> int:
        ...
    @max_num_tactics.setter
    def max_num_tactics(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def num_optimization_profiles(self) -> int:
        ...
    @property
    def plugins_to_serialize(self) -> list[str]:
        ...
    @plugins_to_serialize.setter
    def plugins_to_serialize(self, arg1: collections.abc.Sequence[str]) -> None:
        ...
    @property
    def profile_stream(self) -> int:
        ...
    @profile_stream.setter
    def profile_stream(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def quantization_flags(self) -> int:
        ...
    @quantization_flags.setter
    def quantization_flags(self, arg1: typing.SupportsInt) -> None:
        ...
class ICastLayer(ILayer):
    """

        A layer that represents the cast function.

        This layer casts the element of a given input tensor to a specified data type and returns an output tensor of the same shape in the converted type.

        Conversions between all types except FP8 is supported.

        :ivar to_type: :class:`DataType` The specified data type of the output tensor.
    """
    to_type: DataType
class IConcatenationLayer(ILayer):
    """

        A concatenation layer in an :class:`INetworkDefinition` .

        The output channel size is the sum of the channel sizes of the inputs.
        The other output sizes are the same as the other input sizes, which must all match.

        :ivar axis: :class:`int` The axis along which concatenation occurs. The default axis is the number of tensor dimensions minus three, or zero if the tensor has fewer than three dimensions. For example, for a tensor with dimensions NCHW, it is C.
    """
    @property
    def axis(self) -> int:
        ...
    @axis.setter
    def axis(self, arg1: typing.SupportsInt) -> None:
        ...
class IConditionLayer(IIfConditionalBoundaryLayer):
    """
    Describes the boolean condition of an if-conditional.
    """
class IConstantLayer(ILayer):
    """

        A constant layer in an :class:`INetworkDefinition` .

        Note: This layer does not support boolean and uint8 types.

        :ivar weights: :class:`Weights` The weights for the layer.
        :ivar shape: :class:`Dims` The shape of the layer.
    """
    shape: Dims
    @property
    def weights(self) -> typing.Any:
        ...
    @weights.setter
    def weights(self, arg1: Weights) -> None:
        ...
class IConvolutionLayer(ILayer):
    """

        A convolution layer in an :class:`INetworkDefinition` .

        This layer performs a correlation operation between 3 or 4 dimensional filter with a 4 or 5 dimensional tensor to produce another 4 or 5 dimensional tensor.

        An optional bias argument is supported, which adds a per-channel constant to each value in the output.

        :ivar num_output_maps: :class:`int` The number of output maps for the convolution.
        :ivar pre_padding: :class:`DimsHW` The pre-padding. The start of input will be zero-padded by this number of elements in the height and width directions. Default: (0, 0)
        :ivar post_padding: :class:`DimsHW` The post-padding. The end of input will be zero-padded by this number of elements in the height and width directions. Default: (0, 0)
        :ivar padding_mode: :class:`PaddingMode` The padding mode. Padding mode takes precedence if both :attr:`IConvolutionLayer.padding_mode` and either :attr:`IConvolutionLayer.pre_padding` or :attr:`IConvolutionLayer.post_padding` are set.
        :ivar num_groups: :class:`int` The number of groups for a convolution. The input tensor channels are divided into this many groups, and a convolution is executed for each group, using a filter per group. The results of the group convolutions are concatenated to form the output. **Note** When using groups in int8 mode, the size of the groups (i.e. the channel count divided by the group count) must be a multiple of 4 for both input and output. Default: 1.
        :ivar kernel: :class:`Weights` The kernel weights for the convolution. The weights are specified as a contiguous array in `GKCRS` order, where `G` is the number of groups, `K` the number of output feature maps, `C` the number of input channels, and `R` and `S` are the height and width of the filter.
        :ivar bias: :class:`Weights` The bias weights for the convolution. Bias is optional. To omit bias, set this to an empty :class:`Weights` object. The bias is applied per-channel, so the number of weights (if non-zero) must be equal to the number of output feature maps.
        :ivar kernel_size_nd: :class:`Dims` The multi-dimension kernel size of the convolution.
        :ivar stride_nd: :class:`Dims` The multi-dimension stride of the convolution. Default: (1, ..., 1)
        :ivar padding_nd: :class:`Dims` The multi-dimension padding of the convolution. The input will be zero-padded by this number of elements in each dimension. If the padding is asymmetric, this value corresponds to the pre-padding. Default: (0, ..., 0)
        :ivar dilation_nd: :class:`Dims` The multi-dimension dilation for the convolution. Default: (1, ..., 1)
    """
    dilation_nd: Dims
    kernel_size_nd: Dims
    padding_mode: PaddingMode
    padding_nd: Dims
    post_padding: Dims
    pre_padding: Dims
    stride_nd: Dims
    @property
    def bias(self) -> typing.Any:
        ...
    @bias.setter
    def bias(self, arg1: Weights) -> None:
        ...
    @property
    def kernel(self) -> typing.Any:
        ...
    @kernel.setter
    def kernel(self, arg1: Weights) -> None:
        ...
    @property
    def num_groups(self) -> int:
        ...
    @num_groups.setter
    def num_groups(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def num_output_maps(self) -> int:
        ...
    @num_output_maps.setter
    def num_output_maps(self, arg1: typing.SupportsInt) -> None:
        ...
class ICudaEngine:
    """

        An :class:`ICudaEngine` for executing inference on a built network.

        The engine can be indexed with ``[]`` . When indexed in this way with an integer, it will return the corresponding binding name. When indexed with a string, it will return the corresponding binding index.

        :ivar num_io_tensors: :class:`int` The number of IO tensors.
        :ivar has_implicit_batch_dimension: :class:`bool` [DEPRECATED] Deprecated in TensorRT 10.0. Always flase since the implicit batch dimensions support has been removed.
        :ivar num_layers: :class:`int` The number of layers in the network. The number of layers in the network is not necessarily the number in the original :class:`INetworkDefinition`, as layers may be combined or eliminated as the :class:`ICudaEngine` is optimized. This value can be useful when building per-layer tables, such as when aggregating profiling data over a number of executions.
        :ivar max_workspace_size: :class:`int` The amount of workspace the :class:`ICudaEngine` uses. The workspace size will be no greater than the value provided to the :class:`Builder` when the :class:`ICudaEngine` was built, and will typically be smaller. Workspace will be allocated for each :class:`IExecutionContext` .
        :ivar device_memory_size: :class:`int` The amount of device memory required by an :class:`IExecutionContext` .
        :ivar device_memory_size_v2: :class:`int` The amount of device memory required by an :class:`IExecutionContext`. The return value depends on the weight streaming budget if enabled.
        :ivar refittable: :class:`bool` Whether the engine can be refit.
        :ivar name: :class:`str` The name of the network associated with the engine. The name is set during network creation and is retrieved after building or deserialization.
        :ivar num_optimization_profiles: :class:`int` The number of optimization profiles defined for this engine. This is always at least 1.
        :ivar error_recorder: :class:`IErrorRecorder` Application-implemented error reporting interface for TensorRT objects.
        :ivar engine_capability: :class:`EngineCapability` The engine capability. See :class:`EngineCapability` for details.
        :ivar tactic_sources: :class:`int` The tactic sources required by this engine.
        :ivar profiling_verbosity: The profiling verbosity the builder config was set to when the engine was built.
        :ivar hardware_compatibility_level: The hardware compatibility level of the engine.
        :ivar num_aux_streams: Read-only. The number of auxiliary streams used by this engine, which will be less than or equal to the maximum allowed number of auxiliary streams by setting builder_config.max_aux_streams when the engine is built.
        :ivar weight_streaming_budget: [DEPRECATED] Deprecated in TensorRT 10.1, superceded by weight_streaming_budget_v2. Set and get the current weight streaming budget for inference. The budget may be set to -1 disabling weight streaming at runtime, 0 (default) enabling TRT to choose to weight stream or not, or a positive value in the inclusive range [minimum_weight_streaming_budget, streamable_weights_size - 1].
        :ivar minimum_weight_streaming_budget: [DEPRECATED] Deprecated in TensorRT 10.1, superceded by weight_streaming_budget_v2. Returns the minimum weight streaming budget in bytes required to run the network successfully. The engine must have been built with kWEIGHT_STREAMING.
        :ivar streamable_weights_size: Returns the size of the streamable weights in the engine. This may not include all the weights.
        :ivar weight_streaming_budget_v2: Set and get the current weight streaming budget for inference. The budget may be set any non-negative value. A value of 0 streams the most weights. Values equal to streamable_weights_size (default) or larger will disable weight streaming.
        :ivar weight_streaming_scratch_memory_size: The amount of scratch memory required by a TensorRT ExecutionContext to perform inference. This value may change based on the current weight streaming budget. Please use the V2 memory APIs, engine.device_memory_size_v2 and ExecutionContext.set_device_memory() to provide memory which includes the current weight streaming scratch memory. Not specifying these APIs or using the V1 APIs will not include this memory, so TensorRT will resort to allocating itself.

    """
    error_recorder: IErrorRecorder
    @staticmethod
    def __enter__(this):
        ...
    @staticmethod
    def __exit__(this, exc_type, exc_value, traceback):
        """

            Context managers are deprecated and have no effect. Objects are automatically freed when
            the reference count reaches 0.

        """
    def __del__(self) -> None:
        ...
    def __getitem__(self, arg0: typing.SupportsInt) -> str:
        ...
    def create_engine_inspector(self) -> EngineInspector:
        """
            Create an :class:`IEngineInspector` which prints out the layer information of an engine or an execution context.

            :returns: The :class:`IEngineInspector`.
        """
    @typing.overload
    def create_execution_context(self, strategy: ExecutionContextAllocationStrategy = ...) -> IExecutionContext:
        """
            Create an :class:`IExecutionContext` and specify the device memory allocation strategy.

            :returns: The newly created :class:`IExecutionContext` .
        """
    @typing.overload
    def create_execution_context(self, runtime_config: IRuntimeConfig = None) -> IExecutionContext:
        """
            Create an :class:`IExecutionContext` and specify the device memory allocation strategy.

            :returns: The newly created :class:`IExecutionContext` .
        """
    def create_execution_context_without_device_memory(self) -> IExecutionContext:
        """
            Create an :class:`IExecutionContext` without any device memory allocated
            The memory for execution of this device context must be supplied by the application.

            :returns: An :class:`IExecutionContext` without device memory allocated.
        """
    def create_runtime_config(self) -> IRuntimeConfig:
        """
            Create a runtime configuration.

            :returns: The newly created :class:`IRuntimeConfig` .
        """
    def create_serialization_config(self) -> ISerializationConfig:
        """
            Create a serialization configuration object.
        """
    def get_aliased_input_tensor(self, name: str) -> str:
        """
            Determine if an output tensor is aliased with an input tensor.

            For certain operations like KVCacheUpdate, the output tensor may share the same memory
            buffer as the input tensor to avoid unnecessary copies. This method returns the name
            of the input tensor that shares memory with the given output tensor name. If the given
            tensor name is not an output tensor or the output does not alias with any input, ``None`` should be returned.

            :arg name: The output tensor name.
            :returns: The name of the input tensor that is aliased with this output.
        """
    def get_device_memory_size_for_profile(self, profile_index: typing.SupportsInt) -> int:
        """
            Return the device memory size required for a certain profile.

            :arg profile_index: The index of the profile.
        """
    def get_device_memory_size_for_profile_v2(self, profile_index: typing.SupportsInt) -> int:
        """
            Return the device memory size required for a certain profile.

            The return value will change depending on the following API calls
            1. setWeightStreamingBudgetV2

            :arg profile_index: The index of the profile.
        """
    def get_engine_stat(self, stat: EngineStat = ...) -> int:
        """
            Return the engine statistics specified by the given enum value.
            If STRIPPED_WEIGHTS_SIZE is passed to query a normal engine, this function will
            return -1 to indicate invalid enum value.

            :arg stat: The engine statistic kind to get.
        """
    @typing.overload
    def get_tensor_bytes_per_component(self, name: str) -> int:
        """
            Return the number of bytes per component of an element.

            The vector component size is returned if :func:`get_tensor_vectorized_dim` != -1.

            :arg name: The tensor name.
        """
    @typing.overload
    def get_tensor_bytes_per_component(self, name: str, profile_index: typing.SupportsInt) -> int:
        """
            Return the number of bytes per component of an element.

            The vector component size is returned if :func:`get_tensor_vectorized_dim` != -1.

            :arg name: The tensor name.
        """
    @typing.overload
    def get_tensor_components_per_element(self, name: str) -> int:
        """
            Return the number of components included in one element.

            The number of elements in the vectors is returned if :func:`get_tensor_vectorized_dim` != -1.

            :arg name: The tensor name.
        """
    @typing.overload
    def get_tensor_components_per_element(self, name: str, profile_index: typing.SupportsInt) -> int:
        """
            Return the number of components included in one element.

            The number of elements in the vectors is returned if :func:`get_tensor_vectorized_dim` != -1.

            :arg name: The tensor name.
        """
    def get_tensor_dtype(self, name: str) -> DataType:
        """
            Return the required data type for a buffer from its tensor name.

            :arg name: The tensor name.
        """
    @typing.overload
    def get_tensor_format(self, name: str) -> TensorFormat:
        """
            Return the tensor format.

            :arg name: The tensor name.
        """
    @typing.overload
    def get_tensor_format(self, name: str, profile_index: typing.SupportsInt) -> TensorFormat:
        """
            Return the tensor format.

            :arg name: The tensor name.
        """
    @typing.overload
    def get_tensor_format_desc(self, name: str) -> str:
        """
            Return the human readable description of the tensor format.

            The description includes the order, vectorization, data type, strides, etc. For example:

            |   Example 1: CHW + FP32
            |       "Row major linear FP32 format"
            |   Example 2: CHW2 + FP16
            |       "Two wide channel vectorized row major FP16 format"
            |   Example 3: HWC8 + FP16 + Line Stride = 32
            |       "Channel major FP16 format where C % 8 == 0 and H Stride % 32 == 0"

            :arg name: The tensor name.
        """
    @typing.overload
    def get_tensor_format_desc(self, name: str, profile_index: typing.SupportsInt) -> str:
        """
            Return the human readable description of the tensor format.

            The description includes the order, vectorization, data type, strides, etc. For example:

            |   Example 1: CHW + FP32
            |       "Row major linear FP32 format"
            |   Example 2: CHW2 + FP16
            |       "Two wide channel vectorized row major FP16 format"
            |   Example 3: HWC8 + FP16 + Line Stride = 32
            |       "Channel major FP16 format where C % 8 == 0 and H Stride % 32 == 0"

            :arg name: The tensor name.
        """
    def get_tensor_location(self, name: str) -> TensorLocation:
        """
            Determine whether an input or output tensor must be on GPU or CPU.

            :arg name: The tensor name.
        """
    def get_tensor_mode(self, name: str) -> TensorIOMode:
        """
            Determine whether a tensor is an input or output tensor.

            :arg name: The tensor name.
        """
    def get_tensor_name(self, index: typing.SupportsInt) -> str:
        """
            Return the name of an input or output tensor.

            :arg index: The tensor index.
        """
    def get_tensor_profile_shape(self, name: str, profile_index: typing.SupportsInt) -> list[Dims]:
        """
            Get the minimum/optimum/maximum dimensions for a particular tensor under an optimization profile.

            :arg name: The tensor name.
            :arg profile_index: The index of the profile.
        """
    def get_tensor_profile_values(self, name: typing.SupportsInt, profile_index: str) -> list[list[int]]:
        """
            Get minimum/optimum/maximum values for an input shape binding under an optimization profile. If the specified binding is not an input shape binding, an exception is raised.

            :arg name: The tensor name.
            :arg profile_index: The index of the profile.

            :returns: A ``List[List[int]]`` of length 3, containing the minimum, optimum, and maximum values, in that order. If the values have not been set yet, an empty list is returned.
        """
    def get_tensor_shape(self, name: str) -> Dims:
        """
            Return the shape of an input or output tensor.

            :arg name: The tensor name.
        """
    @typing.overload
    def get_tensor_vectorized_dim(self, name: str) -> int:
        """
            Return the dimension index that the buffer is vectorized.

            Specifically -1 is returned if scalars per vector is 1.

            :arg name: The tensor name.
        """
    @typing.overload
    def get_tensor_vectorized_dim(self, name: str, profile_index: typing.SupportsInt) -> int:
        """
            Return the dimension index that the buffer is vectorized.

            Specifically -1 is returned if scalars per vector is 1.

            :arg name: The tensor name.
        """
    def get_weight_streaming_automatic_budget(self) -> int:
        """
            Get an automatic weight streaming budget based on available device memory. This value may change between TensorRT major and minor versions.
            Please use CudaEngine.weight_streaming_budget_v2 to set the returned budget.
        """
    def is_debug_tensor(self, name: str) -> bool:
        """
            Determine whether the given name corresponds to a debug tensor.

            :arg name: The tensor name.
        """
    def is_shape_inference_io(self, name: str) -> bool:
        """
            Determine whether a tensor is read or written by infer_shapes.

            :arg name: The tensor name.
        """
    def serialize(self) -> IHostMemory:
        """
            Serialize the engine to a stream.

            :returns: An :class:`IHostMemory` object containing the serialized :class:`ICudaEngine` .
        """
    def serialize_with_config(self, arg0: ISerializationConfig) -> IHostMemory:
        """
            Serialize the network to a stream.
        """
    @property
    def device_memory_size(self) -> int:
        ...
    @property
    def device_memory_size_v2(self) -> int:
        ...
    @property
    def engine_capability(self) -> EngineCapability:
        ...
    @property
    def hardware_compatibility_level(self) -> ...:
        ...
    @property
    def has_implicit_batch_dimension(self) -> bool:
        ...
    @property
    def minimum_weight_streaming_budget(self) -> int:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def num_aux_streams(self) -> int:
        ...
    @property
    def num_io_tensors(self) -> int:
        ...
    @property
    def num_layers(self) -> int:
        ...
    @property
    def num_optimization_profiles(self) -> int:
        ...
    @property
    def profiling_verbosity(self) -> ...:
        ...
    @property
    def refittable(self) -> bool:
        ...
    @property
    def streamable_weights_size(self) -> int:
        ...
    @property
    def tactic_sources(self) -> int:
        ...
    @property
    def weight_streaming_budget(self) -> int:
        ...
    @weight_streaming_budget.setter
    def weight_streaming_budget(self, arg1: typing.SupportsInt) -> bool:
        ...
    @property
    def weight_streaming_budget_v2(self) -> int:
        ...
    @weight_streaming_budget_v2.setter
    def weight_streaming_budget_v2(self, arg1: typing.SupportsInt) -> bool:
        ...
    @property
    def weight_streaming_scratch_memory_size(self) -> int:
        ...
class ICumulativeLayer(ILayer):
    """

        A cumulative layer in an :class:`INetworkDefinition` .

        This layer represents a cumulative operation across a tensor.

        It computes successive reductions across an axis of a tensor. The output
        always has the same shape as the input.

        If the reduction operation is summation, then this is also known as
        prefix-sum or cumulative sum.

        The operation has forward vs. reverse variants, and inclusive vs. exclusive variants.

        For example, let the input be a vector x of length n and the output be vector y.
        Then y[j] = sum(x[...]) where ... denotes a sequence of indices from this list:

        - inclusive + forward:   0..j
        - inclusive + reverse:   j..n-1
        - exclusive + forward:   0..j-1
        - exclusive + reverse: j+1..n-1

        For multidimensional tensors, the cumulative applies across a specified axis. For
        example, given a 2D input, a forward inclusive cumulative across axis 0 generates
        cumulative sums within each column.

        :ivar op: :class:`CumulativeOperation` The cumulative operation for the layer.
        :ivar exclusive: :class:`bool` Specifies whether it is an exclusive cumulative or inclusive cumulative.
        :ivar reverse: :class:`bool` Specifies whether the cumulative operation should be applied backward.

    """
    exclusive: bool
    op: CumulativeOperation
    reverse: bool
class IDebugListener:
    """

        A user-implemented class for notification when value of a debug tensor is updated.
    """
    def __init__(self) -> None:
        ...
    def process_debug_tensor(self, addr: typing_extensions.CapsuleType, location: ..., type: DataType, shape: Dims, name: str, stream: typing.SupportsInt) -> None:
        """
            User implemented callback function that is called when value of a debug tensor is updated and the debug state of the tensor is set to true. Content in the given address is only guaranteed to be valid for the duration of the callback.

            :arg location: TensorLocation of the tensor
            :arg addr: pointer to buffer
            :arg type: data Type of the tensor
            :arg shape: shape of the tensor
            :arg name: name of the tensor
            :arg stream: Cuda stream object

            :returns: True on success, False otherwise.
        """
class IDeconvolutionLayer(ILayer):
    """

        A deconvolution layer in an :class:`INetworkDefinition` .

        :ivar num_output_maps: :class:`int` The number of output feature maps for the deconvolution.
        :ivar pre_padding: :class:`DimsHW` The pre-padding. The start of input will be zero-padded by this number of elements in the height and width directions. Default: (0, 0)
        :ivar post_padding: :class:`DimsHW` The post-padding. The end of input will be zero-padded by this number of elements in the height and width directions. Default: (0, 0)
        :ivar padding_mode: :class:`PaddingMode` The padding mode. Padding mode takes precedence if both :attr:`IDeconvolutionLayer.padding_mode` and either :attr:`IDeconvolutionLayer.pre_padding` or :attr:`IDeconvolutionLayer.post_padding` are set.
        :ivar num_groups: :class:`int` The number of groups for a deconvolution. The input tensor channels are divided into this many groups, and a deconvolution is executed for each group, using a filter per group. The results of the group convolutions are concatenated to form the output. **Note** When using groups in int8 mode, the size of the groups (i.e. the channel count divided by the group count) must be a multiple of 4 for both input and output. Default: 1
        :ivar kernel: :class:`Weights` The kernel weights for the deconvolution. The weights are specified as a contiguous array in `CKRS` order, where `C` the number of input channels, `K` the number of output feature maps, and `R` and `S` are the height and width of the filter.
        :ivar bias: :class:`Weights` The bias weights for the deconvolution. Bias is optional. To omit bias, set this to an empty :class:`Weights` object. The bias is applied per-feature-map, so the number of weights (if non-zero) must be equal to the number of output feature maps.
        :ivar kernel_size_nd: :class:`Dims` The multi-dimension kernel size of the convolution.
        :ivar stride_nd: :class:`Dims` The multi-dimension stride of the deconvolution. Default: (1, ..., 1)
        :ivar padding_nd: :class:`Dims` The multi-dimension padding of the deconvolution. The input will be zero-padded by this number of elements in each dimension. Padding is symmetric. Default: (0, ..., 0)
    """
    dilation_nd: Dims
    kernel_size_nd: Dims
    padding_mode: PaddingMode
    padding_nd: Dims
    post_padding: Dims
    pre_padding: Dims
    stride_nd: Dims
    @property
    def bias(self) -> typing.Any:
        ...
    @bias.setter
    def bias(self, arg1: Weights) -> None:
        ...
    @property
    def kernel(self) -> typing.Any:
        ...
    @kernel.setter
    def kernel(self, arg1: Weights) -> None:
        ...
    @property
    def num_groups(self) -> int:
        ...
    @num_groups.setter
    def num_groups(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def num_output_maps(self) -> int:
        ...
    @num_output_maps.setter
    def num_output_maps(self, arg1: typing.SupportsInt) -> None:
        ...
class IDequantizeLayer(ILayer):
    """

        A Dequantize layer in an :class:`INetworkDefinition` .

        This layer accepts a signed 8-bit integer input tensor, and uses the configured scale and zeroPt inputs to
        dequantize the input according to:
        :math:`output = (input - zeroPt) * scale`

        The first input (index 0) is the tensor to be quantized.
        The second (index 1) and third (index 2) are the scale and zero point respectively.
        Each of scale and zeroPt must be either a scalar, or a 1D tensor.

        The zeroPt tensor is optional, and if not set, will be assumed to be zero.  Its data type must be
        tensorrt.int8. zeroPt must only contain zero-valued coefficients, because only symmetric quantization is
        supported.
        The scale value must be either a scalar for per-tensor quantization, or a 1D tensor for per-axis
        quantization. The size of the 1-D scale tensor must match the size of the quantization axis. The size of the
        scale must match the size of the zeroPt.

        The subgraph which terminates with the scale tensor must be a build-time constant.  The same restrictions apply
        to the zeroPt.
        The output type, if constrained, must be constrained to tensorrt.int8 or tensorrt.fp8. The input type, if constrained, must be
        constrained to tensorrt.float32, tensorrt.float16 or tensorrt.bfloat16.
        The output size is the same as the input size.

        IDequantizeLayer supports tensorrt.int8, tensorrt.float8, tensorrt.int4 and tensorrt.fp4 precision and will default to tensorrt.int8 precision during instantiation.
        IDequantizeLayer supports tensorrt.float32, tensorrt.float16 and tensorrt.bfloat16 output.

        :ivar axis: :class:`int` The axis along which dequantization occurs. The dequantization axis is in reference to the input tensor's dimensions.

        :ivar to_type: :class:`DataType` The specified data type of the output tensor. Must be tensorrt.float32 or tensorrt.float16.
    """
    block_shape: Dims
    to_type: DataType
    @property
    def axis(self) -> int:
        ...
    @axis.setter
    def axis(self, arg1: typing.SupportsInt) -> None:
        ...
class IDimensionExpr:
    """

        An `IDimensionExpr` represents an integer expression constructed from constants, input dimensions, and binary operations.

        These expressions are can be used in overrides of `IPluginV2DynamicExt::get_output_dimensions()` to define output dimensions in terms of input dimensions.
    """
    def get_constant_value(self) -> int:
        """
            Get the value of the constant.

            If is_constant(), returns value of the constant.
            Else, return int64 minimum.
        """
    def is_constant(self) -> bool:
        """
            Return true if expression is a build-time constant.
        """
    def is_size_tensor(self) -> bool:
        """
            Return true if this denotes the value of a size tensor.
        """
class IDistCollectiveLayer(ILayer):
    """

        A dist collective layer in an :class:`INetworkDefinition` .
    """
class IDynamicQuantizeLayer(ILayer):
    """

        A DynamicQuantize layer in an :class:`INetworkDefinition` .

        This layer performs dynamic block quantization of its input tensor and outputs the quantized data and the computed block scale-factors.
        The size of the blocked axis must be divisible by the block size.

        The first input (index 0) is the tensor to be quantized. Its data type must be one of DataType::kFLOAT,
        DataType::kHALF, or DataType::kBF16. Currently only 2D and 3D inputs are supported.

        The second input (index 1) is the double quantization scale factor. It is a scalar scale factor used to quantize the computed block scales-factors.

        :ivar axis: :class:`int` The axis that is sliced into blocks. The axis must be the last dimension or the second to last dimension.
        :ivar block_size: :class:`int` The number of elements that are quantized using a shared scale factor. Supports block sizes of 16 with NVFP4 quantization and 32 with MXFP8 quantization.
        :ivar output_type: :class:`DataType` The data type of the quantized output tensor, must be either DataType::kFP4 (NVFP4 quantization) or DataType::kFP8 (MXFP8 quantization).
        :ivar scale_type: :class:`DataType` The data type of the scale factor used for quantizing the input data, must be DataType::kFP8 (NVFP4 quantization) or DataType::kE8M0 (MXFP8 quantization).
    """
    block_shape: Dims
    scale_type: DataType
    to_type: DataType
    @property
    def axis(self) -> int:
        ...
    @axis.setter
    def axis(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def block_size(self) -> int:
        ...
    @block_size.setter
    def block_size(self, arg1: typing.SupportsInt) -> None:
        ...
class IEinsumLayer(ILayer):
    """

        An Einsum layer in an :class:`INetworkDefinition` .

        This layer implements a summation over the elements of the inputs along dimensions specified by the equation parameter, based on the Einstein summation convention.
        The layer can have one or more inputs of rank >= 0. All the inputs must be of same data type. This layer supports all TensorRT data types except :class:`bool`.
        There is one output tensor of the same type as the input tensors. The shape of output tensor is determined by the equation.

        The equation specifies ASCII lower-case letters for each dimension in the inputs in the same order as the dimensions, separated by comma for each input.
        The dimensions labeled with the same subscript must match or be broadcastable.
        Repeated subscript labels in one input take the diagonal.
        Repeating a label across multiple inputs means that those axes will be multiplied.
        Omitting a label from the output means values along those axes will be summed.
        In implicit mode, the indices which appear once in the expression will be part of the output in increasing alphabetical order.
        In explicit mode, the output can be controlled by specifying output subscript labels by adding an arrow (‘->’) followed by subscripts for the output.
        For example, “ij,jk->ik” is equivalent to “ij,jk”.
        Ellipsis (‘...’) can be used in place of subscripts to broadcast the dimensions.
        See the TensorRT Developer Guide for more details on equation syntax.

        Many common operations can be expressed using the Einsum equation.
        For example:
        Matrix Transpose:             ij->ji
        Sum:                          ij->
        Matrix-Matrix Multiplication: ik,kj->ij
        Dot Product:                  i,i->
        Matrix-Vector Multiplication: ik,k->i
        Batch Matrix Multiplication:  ijk,ikl->ijl
        Batch Diagonal:               ...ii->...i

        Note that TensorRT does not support ellipsis or diagonal operations.

        :ivar equation: :class:`str` The Einsum equation of the layer.
            The equation is a comma-separated list of subscript labels, where each label refers to a dimension of the corresponding tensor.

    """
    equation: str
class IElementWiseLayer(ILayer):
    """

        A elementwise layer in an :class:`INetworkDefinition` .

        This layer applies a per-element binary operation between corresponding elements of two tensors.

        The input dimensions of the two input tensors must be equal, and the output tensor is the same size as each input.

        :ivar op: :class:`ElementWiseOperation` The binary operation for the layer.
    """
    op: ElementWiseOperation
class IErrorRecorder:
    """

        Reference counted application-implemented error reporting interface for TensorRT objects.

        The error reporting mechanism is a user defined object that interacts with the internal state of the object
        that it is assigned to in order to determine information about abnormalities in execution. The error recorder
        gets both an error enum that is more descriptive than pass/fail and also a description that gives more
        detail on the exact failure modes. In the safety context, the error strings are all limited to 128 characters
        in length.
        The ErrorRecorder gets passed along to any class that is created from another class that has an ErrorRecorder
        assigned to it. For example, assigning an ErrorRecorder to an Builder allows all INetwork's, ILayer's, and
        ITensor's to use the same error recorder. For functions that have their own ErrorRecorder accessor functions.
        This allows registering a different error recorder or de-registering of the error recorder for that specific
        object.

        The ErrorRecorder object implementation must be thread safe if the same ErrorRecorder is passed to different
        interface objects being executed in parallel in different threads. All locking and synchronization is
        pushed to the interface implementation and TensorRT does not hold any synchronization primitives when accessing
        the interface functions.
    """
    def __init__(self) -> None:
        ...
    def clear(self) -> None:
        """
            Clear the error stack on the error recorder.

            Removes all the tracked errors by the error recorder.  This function must guarantee that after
            this function is called, and as long as no error occurs, :attr:`num_errors` will be zero.
        """
    def get_error_code(self, arg0: typing.SupportsInt) -> ErrorCodeTRT:
        """
            Returns the ErrorCode enumeration.

            The error_idx specifies what error code from 0 to :attr:`num_errors`-1 that the application
            wants to analyze and return the error code enum.

            :arg error_idx: A 32bit integer that indexes into the error array.

            :returns: Returns the enum corresponding to error_idx.
        """
    def get_error_desc(self, arg0: typing.SupportsInt) -> str:
        """
            Returns description of the error.

            For the error specified by the idx value, return description of the error. In the safety context there is a
            constant length requirement to remove any dynamic memory allocations and the error message
            may be truncated. The format of the error description is "<EnumAsStr> - <Description>".

            :arg error_idx: A 32bit integer that indexes into the error array.

            :returns: Returns description of the error.
        """
    def has_overflowed(self) -> bool:
        """
            Determine if the error stack has overflowed.

            In the case when the number of errors is large, this function is used to query if one or more
            errors have been dropped due to lack of storage capacity. This is especially important in the
            automotive safety case where the internal error handling mechanisms cannot allocate memory.

            :returns: True if errors have been dropped due to overflowing the error stack.
        """
    def num_errors(self) -> int:
        """
            Return the number of errors

            Determines the number of errors that occurred between the current point in execution
            and the last time that the clear() was executed. Due to the possibility of asynchronous
            errors occuring, a TensorRT API can return correct results, but still register errors
            with the Error Recorder. The value of getNbErrors must monotonically increases until clear()
            is called.

            :returns: Returns the number of errors detected, or 0 if there are no errors.
        """
    def report_error(self, arg0: ErrorCodeTRT, arg1: str) -> bool:
        """
            Clear the error stack on the error recorder.

            Report an error to the user that has a given value and human readable description. The function returns false
            if processing can continue, which implies that the reported error is not fatal. This does not guarantee that
            processing continues, but provides a hint to TensorRT.

            :arg val: The error code enum that is being reported.
            :arg desc: The description of the error.

            :returns: True if the error is determined to be fatal and processing of the current function must end.
        """
    @property
    def MAX_DESC_LENGTH() -> int:
        ...
class IExecutionContext:
    """

        Context for executing inference using an :class:`ICudaEngine` .
        Multiple :class:`IExecutionContext` s may exist for one :class:`ICudaEngine` instance, allowing the same
        :class:`ICudaEngine` to be used for the execution of multiple batches simultaneously.

        :ivar debug_sync: :class:`bool` The debug sync flag. If this flag is set to true, the :class:`ICudaEngine` will log the successful execution for each kernel during execute_v2().
        :ivar profiler: :class:`IProfiler` The profiler in use by this :class:`IExecutionContext` .
        :ivar engine: :class:`ICudaEngine` The associated :class:`ICudaEngine` .
        :ivar name: :class:`str` The name of the :class:`IExecutionContext` .
        :ivar device_memory: :class:`capsule` The device memory for use by this execution context. The memory must be aligned with cuda memory alignment property (using :func:`cuda.cudart.cudaGetDeviceProperties()`), and its size must be large enough for performing inference with the given network inputs. :func:`engine.device_memory_size` and :func:`engine.get_device_memory_size_for_profile` report upper bounds of the size. Setting memory to nullptr is acceptable if the reported size is 0. If using :func:`execute_async_v3()` to run the network, the memory is in use from the invocation of :func:`execute_async_v3()` until network execution is complete. If using :func:`execute_v2()`, it is in use until :func:`execute_v2()` returns. Releasing or otherwise using the memory for other purposes, including using it in another execution context running in parallel, during this time will result in undefined behavior.
        :ivar active_optimization_profile: :class:`int` The active optimization profile for the context. The selected profile will be used in subsequent calls to :func:`execute_v2()`. Profile 0 is selected by default. This is a readonly property and active optimization profile can be changed with :func:`set_optimization_profile_async()`. Changing this value will invalidate all dynamic bindings for the current execution context, so that they have to be set again using :func:`set_input_shape` before calling either :func:`execute_v2()`.
        :ivar all_binding_shapes_specified: :class:`bool` Whether all dynamic dimensions of input tensors have been specified by calling :func:`set_input_shape` . Trivially true if network has no dynamically shaped input tensors. Does not work with name-base interfaces eg. :func:`set_input_shape()`. Use :func:`infer_shapes()` instead.
        :ivar all_shape_inputs_specified: :class:`bool` Whether values for all input shape tensors have been specified by calling :func:`set_shape_input` . Trivially true if network has no input shape bindings. Does not work with name-base interfaces eg. :func:`set_input_shape()`. Use :func:`infer_shapes()` instead.
        :ivar error_recorder: :class:`IErrorRecorder` Application-implemented error reporting interface for TensorRT objects.
        :ivar enqueue_emits_profile: :class:`bool` Whether enqueue emits layer timing to the profiler. The default value is :class:`True`. If set to :class:`False`, enqueue will be asynchronous if there is a profiler attached. An extra method :func:`IExecutionContext::report_to_profiler()` needs to be called to obtain the profiling data and report to the profiler attached.
        :ivar persistent_cache_limit: The maximum size of persistent L2 cache that this execution context may use for activation caching. Activation caching is not supported on all architectures - see "How TensorRT uses Memory" in the developer guide for details. The default is 0 Bytes.
        :ivar nvtx_verbosity: The NVTX verbosity of the execution context. Building with DETAILED verbosity will generally increase latency in enqueueV3(). Call this method to select NVTX verbosity in this execution context at runtime. The default is the verbosity with which the engine was built, and the verbosity may not be raised above that level. This function does not affect how IEngineInspector interacts with the engine.
        :ivar temporary_allocator: :class:`IGpuAllocator` The GPU allocator used for internal temporary storage.
    """
    debug_sync: bool
    enqueue_emits_profile: bool
    error_recorder: IErrorRecorder
    name: str
    nvtx_verbosity: ...
    profiler: IProfiler
    temporary_allocator: ...
    unfused_tensors_debug_state: bool
    @staticmethod
    def __enter__(this):
        ...
    @staticmethod
    def __exit__(this, exc_type, exc_value, traceback):
        """

            Context managers are deprecated and have no effect. Objects are automatically freed when
            the reference count reaches 0.

        """
    def __del__(self) -> None:
        ...
    def execute_async_v3(self, stream_handle: typing.SupportsInt) -> bool:
        """
            Asynchronously execute inference.

            Modifying or releasing memory that has been registered for the tensors before stream synchronization or the event passed to :func:`set_input_consumed_event` has been triggered results in undefined behavior.

            Input tensors can be released after the :func:`set_input_consumed_event` whereas output tensors require stream synchronization.

            :arg stream_handle: The cuda stream on which the inference kernels will be enqueued. Using default stream may lead to performance issues due to additional cudaDeviceSynchronize() calls by TensorRT to ensure correct synchronizations. Please use non-default stream instead.
        """
    def execute_v2(self, bindings: collections.abc.Sequence[typing.SupportsInt]) -> bool:
        """
            Synchronously execute inference on a batch.
            This method requires a array of input and output buffers.

            :arg bindings: A list of integers representing input and output buffer addresses for the network.

            :returns: True if execution succeeded.
        """
    def get_debug_listener(self) -> IDebugListener:
        """
            Get debug listener for execution context.

            :returns: The :class:`IDebugListener` of the execution context.
        """
    def get_debug_state(self, name: str) -> bool:
        """
            Get the debug state of the tensor.

            :arg name: The name of the tensor.
        """
    def get_input_consumed_event(self) -> int:
        """
            Return the event associated with consuming the input tensors.
        """
    def get_max_output_size(self, name: str) -> int:
        """
            Return the upper bound on an output tensor's size, in bytes, based on the current optimization profile.

            If the profile or input shapes are not yet set, or the provided name does not map to an output, returns -1.

            :arg name: The tensor name.
        """
    def get_output_allocator(self, name: str) -> IOutputAllocator:
        """
            Return the output allocator associated with given output tensor, or ``None`` if the provided name does not map to an output tensor.

            :arg name: The tensor name.
        """
    def get_runtime_config(self) -> IRuntimeConfig:
        """
            Get the runtime configuration. From the execution context.

            :returns: The runtime configuration.
        """
    def get_tensor_address(self, name: str) -> int:
        """
            Get memory address for the given input or output tensor.

            :arg name: The tensor name.
        """
    def get_tensor_shape(self, name: str) -> Dims:
        """
            Return the shape of the given input or output tensor.

            :arg name: The tensor name.
        """
    def get_tensor_strides(self, name: str) -> Dims:
        """
            Return the strides of the buffer for the given tensor name.

            Note that strides can be different for different execution contexts with dynamic shapes.

            :arg name: The tensor name.
        """
    def infer_shapes(self) -> list[str]:
        """
            Infer shapes and return the names of any tensors that are insufficiently specified.

            An input tensor is insufficiently specified if either of the following is true:

            * It has dynamic dimensions and its runtime dimensions have not yet
              been specified via :func:`set_input_shape` .

            * is_shape_inference_io(t) is True and the tensor's address has not yet been set.

            :returns: A ``List[str]`` indicating the names of any tensors which have not been sufficiently
                specified, or an empty list on success.

            :raises: RuntimeError if shape inference fails due to reasons other than insufficiently specified tensors.
        """
    def report_to_profiler(self) -> bool:
        """
            Calculate layer timing info for the current optimization profile in IExecutionContext and update the profiler after one iteration of inference launch.

            If the enqueue_emits_profiler flag was set to true, the enqueue function will calculate layer timing implicitly if a profiler is provided. There is no need to call this function.
            If the enqueue_emits_profiler flag was set to false, the enqueue function will record the CUDA event timers if a profiler is provided. But it will not perform the layer timing calculation. This function needs to be called explicitly to calculate layer timing for the previous inference launch.

            In the CUDA graph launch scenario, it will record the same set of CUDA events as in regular enqueue functions if the graph is captured from an :class:`IExecutionContext` with profiler enabled. This function needs to be called after graph launch to report the layer timing info to the profiler.

            Profiling CUDA graphs is only available from CUDA 11.1 onwards.

            :returns: :class:`True` if the call succeeded, else :class:`False` (e.g. profiler not provided, in CUDA graph capture mode, etc.)
        """
    def set_all_tensors_debug_state(self, flag: bool) -> bool:
        """
            Turn the debug state of all debug tensors on or off.

            :arg flag: True if turning on debug state of tensor. False if turning off.
        """
    def set_aux_streams(self, aux_streams: collections.abc.Sequence[typing.SupportsInt]) -> None:
        """
            Set the auxiliary streams that TensorRT should launch kernels on in the next execute_async_v3() call.

            If set, TensorRT will launch the kernels that are supposed to run on the auxiliary streams using the streams provided by the user with this API. If this API is not called before the execute_async_v3() call, then TensorRT will use the auxiliary streams created by TensorRT internally.

            TensorRT will always insert event synchronizations between the main stream provided via execute_async_v3() call and the auxiliary streams:
             - At the beginning of the execute_async_v3() call, TensorRT will make sure that all the auxiliary streams wait on the activities on the main stream.
             - At the end of the execute_async_v3() call, TensorRT will make sure that the main stream wait on the activities on all the auxiliary streams.

            The provided auxiliary streams must not be the default stream and must all be different to avoid deadlocks.

            :arg aux_streams: A list of cuda streams. If the length of the list is greater than engine.num_aux_streams, then only the first "engine.num_aux_streams" streams will be used. If the length is less than engine.num_aux_streams, such as an empty list, then TensorRT will use the provided streams for the first few auxiliary streams, and will create additional streams internally for the rest of the auxiliary streams.
        """
    def set_communicator(self, communicator: typing_extensions.CapsuleType) -> bool:
        """
            Set the NCCL communicator for the execution context.

            :param communicator: A pointer to the NCCL communicator that is used by the execution context.

            The communicator must be uniform across all multi-device instances or undefined
            behavior occurs.

            :returns: True if the communicator was set successfully, False otherwise.
        """
    def set_debug_listener(self, listener: IDebugListener) -> bool:
        """
            Set debug listener for execution context.

            :arg listener: The :class:`IDebugListener`.
        """
    def set_device_memory(self, memory: typing.SupportsInt, size: typing.SupportsInt) -> None:
        """
            The device memory for use by this :class:`IExecutionContext` .

            :arg memory: 256-byte aligned device memory.
            :arg size: Size of the provided memory. This must be at least as large as CudaEngine.get_device_memory_size_v2

            If using :func:`enqueue_v3()`, it is in use until :func:`enqueue_v3()` returns. Releasing or otherwise using the memory for other
            purposes during this time will result in undefined behavior. This includes using the same memory for a parallel execution context.
        """
    def set_input_consumed_event(self, event: typing.SupportsInt) -> bool:
        """
            Mark all input tensors as consumed.

            :arg event: The cuda event that is triggered after all input tensors have been consumed.
        """
    @typing.overload
    def set_input_shape(self, name: str, shape: tuple) -> bool:
        """
            Set shape for the given input tensor.

            :arg name: The input tensor name.
            :arg shape: The input tensor shape.
        """
    @typing.overload
    def set_input_shape(self, name: str, shape: list) -> bool:
        """
            Set shape for the given input tensor.

            :arg name: The input tensor name.
            :arg shape: The input tensor shape.
        """
    @typing.overload
    def set_input_shape(self, name: str, shape: Dims) -> bool:
        """
            Set shape for the given input tensor.

            :arg name: The input tensor name.
            :arg shape: The input tensor shape.
        """
    def set_optimization_profile_async(self, profile_index: typing.SupportsInt, stream_handle: typing.SupportsInt) -> bool:
        """
            Set the optimization profile with async semantics

            :arg profile_index: The index of the optimization profile

            :arg stream_handle: cuda stream on which the work to switch optimization profile can be enqueued

            When an optimization profile is switched via this API, TensorRT may require that data is copied via cudaMemcpyAsync. It is the
            application's responsibility to guarantee that synchronization between the profile sync stream and the enqueue stream occurs.

            :returns: :class:`True` if the optimization profile was set successfully
        """
    def set_output_allocator(self, name: str, output_allocator: IOutputAllocator) -> bool:
        """
            Set output allocator to use for the given output tensor.

            Pass ``None`` to unset the output allocator.

            The allocator is called by :func:`execute_async_v3`.

            :arg name: The tensor name.
            :arg output_allocator: The output allocator.
        """
    def set_tensor_address(self, name: str, memory: typing.SupportsInt) -> bool:
        """
            Set memory address for the given input or output tensor.

            :arg name: The tensor name.
            :arg memory: The memory address.
        """
    def set_tensor_debug_state(self, name: str, flag: bool) -> bool:
        """
            Turn the debug state of a tensor on or off. The Tensor must have been marked as a debug tensor during build time.

            :arg name: The name of the target tensor.
            :arg flag: True if turning on debug state of tensor. False if turning off.
        """
    def update_device_memory_size_for_shapes(self) -> int:
        """
            Recompute the internal activation buffer sizes based on the current input shapes, and return the total amount of memory required.

            Users can allocate the device memory based on the size returned and provided the memory to TRT with an assignment to IExecutionContext.device_memory. Must specify all input shapes and the optimization profile to use before calling this function, otherwise the partition will be invalidated.
        """
    @property
    def active_optimization_profile(self) -> int:
        ...
    @property
    def all_binding_shapes_specified(self) -> bool:
        ...
    @property
    def all_shape_inputs_specified(self) -> bool:
        ...
    @property
    def engine(self) -> ...:
        ...
    @property
    def persistent_cache_limit(self) -> int:
        ...
    @persistent_cache_limit.setter
    def persistent_cache_limit(self, arg1: typing.SupportsInt) -> None:
        ...
class IExprBuilder:
    """

        Object for constructing `IDimensionExpr`.

        There is no public way to construct an `IExprBuilder`. It appears as an argument to method `IPluginV2DynamicExt::get_output_dimensions()`. Overrides of that method can use that `IExprBuilder` argument to construct expressions that define output dimensions in terms of input dimensions.

        Clients should assume that any values constructed by the `IExprBuilder` are destroyed after `IPluginV2DynamicExt::get_output_dimensions()` returns.
    """
    def __init__(self) -> None:
        ...
    def constant(self, arg0: typing.SupportsInt) -> IDimensionExpr:
        """
            Return a IDimensionExpr for the given value.
        """
    def declare_size_tensor(self, arg0: typing.SupportsInt, arg1: IDimensionExpr, arg2: IDimensionExpr) -> IDimensionExpr:
        """
            Declare a size tensor at the given output index, with the specified auto-tuning formula and upper bound.

            A size tensor allows a plugin to have output dimensions that cannot be computed solely from input dimensions.
            For example, suppose a plugin implements the equivalent of INonZeroLayer for 2D input. The plugin can
            have one output for the indices of non-zero elements, and a second output containing the number of non-zero
            elements. Suppose the input has size [M,N] and has K non-zero elements. The plugin can write K to the second
            output. When telling TensorRT that the first output has shape [2,K], plugin uses IExprBuilder.constant() and
            IExprBuilder.declare_size_tensor(1,...) to create the IDimensionExpr that respectively denote 2 and K.

            TensorRT also needs to know the value of K to use for auto-tuning and an upper bound on K so that it can
            allocate memory for the output tensor. In the example, suppose typically half of the plugin's input elements
            are non-zero, and all the elements might be nonzero. then using M*N/2 might be a good expression for the opt
            parameter, and M*N for the upper bound. IDimensionsExpr for these expressions can be constructed from
            IDimensionsExpr for the input dimensions.
        """
    def operation(self, arg0: DimensionOperation, arg1: IDimensionExpr, arg2: IDimensionExpr) -> IDimensionExpr:
        """
            Return a IDimensionExpr that represents the given operation applied to first and second.
            Returns None if op is not a valid DimensionOperation.
        """
class IFillLayer(ILayer):
    """

        A fill layer in an :class:`INetworkDefinition` .

        The data type of the output tensor can be specified by :attr:`to_type`. Supported output types for each fill operation is as follows.

        ================   =====================
        Operation          to_type
        ================   =====================
        kLINSPACE          int32, int64, float32
        kRANDOM_UNIFORM    float16, float32
        kRANDOM_NORMAL     float16, float32
        ================   =====================

        :ivar to_type: :class:`DataType` The specified data type of the output tensor. Defaults to tensorrt.float32.
    """
    alpha: typing.Any
    beta: typing.Any
    operation: FillOperation
    shape: Dims
    to_type: DataType
    def is_alpha_beta_int64(self) -> bool:
        ...
    def set_input(self: ILayer, index: typing.SupportsInt, tensor: ITensor) -> None:
        """
            replace an input of this layer with a specific tensor.

            =====   ==========================================================================================================
            Index   Description for kLINSPACE
            =====   ==========================================================================================================
                0     Shape tensor, represents the output tensor's dimensions.
                1     Start, a scalar, represents the start value.
                2     Delta, a 1D tensor, length equals to shape tensor's nbDims, represents the delta value for each dimension.
            =====   ==========================================================================================================

            =====   ========================================================
            Index   Description for kRANDOM_UNIFORM
            =====   ========================================================
                0     Shape tensor, represents the output tensor's dimensions.
                1     Minimum, a scalar, represents the minimum random value.
                2     Maximum, a scalar, represents the maximal random value.
            =====   ========================================================

            =====   ========================================================
            Index   Description for kRANDOM_NORMAL
            =====   ========================================================
                0     Shape tensor, represents the output tensor's dimensions.
                1     Mean, a scalar, represents the mean of the normal distribution.
                2     Scale, a scalar, represents the standard deviation of the normal distribution.
            =====   ========================================================

            :arg index: the index of the input to modify.
            :arg tensor: the input tensor.
        """
class IGatherLayer(ILayer):
    """

        A gather layer in an :class:`INetworkDefinition` .

        :ivar axis: :class:`int` The non-batch dimension axis to gather on. The axis must be less than the number of non-batch dimensions in the data input.
        :ivar num_elementwise_dims: :class:`int` The number of leading dimensions of indices tensor to be handled elementwise. For `GatherMode.DEFAULT`, it can be 0 or 1. For `GatherMode::kND`, it can be between 0 and one less than rank(data). For `GatherMode::kELEMENT`, it must be 0.
        :ivar mode: :class:`GatherMode` The gather mode.
    """
    mode: ...
    @property
    def axis(self) -> int:
        ...
    @axis.setter
    def axis(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def num_elementwise_dims(self) -> int:
        ...
    @num_elementwise_dims.setter
    def num_elementwise_dims(self, arg1: typing.SupportsInt) -> None:
        ...
class IGpuAllocator(IVersionedInterface):
    """
    Application-implemented class for controlling allocation on the GPU.

    To implement a custom allocator, ensure that you explicitly instantiate the base class in :func:`__init__` :
    ::

        class MyAllocator(trt.IGpuAllocator):
            def __init__(self):
                trt.IGpuAllocator.__init__(self)

            ...

    Note that all methods below (allocate, reallocate, deallocate, allocate_async, deallocate_async) must be overridden in the custom allocator, or else pybind11 would not be able to call the method from a custom allocator.
    """
    def __init__(self) -> None:
        ...
    def allocate(self, size: typing.SupportsInt, alignment: typing.SupportsInt, flags: typing.SupportsInt) -> typing_extensions.CapsuleType:
        """
            [DEPRECATED] Deprecated in TensorRT 10.0. Please use allocate_async instead.
            A callback implemented by the application to handle acquisition of GPU memory.
            If an allocation request of size 0 is made, ``None`` should be returned.

            If an allocation request cannot be satisfied, ``None`` should be returned.

            :arg size: The size of the memory required.
            :arg alignment: The required alignment of memory. Alignment will be zero
                or a power of 2 not exceeding the alignment guaranteed by cudaMalloc.
                Thus this allocator can be safely implemented with cudaMalloc/cudaFree.
                An alignment value of zero indicates any alignment is acceptable.
            :arg flags: Allocation flags. See :class:`AllocatorFlag`

            :returns: The address of the allocated memory
        """
    def allocate_async(self, size: typing.SupportsInt, alignment: typing.SupportsInt, flags: typing.SupportsInt, stream: typing.SupportsInt) -> typing_extensions.CapsuleType:
        """
            A callback implemented by the application to handle acquisition of GPU memory asynchronously.
            This is just a wrapper around a syncronous method allocate.
            For the asynchronous allocation please use the corresponding IGpuAsyncAllocator class.
            If an allocation request of size 0 is made, ``None`` should be returned.

            If an allocation request cannot be satisfied, ``None`` should be returned.

            :arg size: The size of the memory required.
            :arg alignment: The required alignment of memory. Alignment will be zero
                or a power of 2 not exceeding the alignment guaranteed by cudaMalloc.
                Thus this allocator can be safely implemented with cudaMalloc/cudaFree.
                An alignment value of zero indicates any alignment is acceptable.
            :arg flags: Allocation flags. See :class:`AllocatorFlag`
            :arg stream: CUDA stream

            :returns: The address of the allocated memory
        """
    def deallocate(self, memory: typing_extensions.CapsuleType) -> bool:
        """
            [DEPRECATED] Deprecated in TensorRT 10.0. Please use dealocate_async instead;
            A callback implemented by the application to handle release of GPU memory.

            TensorRT may pass a 0 to this function if it was previously returned by ``allocate()``.

            :arg memory: The memory address of the memory to release.

            :returns: True if the acquired memory is released successfully.
        """
    def deallocate_async(self, memory: typing_extensions.CapsuleType, stream: typing.SupportsInt) -> bool:
        """
            A callback implemented by the application to handle release of GPU memory asynchronously.
            This is just a wrapper around a syncronous method deallocate.
            For the asynchronous deallocation please use the corresponding IGpuAsyncAllocator class.

            TensorRT may pass a 0 to this function if it was previously returned by ``allocate()``.

            :arg memory: The memory address of the memory to release.
            :arg stream: CUDA stream

            :returns: True if the acquired memory is released successfully.
        """
    def reallocate(self, address: typing_extensions.CapsuleType, alignment: typing.SupportsInt, new_size: typing.SupportsInt) -> typing_extensions.CapsuleType:
        """
            A callback implemented by the application to resize an existing allocation.

            Only allocations which were allocated with AllocatorFlag.RESIZABLE will be resized.

            Options are one of:
            - resize in place leaving min(old_size, new_size) bytes unchanged and return the original address
            - move min(old_size, new_size) bytes to a new location of sufficient size and return its address
            - return None, to indicate that the request could not be fulfilled.

            If None is returned, TensorRT will assume that resize() is not implemented, and that the
            allocation at address is still valid.

            This method is made available for use cases where delegating the resize
            strategy to the application provides an opportunity to improve memory management.
            One possible implementation is to allocate a large virtual device buffer and
            progressively commit physical memory with cuMemMap. CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
            is suggested in this case.

            TensorRT may call realloc to increase the buffer by relatively small amounts.

            :arg address: the address of the original allocation.
            :arg alignment: The alignment used by the original allocation.
            :arg new_size: The new memory size required.

            :returns: The address of the reallocated memory
        """
class IGpuAsyncAllocator(IGpuAllocator):
    """
    Application-implemented class for controlling allocation on the GPU.

    To implement a custom allocator, ensure that you explicitly instantiate the base class in :func:`__init__` :
    ::

        class MyAllocator(trt.IGpuAsyncAllocator):
            def __init__(self):
                trt.IGpuAllocator.__init__(self)

            ...

    Note that all methods below (allocate, reallocate, deallocate, allocate_async, reallocate_async, deallocate_async) must be overridden in the custom allocator, or else pybind11 would not be able to call the method from a custom allocator.
    """
    def __init__(self) -> None:
        ...
    def allocate(self, size: typing.SupportsInt, alignment: typing.SupportsInt, flags: typing.SupportsInt) -> typing_extensions.CapsuleType:
        """
            [DEPRECATED] Deprecated in TensorRT 10.0. Please use allocate_async instead.
            A callback implemented by the application to handle acquisition of GPU memory.
            This is just a wrapper around a synchronous method allocate_async passing the default stream.

            If an allocation request of size 0 is made, ``None`` should be returned.

            If an allocation request cannot be satisfied, ``None`` should be returned.

            :arg size: The size of the memory required.
            :arg alignment: The required alignment of memory. Alignment will be zero
                or a power of 2 not exceeding the alignment guaranteed by cudaMalloc.
                Thus this allocator can be safely implemented with cudaMalloc/cudaFree.
                An alignment value of zero indicates any alignment is acceptable.
            :arg flags: Allocation flags. See :class:`AllocatorFlag`

            :returns: The address of the allocated memory
        """
    def allocate_async(self: IGpuAllocator, size: typing.SupportsInt, alignment: typing.SupportsInt, flags: typing.SupportsInt, stream: typing.SupportsInt) -> typing_extensions.CapsuleType:
        """
            A callback implemented by the application to handle acquisition of GPU memory asynchronously.
            If an allocation request of size 0 is made, ``None`` should be returned.

            If an allocation request cannot be satisfied, ``None`` should be returned.

            :arg size: The size of the memory required.
            :arg alignment: The required alignment of memory. Alignment will be zero
                or a power of 2 not exceeding the alignment guaranteed by cudaMalloc.
                Thus this allocator can be safely implemented with cudaMalloc/cudaFree.
                An alignment value of zero indicates any alignment is acceptable.
            :arg flags: Allocation flags. See :class:`AllocatorFlag`
            :arg stream: CUDA stream

            :returns: The address of the allocated memory
        """
    def deallocate(self, memory: typing_extensions.CapsuleType) -> bool:
        """
            [DEPRECATED] Deprecated in TensorRT 10.0. Please use deallocate_async instead.
            A callback implemented by the application to handle release of GPU memory.
            This is just a wrapper around a synchronous method deallocate_async passing the default stream.

            TensorRT may pass a 0 to this function if it was previously returned by ``allocate()``.

            :arg memory: The memory address of the memory to release.

            :returns: True if the acquired memory is released successfully.
        """
    def deallocate_async(self: IGpuAllocator, memory: typing_extensions.CapsuleType, stream: typing.SupportsInt) -> bool:
        """
            A callback implemented by the application to handle release of GPU memory asynchronously.

            TensorRT may pass a 0 to this function if it was previously returned by ``allocate()``.

            :arg memory: The memory address of the memory to release.
            :arg stream: CUDA stream

            :returns: True if the acquired memory is released successfully.
        """
class IGridSampleLayer(ILayer):
    """

        A grid sample layer in an :class:`INetworkDefinition` .

        This layer uses an input tensor and a grid tensor to produce an interpolated output tensor.
        The input and grid tensors must shape tensors of rank 4. The only supported `SampleMode` s are
        trt.samplemode.CLAMP, trt.samplemode.FILL, and trt.samplemode.REFLECT.

        :ivar interpolation_mode: class:`InterpolationMode` The interpolation type to use. Defaults to LINEAR.
        :ivar align_corners: class:`bool` the align mode to use. Defaults to False.
        :ivar sample_mode: :class:`SampleMode` The sample mode to use. Defaults to FILL.
    """
    align_corners: bool
    interpolation_mode: InterpolationMode
    sample_mode: SampleMode
class IHostMemory:
    """

        Handles library allocated memory that is accessible to the user.

        The memory allocated via the host memory object is owned by the library and will be de-allocated when object is destroyed.

        This class exposes a buffer interface using Python's buffer protocol.

        :ivar dtype: :class:`DataType` The data type of this buffer.
        :ivar nbytes: :class:`int` Total bytes consumed by the elements of the buffer.
    """
    @staticmethod
    def __enter__(this):
        ...
    @staticmethod
    def __exit__(this, exc_type, exc_value, traceback):
        """

            Context managers are deprecated and have no effect. Objects are automatically freed when
            the reference count reaches 0.

        """
    def __buffer__(self, flags):
        """
        Return a buffer object that exposes the underlying memory of the object.
        """
    def __del__(self) -> None:
        ...
    def __release_buffer__(self, buffer):
        """
        Release the buffer object that exposes the underlying memory of the object.
        """
    @property
    def dtype(self) -> DataType:
        ...
    @property
    def nbytes(self) -> int:
        ...
class IIdentityLayer(ILayer):
    """

        A layer that represents the identity function.

        If tensor precision is explicitly specified, it can be used to transform from one precision to another.

        Other than conversions between the same type (``float32`` -> ``float32`` for example), the only valid conversions are:

        (``float32`` | ``float16`` | ``int32`` | ``bool``) -> (``float32`` | ``float16`` | ``int32`` | ``bool``)

        (``float32`` | ``float16``) -> ``uint8``

        ``uint8`` -> (``float32`` | ``float16``)
    """
class IIfConditional:
    """

        Helper for constructing conditionally-executed subgraphs.

        An If-conditional conditionally executes (lazy evaluation) part of the network according
        to the following pseudo-code:

        .. code-block:: none

            If condition is true Then:
                output = trueSubgraph(trueInputs);
            Else:
                output = falseSubgraph(falseInputs);
            Emit output

        Condition is a 0D boolean tensor (representing a scalar).
        trueSubgraph represents a network subgraph that is executed when condition is evaluated to True.
        falseSubgraph represents a network subgraph that is executed when condition is evaluated to False.

        The following constraints apply to If-conditionals:
        - Both the trueSubgraph and falseSubgraph must be defined.
        - The number of output tensors in both subgraphs is the same.
        - The type and shape of each output tensor from the true/false subgraphs are the same, except that the shapes are allowed to differ if the condition is a build-time constant.

    """
    name: str
    def add_input(self, input: ITensor) -> IIfConditionalInputLayer:
        """
            Make an input for this if-conditional, based on the given tensor.

            :param input: An input to the conditional that can be used by either or both of the conditional’s subgraphs.
        """
    def add_output(self, true_subgraph_output: ITensor, false_subgraph_output: ITensor) -> IIfConditionalOutputLayer:
        """
            Make an output for this if-conditional, based on the given tensors.

            Each output layer of the if-conditional represents a single output of either the true-subgraph or the
            false-subgraph of the if-conditional, depending on which subgraph was executed.

            :param true_subgraph_output: The output of the subgraph executed when this conditional's condition input evaluates to true.
            :param false_subgraph_output: The output of the subgraph executed when this conditional's condition input evaluates to false.

            :returns: The :class:`IIfConditionalOutputLayer` , or :class:`None` if it could not be created.
        """
    def set_condition(self, condition: ITensor) -> IConditionLayer:
        """
            Set the condition tensor for this If-Conditional construct.

            The ``condition`` tensor must be a 0D data tensor (scalar) with type :class:`bool`.

            :param condition: The condition tensor that will determine which subgraph to execute.

            :returns: The :class:`IConditionLayer` , or :class:`None` if it could not be created.
        """
class IIfConditionalBoundaryLayer(ILayer):
    """

        :ivar conditional: :class:`IIfConditional` associated with this boundary layer.
    """
    @property
    def conditional(self) -> ...:
        ...
class IIfConditionalInputLayer(IIfConditionalBoundaryLayer):
    """
    Describes kinds of if-conditional inputs.
    """
class IIfConditionalOutputLayer(IIfConditionalBoundaryLayer):
    """
    Describes kinds of if-conditional outputs.
    """
class IInt8Calibrator:
    """

        [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization.

        Application-implemented interface for calibration. Calibration is a step performed by the builder when deciding suitable scale factors for 8-bit inference. It must also provide a method for retrieving representative images which the calibration process can use to examine the distribution of activations. It may optionally implement a method for caching the calibration result for reuse on subsequent runs.

        To implement a custom calibrator, ensure that you explicitly instantiate the base class in :func:`__init__` :
        ::

            class MyCalibrator(trt.IInt8Calibrator):
                def __init__(self):
                    trt.IInt8Calibrator.__init__(self)

        :ivar batch_size: :class:`int` The batch size used for calibration batches.
        :ivar algorithm: :class:`CalibrationAlgoType` The algorithm used by this calibrator.
    """
    def __init__(self) -> None:
        ...
    def get_algorithm(self) -> CalibrationAlgoType:
        """
            Get the algorithm used by this calibrator.

            :returns: The algorithm used by this calibrator.
        """
    def get_batch(self, names: collections.abc.Sequence[str]) -> list[int]:
        """
            Get a batch of input for calibration. The batch size of the input must match the batch size returned by :func:`get_batch_size` .

            A possible implementation may look like this:
            ::

                def get_batch(names):
                    try:
                        # Assume self.batches is a generator that provides batch data.
                        data = next(self.batches)
                        # Assume that self.device_input is a device buffer allocated by the constructor.
                        cuda.memcpy_htod(self.device_input, data)
                        return [int(self.device_input)]
                    except StopIteration:
                        # When we're out of batches, we return either [] or None.
                        # This signals to TensorRT that there is no calibration data remaining.
                        return None

            :arg names: The names of the network inputs for each object in the bindings array.

            :returns: A :class:`list` of device memory pointers set to the memory containing each network input data, or an empty :class:`list` if there are no more batches for calibration. You can allocate these device buffers with pycuda, for example, and then cast them to :class:`int` to retrieve the pointer.
        """
    def get_batch_size(self) -> int:
        """
            Get the batch size used for calibration batches.

            :returns: The batch size.
        """
    def read_calibration_cache(self) -> collections.abc.Buffer:
        """
            Load a calibration cache.

            Calibration is potentially expensive, so it can be useful to generate the calibration data once, then use it on subsequent builds
            of the network. The cache includes the regression cutoff and quantile values used to generate it, and will not be used if
            these do not match the settings of the current calibrator. However, the network should also be recalibrated if its structure
            changes, or the input data set changes, and it is the responsibility of the application to ensure this.

            Reading a cache is just like reading any other file in Python. For example, one possible implementation is:
            ::

                def read_calibration_cache(self):
                    # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
                    if os.path.exists(self.cache_file):
                        with open(self.cache_file, "rb") as f:
                            return f.read()

            :returns: A cache object or None if there is no data.
        """
    def write_calibration_cache(self, cache: collections.abc.Buffer) -> None:
        """
            Save a calibration cache.

            Writing a cache is just like writing any other buffer in Python. For example, one possible implementation is:
            ::

                def write_calibration_cache(self, cache):
                    with open(self.cache_file, "wb") as f:
                        f.write(cache)

            :arg cache: The calibration cache to write.
        """
class IInt8EntropyCalibrator(IInt8Calibrator):
    """

        [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization.

        Extends the :class:`IInt8Calibrator` class.

        To implement a custom calibrator, ensure that you explicitly instantiate the base class in :func:`__init__` :
        ::

            class MyCalibrator(trt.IInt8EntropyCalibrator):
                def __init__(self):
                    trt.IInt8EntropyCalibrator.__init__(self)


        This is the Legacy Entropy calibrator. It is less complicated than the legacy calibrator and produces better results.
    """
    def __init__(self) -> None:
        ...
    def get_algorithm(self) -> CalibrationAlgoType:
        """
            Signals that this is the entropy calibrator.

            :returns: :class:`CalibrationAlgoType.ENTROPY_CALIBRATION`
        """
    def get_batch(self, names: collections.abc.Sequence[str]) -> list[int]:
        """
            Get a batch of input for calibration. The batch size of the input must match the batch size returned by :func:`get_batch_size` .

            A possible implementation may look like this:
            ::

                def get_batch(names):
                    try:
                        # Assume self.batches is a generator that provides batch data.
                        data = next(self.batches)
                        # Assume that self.device_input is a device buffer allocated by the constructor.
                        cuda.memcpy_htod(self.device_input, data)
                        return [int(self.device_input)]
                    except StopIteration:
                        # When we're out of batches, we return either [] or None.
                        # This signals to TensorRT that there is no calibration data remaining.
                        return None

            :arg names: The names of the network inputs for each object in the bindings array.

            :returns: A :class:`list` of device memory pointers set to the memory containing each network input data, or an empty :class:`list` if there are no more batches for calibration. You can allocate these device buffers with pycuda, for example, and then cast them to :class:`int` to retrieve the pointer.
        """
    def get_batch_size(self: IInt8Calibrator) -> int:
        """
            Get the batch size used for calibration batches.

            :returns: The batch size.
        """
    def read_calibration_cache(self) -> collections.abc.Buffer:
        """
            Load a calibration cache.

            Calibration is potentially expensive, so it can be useful to generate the calibration data once, then use it on subsequent builds
            of the network. The cache includes the regression cutoff and quantile values used to generate it, and will not be used if
            these do not match the settings of the current calibrator. However, the network should also be recalibrated if its structure
            changes, or the input data set changes, and it is the responsibility of the application to ensure this.

            Reading a cache is just like reading any other file in Python. For example, one possible implementation is:
            ::

                def read_calibration_cache(self):
                    # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
                    if os.path.exists(self.cache_file):
                        with open(self.cache_file, "rb") as f:
                            return f.read()

            :returns: A cache object or None if there is no data.
        """
    def write_calibration_cache(self, cache: collections.abc.Buffer) -> None:
        """
            Save a calibration cache.

            Writing a cache is just like writing any other buffer in Python. For example, one possible implementation is:
            ::

                def write_calibration_cache(self, cache):
                    with open(self.cache_file, "wb") as f:
                        f.write(cache)

            :arg cache: The calibration cache to write.
        """
class IInt8EntropyCalibrator2(IInt8Calibrator):
    """

        [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization.

        Extends the :class:`IInt8Calibrator` class.

        To implement a custom calibrator, ensure that you explicitly instantiate the base class in :func:`__init__` :
        ::

            class MyCalibrator(trt.IInt8EntropyCalibrator2):
                def __init__(self):
                    trt.IInt8EntropyCalibrator2.__init__(self)

        This is the preferred calibrator. This is the required calibrator for DLA, as it supports per activation tensor scaling.
    """
    def __init__(self) -> None:
        ...
    def get_algorithm(self) -> CalibrationAlgoType:
        """
            Signals that this is the entropy calibrator 2.

            :returns: :class:`CalibrationAlgoType.ENTROPY_CALIBRATION_2`
        """
    def get_batch(self, names: collections.abc.Sequence[str]) -> list[int]:
        """
            Get a batch of input for calibration. The batch size of the input must match the batch size returned by :func:`get_batch_size` .

            A possible implementation may look like this:
            ::

                def get_batch(names):
                    try:
                        # Assume self.batches is a generator that provides batch data.
                        data = next(self.batches)
                        # Assume that self.device_input is a device buffer allocated by the constructor.
                        cuda.memcpy_htod(self.device_input, data)
                        return [int(self.device_input)]
                    except StopIteration:
                        # When we're out of batches, we return either [] or None.
                        # This signals to TensorRT that there is no calibration data remaining.
                        return None

            :arg names: The names of the network inputs for each object in the bindings array.

            :returns: A :class:`list` of device memory pointers set to the memory containing each network input data, or an empty :class:`list` if there are no more batches for calibration. You can allocate these device buffers with pycuda, for example, and then cast them to :class:`int` to retrieve the pointer.
        """
    def get_batch_size(self: IInt8Calibrator) -> int:
        """
            Get the batch size used for calibration batches.

            :returns: The batch size.
        """
    def read_calibration_cache(self) -> collections.abc.Buffer:
        """
            Load a calibration cache.

            Calibration is potentially expensive, so it can be useful to generate the calibration data once, then use it on subsequent builds
            of the network. The cache includes the regression cutoff and quantile values used to generate it, and will not be used if
            these do not match the settings of the current calibrator. However, the network should also be recalibrated if its structure
            changes, or the input data set changes, and it is the responsibility of the application to ensure this.

            Reading a cache is just like reading any other file in Python. For example, one possible implementation is:
            ::

                def read_calibration_cache(self):
                    # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
                    if os.path.exists(self.cache_file):
                        with open(self.cache_file, "rb") as f:
                            return f.read()

            :returns: A cache object or None if there is no data.
        """
    def write_calibration_cache(self, cache: collections.abc.Buffer) -> None:
        """
            Save a calibration cache.

            Writing a cache is just like writing any other buffer in Python. For example, one possible implementation is:
            ::

                def write_calibration_cache(self, cache):
                    with open(self.cache_file, "wb") as f:
                        f.write(cache)

            :arg cache: The calibration cache to write.
        """
class IInt8LegacyCalibrator(IInt8Calibrator):
    """

        [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization.

        Extends the :class:`IInt8Calibrator` class.
        This calibrator requires user parameterization, and is provided as a fallback option if the other calibrators yield poor results.

        To implement a custom calibrator, ensure that you explicitly instantiate the base class in :func:`__init__` :
        ::

            class MyCalibrator(trt.IInt8LegacyCalibrator):
                def __init__(self):
                    trt.IInt8LegacyCalibrator.__init__(self)

        :ivar quantile: :class:`float` The quantile (between 0 and 1) that will be used to select the region maximum when the quantile method is in use. See the user guide for more details on how the quantile is used.
        :ivar regression_cutoff: :class:`float` The fraction (between 0 and 1) of the maximum used to define the regression cutoff when using regression to determine the region maximum. See the user guide for more details on how the regression cutoff is used
    """
    def __init__(self) -> None:
        ...
    def get_algorithm(self) -> CalibrationAlgoType:
        """
            Signals that this is the legacy calibrator.

            :returns: :class:`CalibrationAlgoType.LEGACY_CALIBRATION`
        """
    def get_batch(self, names: collections.abc.Sequence[str]) -> list[int]:
        """
            Get a batch of input for calibration. The batch size of the input must match the batch size returned by :func:`get_batch_size` .

            A possible implementation may look like this:
            ::

                def get_batch(names):
                    try:
                        # Assume self.batches is a generator that provides batch data.
                        data = next(self.batches)
                        # Assume that self.device_input is a device buffer allocated by the constructor.
                        cuda.memcpy_htod(self.device_input, data)
                        return [int(self.device_input)]
                    except StopIteration:
                        # When we're out of batches, we return either [] or None.
                        # This signals to TensorRT that there is no calibration data remaining.
                        return None

            :arg names: The names of the network inputs for each object in the bindings array.

            :returns: A :class:`list` of device memory pointers set to the memory containing each network input data, or an empty :class:`list` if there are no more batches for calibration. You can allocate these device buffers with pycuda, for example, and then cast them to :class:`int` to retrieve the pointer.
        """
    def get_batch_size(self: IInt8Calibrator) -> int:
        """
            Get the batch size used for calibration batches.

            :returns: The batch size.
        """
    def read_calibration_cache(self) -> collections.abc.Buffer:
        """
            Load a calibration cache.

            Calibration is potentially expensive, so it can be useful to generate the calibration data once, then use it on subsequent builds
            of the network. The cache includes the regression cutoff and quantile values used to generate it, and will not be used if
            these do not match the settings of the current calibrator. However, the network should also be recalibrated if its structure
            changes, or the input data set changes, and it is the responsibility of the application to ensure this.

            Reading a cache is just like reading any other file in Python. For example, one possible implementation is:
            ::

                def read_calibration_cache(self):
                    # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
                    if os.path.exists(self.cache_file):
                        with open(self.cache_file, "rb") as f:
                            return f.read()

            :returns: A cache object or None if there is no data.
        """
    def write_calibration_cache(self, cache: collections.abc.Buffer) -> None:
        """
            Save a calibration cache.

            Writing a cache is just like writing any other buffer in Python. For example, one possible implementation is:
            ::

                def write_calibration_cache(self, cache):
                    with open(self.cache_file, "wb") as f:
                        f.write(cache)

            :arg cache: The calibration cache to write.
        """
class IInt8MinMaxCalibrator(IInt8Calibrator):
    """

        [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization.

        Extends the :class:`IInt8Calibrator` class.

        To implement a custom calibrator, ensure that you explicitly instantiate the base class in :func:`__init__` :
        ::

            class MyCalibrator(trt.IInt8MinMaxCalibrator):
                def __init__(self):
                    trt.IInt8MinMaxCalibrator.__init__(self)

        This is the preferred calibrator for NLP tasks for all backends. It supports per activation tensor scaling.
    """
    def __init__(self) -> None:
        ...
    def get_algorithm(self) -> CalibrationAlgoType:
        """
            Signals that this is the minmax calibrator.

            :returns: :class:`CalibrationAlgoType.MINMAX_CALIBRATION`
        """
    def get_batch(self, names: collections.abc.Sequence[str]) -> list[int]:
        """
            Get a batch of input for calibration. The batch size of the input must match the batch size returned by :func:`get_batch_size` .

            A possible implementation may look like this:
            ::

                def get_batch(names):
                    try:
                        # Assume self.batches is a generator that provides batch data.
                        data = next(self.batches)
                        # Assume that self.device_input is a device buffer allocated by the constructor.
                        cuda.memcpy_htod(self.device_input, data)
                        return [int(self.device_input)]
                    except StopIteration:
                        # When we're out of batches, we return either [] or None.
                        # This signals to TensorRT that there is no calibration data remaining.
                        return None

            :arg names: The names of the network inputs for each object in the bindings array.

            :returns: A :class:`list` of device memory pointers set to the memory containing each network input data, or an empty :class:`list` if there are no more batches for calibration. You can allocate these device buffers with pycuda, for example, and then cast them to :class:`int` to retrieve the pointer.
        """
    def get_batch_size(self: IInt8Calibrator) -> int:
        """
            Get the batch size used for calibration batches.

            :returns: The batch size.
        """
    def read_calibration_cache(self) -> collections.abc.Buffer:
        """
            Load a calibration cache.

            Calibration is potentially expensive, so it can be useful to generate the calibration data once, then use it on subsequent builds
            of the network. The cache includes the regression cutoff and quantile values used to generate it, and will not be used if
            these do not match the settings of the current calibrator. However, the network should also be recalibrated if its structure
            changes, or the input data set changes, and it is the responsibility of the application to ensure this.

            Reading a cache is just like reading any other file in Python. For example, one possible implementation is:
            ::

                def read_calibration_cache(self):
                    # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
                    if os.path.exists(self.cache_file):
                        with open(self.cache_file, "rb") as f:
                            return f.read()

            :returns: A cache object or None if there is no data.
        """
    def write_calibration_cache(self, cache: collections.abc.Buffer) -> None:
        """
            Save a calibration cache.

            Writing a cache is just like writing any other buffer in Python. For example, one possible implementation is:
            ::

                def write_calibration_cache(self, cache):
                    with open(self.cache_file, "wb") as f:
                        f.write(cache)

            :arg cache: The calibration cache to write.
        """
class IIteratorLayer(ILoopBoundaryLayer):
    """

        :ivar axis: The axis to iterate over
        :ivar reverse: For reverse=false, the layer is equivalent to add_gather(tensor, I, 0) where I is a
            scalar tensor containing the loop iteration number.
            For reverse=true, the layer is equivalent to add_gather(tensor, M-1-I, 0) where M is the trip count
            computed from TripLimits of kind ``COUNT``.
            The default is reverse=false.
    """
    reverse: bool
    @property
    def axis(self) -> int:
        ...
    @axis.setter
    def axis(self, arg1: typing.SupportsInt) -> None:
        ...
class IKVCacheUpdateLayer(ILayer):
    """

        A KVCacheUpdate layer in a :class:`INetworkDefinition` .

        This layer caches Key (K) or Value (V) tensors for reuse in subsequent attention computations.
        Users provide newly computed K/V values, and the layer will output the updated K/V cache.
        The write_indices input specifies where to write K/V updates for each sequence in the batch.
        Separate KVCacheUpdate layers should be used for K and V.

        :ivar cache_mode: :class:`KVCacheMode` The mode of the KVCacheUpdate layer.
    """
    cache_mode: KVCacheMode
    def set_input(self: ILayer, index: typing.SupportsInt, tensor: ITensor) -> None:
        """
            Sets the input tensor specified by the given index.

            The indices are as follows:

            =====   ==================================================================================
            Index   Description
            =====   ==================================================================================
                0     cache.
                1     update.
                2     write_indices.
            =====   ==================================================================================

            :arg index: The index of the input tensor.
            :arg tensor: The input tensor.
        """
class ILRNLayer(ILayer):
    """

        A LRN layer in an :class:`INetworkDefinition` . The output size is the same as the input size.

        :ivar window_size: :class:`int` The LRN window size. The window size must be odd and in the range of [1, 15].
        :ivar alpha: :class:`float` The LRN alpha value. The valid range is [-1e20, 1e20].
        :ivar beta: :class:`float` The LRN beta value. The valid range is [0.01, 1e5f].
        :ivar k: :class:`float` The LRN K value. The valid range is [1e-5, 1e10].
    """
    @property
    def alpha(self) -> float:
        ...
    @alpha.setter
    def alpha(self, arg1: typing.SupportsFloat) -> None:
        ...
    @property
    def beta(self) -> float:
        ...
    @beta.setter
    def beta(self, arg1: typing.SupportsFloat) -> None:
        ...
    @property
    def k(self) -> float:
        ...
    @k.setter
    def k(self, arg1: typing.SupportsFloat) -> None:
        ...
    @property
    def window_size(self) -> int:
        ...
    @window_size.setter
    def window_size(self, arg1: typing.SupportsInt) -> None:
        ...
class ILayer:
    """

        Base class for all layer classes in an :class:`INetworkDefinition` .

        :ivar name: :class:`str` The name of the layer.
        :ivar metadata: :class:`str` The per-layer metadata.
        :ivar num_ranks: :class:`int` The number of ranks for multi-device execution (default: 1).
        :ivar type: :class:`LayerType` The type of the layer.
        :ivar num_inputs: :class:`int` The number of inputs of the layer.
        :ivar num_outputs: :class:`int` The number of outputs of the layer.
        :ivar precision: :class:`DataType` The computation precision.
        :ivar precision_is_set: :class:`bool` Whether the precision is set or not.
    """
    metadata: str
    name: str
    precision: DataType
    def get_input(self, index: typing.SupportsInt) -> ITensor:
        """
            Get the layer input corresponding to the given index.

            :arg index: The index of the input tensor.

            :returns: The input tensor, or :class:`None` if the index is out of range.
        """
    def get_output(self, index: typing.SupportsInt) -> ITensor:
        """
            Get the layer output corresponding to the given index.

            :arg index: The index of the output tensor.

            :returns: The output tensor, or :class:`None` if the index is out of range.
        """
    def get_output_type(self, index: typing.SupportsInt) -> DataType:
        """
            Get the output type of the layer.

            :arg index: The index of the output tensor.

            :returns: The output precision. Default : DataType.FLOAT.
        """
    def output_type_is_set(self, index: typing.SupportsInt) -> bool:
        """
            [DEPRECATED] Deprecated in TensorRT 10.12. Superseded by strong typing.
            Whether the output type has been set for this layer.

            :arg index: The index of the output.

            :returns: Whether the output type has been explicitly set.
        """
    def reset_output_type(self, index: typing.SupportsInt) -> None:
        """
            [DEPRECATED] Deprecated in TensorRT 10.12. Superseded by strong typing.
            Reset output type of this layer.

            :arg index: The index of the output.
        """
    def reset_precision(self) -> None:
        """
            [DEPRECATED] Deprecated in TensorRT 10.12. Superseded by strong typing.
            Reset the computation precision of the layer.
        """
    def set_input(self, index: typing.SupportsInt, tensor: ITensor) -> None:
        """
            Set the layer input corresponding to the given index.

            :arg index: The index of the input tensor.
            :arg tensor: The input tensor.
        """
    def set_output_type(self, index: typing.SupportsInt, dtype: DataType) -> None:
        """
            [DEPRECATED] Deprecated in TensorRT 10.12. Superseded by strong typing.
            Constraint layer to generate output data with given type.
            Note that this method cannot be used to set the data type
            of the second output tensor of the topK layer. The data
            type of the second output tensor of the topK layer is always :class:`int32` .

            :arg index: The index of the output tensor to set the type.
            :arg dtype: DataType of the output.
        """
    @property
    def num_inputs(self) -> int:
        ...
    @property
    def num_outputs(self) -> int:
        ...
    @property
    def num_ranks(self) -> int:
        """
            :class:`int` The number of ranks for multi-device execution.

            Currently, setting num_ranks > 1 via ILayer is only allowed for IDistCollectiveLayer, which uses it to
            determine output shape for kALL_GATHER and kREDUCE_SCATTER operations.

            For attention layers, use IAttention.num_ranks instead.

            Default value is 1.
        """
    @num_ranks.setter
    def num_ranks(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def precision_is_set(self) -> bool:
        ...
    @property
    def type(self) -> LayerType:
        ...
class ILogger:
    """

    Abstract base Logger class for the :class:`Builder`, :class:`ICudaEngine` and :class:`Runtime` .

    To implement a custom logger, ensure that you explicitly instantiate the base class in :func:`__init__` :
    ::

        class MyLogger(trt.ILogger):
            def __init__(self):
                trt.ILogger.__init__(self)

            def log(self, severity, msg):
                ... # Your implementation here


    :arg min_severity: The initial minimum severity of this Logger.

    :ivar min_severity: :class:`Logger.Severity` This minimum required severity of messages for the logger to log them.

    The logger used to create an instance of IBuilder, IRuntime or IRefitter is used for all objects created through that interface.
    The logger should be valid until all objects created are released.
    """
    class Severity:
        """

            Indicates the severity of a message. The values in this enum are also accessible in the :class:`ILogger` directly.
            For example, ``tensorrt.ILogger.INFO`` corresponds to ``tensorrt.ILogger.Severity.INFO`` .


        Members:

          INTERNAL_ERROR :
            Represents an internal error. Execution is unrecoverable.


          ERROR :
            Represents an application error.


          WARNING :
            Represents an application error that TensorRT has recovered from or fallen back to a default.


          INFO :
            Represents informational messages.


          VERBOSE :
            Verbose messages with debugging information.
        """
        ERROR: typing.ClassVar[ILogger.Severity]  # value = <Severity.ERROR: 1>
        INFO: typing.ClassVar[ILogger.Severity]  # value = <Severity.INFO: 3>
        INTERNAL_ERROR: typing.ClassVar[ILogger.Severity]  # value = <Severity.INTERNAL_ERROR: 0>
        VERBOSE: typing.ClassVar[ILogger.Severity]  # value = <Severity.VERBOSE: 4>
        WARNING: typing.ClassVar[ILogger.Severity]  # value = <Severity.WARNING: 2>
        __members__: typing.ClassVar[dict[str, ILogger.Severity]]  # value = {'INTERNAL_ERROR': <Severity.INTERNAL_ERROR: 0>, 'ERROR': <Severity.ERROR: 1>, 'WARNING': <Severity.WARNING: 2>, 'INFO': <Severity.INFO: 3>, 'VERBOSE': <Severity.VERBOSE: 4>}
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __ge__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __gt__(self, other: typing.Any) -> bool:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self) -> int:
            ...
        def __init__(self, value: typing.SupportsInt) -> None:
            ...
        def __int__(self) -> int:
            ...
        def __le__(self, other: typing.Any) -> bool:
            ...
        def __lt__(self, other: typing.Any) -> bool:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self, state: typing.SupportsInt) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    ERROR: typing.ClassVar[ILogger.Severity]  # value = <Severity.ERROR: 1>
    INFO: typing.ClassVar[ILogger.Severity]  # value = <Severity.INFO: 3>
    INTERNAL_ERROR: typing.ClassVar[ILogger.Severity]  # value = <Severity.INTERNAL_ERROR: 0>
    VERBOSE: typing.ClassVar[ILogger.Severity]  # value = <Severity.VERBOSE: 4>
    WARNING: typing.ClassVar[ILogger.Severity]  # value = <Severity.WARNING: 2>
    @staticmethod
    def __enter__(this):
        ...
    @staticmethod
    def __exit__(this, exc_type, exc_value, traceback):
        ...
    def __init__(self) -> None:
        ...
    def log(self, severity: ILogger.Severity, msg: str) -> None:
        """
        Logs a message to `stderr` . This function must be overriden by a derived class.

        :arg severity: The severity of the message.
        :arg msg: The log message.
        """
class ILoop:
    """

        Helper for creating a recurrent subgraph.

        :ivar name: The name of the loop. The name is used in error diagnostics.
    """
    name: str
    def add_iterator(self, tensor: ITensor, axis: typing.SupportsInt = 0, reverse: bool = False) -> IIteratorLayer:
        """
            Return layer that subscripts tensor by loop iteration.

            For reverse=false, this is equivalent to add_gather(tensor, I, 0) where I is a
            scalar tensor containing the loop iteration number.
            For reverse=true, this is equivalent to add_gather(tensor, M-1-I, 0) where M is the trip count
            computed from TripLimits of kind ``COUNT``.

            :param tensor: The tensor to iterate over.
            :param axis: The axis along which to iterate.
            :param reverse: Whether to iterate in the reverse direction.

            :returns: The :class:`IIteratorLayer` , or :class:`None` if it could not be created.
        """
    def add_loop_output(self, tensor: ITensor, kind: LoopOutput, axis: typing.SupportsInt = 0) -> ILoopOutputLayer:
        """
            Make an output for this loop, based on the given tensor.

            If ``kind`` is ``CONCATENATE`` or ``REVERSE``, a second input specifying the
            concatenation dimension must be added via method :func:`ILoopOutputLayer.set_input` .

            :param kind: The kind of loop output. See :class:`LoopOutput`
            :param axis: The axis for concatenation (if using ``kind`` of ``CONCATENATE`` or ``REVERSE``).

            :returns: The added :class:`ILoopOutputLayer` , or :class:`None` if it could not be created.
        """
    def add_recurrence(self, initial_value: ITensor) -> IRecurrenceLayer:
        """
            Create a recurrence layer for this loop with initial_value as its first input.

            :param initial_value: The initial value of the recurrence layer.

            :returns: The added :class:`IRecurrenceLayer` , or :class:`None` if it could not be created.
        """
    def add_trip_limit(self, tensor: ITensor, kind: TripLimit) -> ITripLimitLayer:
        """
            Add a trip-count limiter, based on the given tensor.

            There may be at most one ``COUNT`` and one ``WHILE`` limiter for a loop.
            When both trip limits exist, the loop exits when the
            count is reached or condition is falsified.
            It is an error to not add at least one trip limiter.

            For ``WHILE``, the input tensor must be the output of a subgraph that contains
            only layers that are not :class:`ITripLimitLayer` , :class:`IIteratorLayer` or :class:`ILoopOutputLayer` .
            Any :class:`IRecurrenceLayer` s in the subgraph must belong to the same loop as the
            :class:`ITripLimitLayer` . A trivial example of this rule is that the input to the ``WHILE``
            is the output of an :class:`IRecurrenceLayer` for the same loop.


            :param tensor: The input tensor. Must be available before the loop starts.
            :param kind: The kind of trip limit. See :class:`TripLimit`

            :returns: The added :class:`ITripLimitLayer` , or :class:`None` if it could not be created.
        """
class ILoopBoundaryLayer(ILayer):
    """

        :ivar loop: :class:`ILoop` associated with this boundary layer.
    """
    @property
    def loop(self) -> ...:
        ...
class ILoopOutputLayer(ILoopBoundaryLayer):
    """

        An :class:`ILoopOutputLayer` is the sole way to get output from a loop.

        The first input tensor must be defined inside the loop; the output tensor is outside the loop.
        The second input tensor, if present, must be defined outside the loop.

        If :attr:`kind` is ``LAST_VALUE``, a single input must be provided.

        If :attr:`kind` is ``CONCATENATE`` or ``REVERSE``, a second input must be provided.
        The second input must be a scalar “shape tensor”, defined before the loop commences,
        that specifies the concatenation length of the output.

        The output tensor has j more dimensions than the input tensor, where
        j == 0 if :attr:`kind` is ``LAST_VALUE``
        j == 1 if :attr:`kind` is ``CONCATENATE`` or ``REVERSE``.

        :ivar axis: The contenation axis. Ignored if :attr:`kind` is ``LAST_VALUE``.
            For example, if the input tensor has dimensions [b,c,d],
            and :attr:`kind` is  ``CONCATENATE``, the output has four dimensions.
            Let a be the value of the second input.
            axis=0 causes the output to have dimensions [a,b,c,d].
            axis=1 causes the output to have dimensions [b,a,c,d].
            axis=2 causes the output to have dimensions [b,c,a,d].
            axis=3 causes the output to have dimensions [b,c,d,a].
            Default is axis is 0.
        :ivar kind: The kind of loop output. See :class:`LoopOutput`
    """
    def set_input(self: ILayer, index: typing.SupportsInt, tensor: ITensor) -> None:
        """
            Like :func:`ILayer.set_input`, but additionally works if index==1, :attr:`num_inputs`==1, in which case :attr:`num_inputs` changes to 2.
        """
    @property
    def axis(self) -> int:
        ...
    @axis.setter
    def axis(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def kind(self) -> LoopOutput:
        ...
class IMatrixMultiplyLayer(ILayer):
    """

        A matrix multiply layer in an :class:`INetworkDefinition` .

        Let A be op(getInput(0)) and B be op(getInput(1)) where
        op(x) denotes the corresponding MatrixOperation.

        When A and B are matrices or vectors, computes the inner product A * B:

        |   matrix * matrix -> matrix
        |   matrix * vector -> vector
        |   vector * matrix -> vector
        |   vector * vector -> scalar

        Inputs of higher rank are treated as collections of matrices or vectors.
        The output will be a corresponding collection of matrices, vectors, or scalars.

        :ivar op0: :class:`MatrixOperation` How to treat the first input.
        :ivar op1: :class:`MatrixOperation` How to treat the second input.
    """
    op0: MatrixOperation
    op1: MatrixOperation
class IMoELayer(ILayer):
    """

        A MoE layer in :class:`INetworkDefinition`.

        :ivar activation_type: :class:`MoEActType` Specifies the activation type for the MoE layer.
        :ivar quantization_to_type: :class:`DataType` Specifies the quantization type for the MoE layer.
        :ivar quantization_block_shape: :class:`Dims` Specifies the quantization block shape for the MoE layer.
        :ivar dyn_q_output_scale_type: :class:`DataType` Specifies the dynamic quantization output scale type for the MoE layer.
        :ivar swiglu_param_limit: :class:`float` Specifies the swiglu parameter limit for the MoE layer.
        :ivar swiglu_param_alpha: :class:`float` Specifies the swiglu parameter alpha for the MoE layer.
        :ivar swiglu_param_beta: :class:`float` Specifies the swiglu parameter beta for the MoE layer.
    """
    activation_type: MoEActType
    dyn_q_output_scale_type: DataType
    quantization_block_shape: Dims
    quantization_to_type: DataType
    def set_gated_biases(self, fc_gate_biases: ITensor, fc_up_biases: ITensor, fc_down_biases: ITensor) -> None:
        """
            Set the gated biases for the MoE layer.
            :arg fc_gate_biases: The biases for the gate-projection layer of all experts in MoE.
            :arg fc_up_biases: The biases for the up-projection layer of all experts in MoE.
            :arg fc_down_biases: The biases for the down-projection layer of all experts in MoE.
        """
    def set_gated_weights(self, fc_gate_weights: ITensor, fc_up_weights: ITensor, fc_down_weights: ITensor, activation_type: MoEActType) -> None:
        """
            Set the gated weights for the MoE layer.
            :arg fc_gate_weights: The weights for the gate-projection layer of all experts in MoE.
            :arg fc_up_weights: The weights for the up-projection layer of all experts in MoE.
            :arg fc_down_weights: The weights for the down-projection layer of all experts in MoE.
            :arg activation_type: The activation type for the MoE layer.
        """
    def set_input(self, index: typing.SupportsInt, tensor: ITensor) -> None:
        """
            Set the input tensor specified by the given index.

            :arg index: The index of the input tensor.
            :arg tensor: The input tensor.

            The indices are as follows:

            Input 0: hidden_states: the hidden states tensor.
            Input 1: selected_experts_for_tokens: the selected experts for tokens tensor.
            Input 2: scores_for_selected_experts: the scores for selected experts tensor.
        """
    def set_quantization_dynamic_dbl_q(self, fc_down_activation_dbl_q_scale: ITensor, data_type: DataType, block_shape: Dims, dyn_q_output_scale_type: DataType) -> None:
        """
            Set the quantization dynamic double quantization for the MoE layer.
            :arg fc_down_activation_dbl_q_scale: The down activation double quantization scale tensor.
            :arg data_type: The data type for the quantization.
            :arg block_shape: The block shape for the quantization.
            :arg dyn_q_output_scale_type: The dynamic quantization output scale type.
        """
    def set_quantization_static(self, fc_down_activation_scale: ITensor, data_type: DataType) -> None:
        """
            Set the quantization static for the MoE layer.
            :arg fc_down_activation_scale: The down activation scale tensor.
            :arg data_type: The data type for the quantization.
        """
    def set_swiglu_params(self, limit: typing.SupportsFloat, alpha: typing.SupportsFloat, beta: typing.SupportsFloat) -> None:
        """
            Set the swiglu parameters for the MoE layer.
            :arg limit: The limit for the swiglu parameters.
            :arg alpha: The alpha for the swiglu parameters.
            :arg beta: The beta for the swiglu parameters.
        """
    @property
    def swiglu_param_alpha(self) -> float:
        ...
    @swiglu_param_alpha.setter
    def swiglu_param_alpha(self, arg1: typing.SupportsFloat) -> None:
        ...
    @property
    def swiglu_param_beta(self) -> float:
        ...
    @swiglu_param_beta.setter
    def swiglu_param_beta(self, arg1: typing.SupportsFloat) -> None:
        ...
    @property
    def swiglu_param_limit(self) -> float:
        ...
    @swiglu_param_limit.setter
    def swiglu_param_limit(self, arg1: typing.SupportsFloat) -> None:
        ...
class INMSLayer(ILayer):
    """

        A non-maximum suppression layer in an :class:`INetworkDefinition` .

        Boxes: The input boxes tensor to the layer.
        This tensor contains the input bounding boxes. It is a linear tensor of type ``float32`` or ``float16``.
        It has shape [batchSize, numInputBoundingBoxes, numClasses, 4] if the boxes are per class, or
        [batchSize, numInputBoundingBoxes, 4] if the same boxes are to be used for each class.

        Scores: The input scores tensor to the layer.
        This tensor contains the per-box scores. It is a linear tensor of the same type as the boxes tensor.
        It has shape [batchSize, numInputBoundingBoxes, numClasses].

        MaxOutputBoxesPerClass: The input maxOutputBoxesPerClass tensor to the layer.
        This tensor contains the maximum number of output boxes per batch item per class.
        It is a scalar (0D tensor) of type ``int32``.

        IoUThreshold is the maximum IoU for selected boxes.
        It is a scalar (0D tensor) of type ``float32`` in the range [0.0, 1.0].
        It is an optional input with default 0.0.
        Use :func:`set_input` to add this optional tensor.

        ScoreThreshold is the value that a box score must exceed in order to be selected.
        It is a scalar (0D tensor) of type ``float32``. It is an optional input with default 0.0.
        Use :func:`set_input` to add this optional tensor.

        The SelectedIndices output tensor contains the indices of the selected boxes.
        It is a linear tensor of type ``int32`` or ``int64``. It has shape [NumOutputBoxes, 3].]
        Each row contains a (batchIndex, classIndex, boxIndex) tuple.
        The output boxes are sorted in order of increasing batchIndex and then in order of decreasing score within each batchIndex.
        For each batchIndex, the ordering of output boxes with the same score is unspecified.
        If MaxOutputBoxesPerClass is a constant input, the maximum number of output boxes is
        batchSize * numClasses * min(numInputBoundingBoxes, MaxOutputBoxesPerClass).
        Otherwise, the maximum number of output boxes is batchSize * numClasses * numInputBoundingBoxes.
        The maximum number of output boxes is used to determine the upper-bound on allocated memory for this output tensor.

        The NumOutputBoxes output tensor contains the number of output boxes in selectedIndices.
        It is a scalar (0D tensor) of type ``int32``.

        The NMS algorithm iterates through a set of bounding boxes and their confidence scores,
        in decreasing order of score. Boxes are selected if their score is above a given threshold,
        and their intersection-over-union (IoU) with previously selected boxes is less than or equal
        to a given threshold.
        This layer implements NMS per batch item and per class.

        For each batch item, the ordering of candidate bounding boxes with the same score is unspecified.

        :ivar bounding_box_format: :class:`BoundingBoxFormat` The bounding box format used by the layer. Default is CORNER_PAIRS.
        :ivar topk_box_limit: :class:`int` The maximum number of filtered boxes considered for selection. Default is 2000 for SM 5.3 and 6.2 devices, and 5000 otherwise. The TopK box limit must be less than or equal to {2000 for SM 5.3 and 6.2 devices, 5000 otherwise}.
        :ivar indices_type: :class:`DataType` The specified data type of the output indices tensor. Must be tensorrt.int32 or tensorrt.int64.
    """
    bounding_box_format: BoundingBoxFormat
    indices_type: DataType
    def set_input(self: ILayer, index: typing.SupportsInt, tensor: ITensor) -> None:
        """
            Sets the input tensor for the given index.
            The indices are as follows:

            ======= ========================================================================
             Index   Description
            ======= ========================================================================
                0     The required Boxes tensor.
                1     The required Scores tensor.
                2     The required MaxOutputBoxesPerClass tensor.
                3     The optional IoUThreshold tensor.
                4     The optional ScoreThreshold tensor.
            ======= ========================================================================

            If this function is called for an index greater or equal to :attr:`num_inputs`,
            then afterwards :attr:`num_inputs` returns index + 1, and any missing intervening
            inputs are set to null. Note that only optional inputs can be missing.

            :arg index: The index of the input tensor.
            :arg tensor: The input tensor.
        """
    @property
    def topk_box_limit(self) -> int:
        ...
    @topk_box_limit.setter
    def topk_box_limit(self, arg1: typing.SupportsInt) -> None:
        ...
class INetworkDefinition:
    """

        Represents a TensorRT Network from which the Builder can build an Engine

        :ivar num_layers: :class:`int` The number of layers in the network.
        :ivar num_inputs: :class:`int` The number of inputs of the network.
        :ivar num_outputs: :class:`int` The number of outputs of the network.
        :ivar name: :class:`str` The name of the network. This is used so that it can be associated with a built engine. The name must be at most 128 characters in length. TensorRT makes no use of this string except storing it as part of the engine so that it may be retrieved at runtime. A name unique to the builder will be generated by default.
        :ivar has_implicit_batch_dimension: :class:`bool` [DEPRECATED] Deprecated in TensorRT 10.0. Always flase since the implicit batch dimensions support has been removed.
        :ivar error_recorder: :class:`IErrorRecorder` Application-implemented error reporting interface for TensorRT objects.
        :flags: :int: A bitset of the ``NetworkDefinitionCreationFlag`` s set for this network.
    """
    error_recorder: ...
    name: str
    @staticmethod
    def __enter__(this):
        ...
    @staticmethod
    def __exit__(this, exc_type, exc_value, traceback):
        """

            Context managers are deprecated and have no effect. Objects are automatically freed when
            the reference count reaches 0.

        """
    def __del__(self) -> None:
        ...
    def __getitem__(self, arg0: typing.SupportsInt) -> ILayer:
        ...
    def __len__(self) -> int:
        ...
    def add_activation(self, input: ITensor, type: ActivationType) -> IActivationLayer:
        """
            Add an activation layer to the network.
            See :class:`IActivationLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg type: The type of activation function to apply.

            :returns: The new activation layer, or :class:`None` if it could not be created.
        """
    def add_assertion(self, condition: ITensor, message: str) -> IAssertionLayer:
        """
            Add a assertion layer.
            See :class:`IAssertionLayer` for more information.

            :arg condition: The condition tensor to the layer.
            :arg message: The message to print if the assertion fails.

            :returns: The new assertion layer, or :class:`None` if it could not be created.
        """
    def add_attention(self, query: ITensor, key: ITensor, value: ITensor, norm_op: AttentionNormalizationOp, causal: bool) -> IAttention:
        """
            Add an attention to the network.
            See :class:`IAttention` for more information.

            :arg query: The 4d query input tensor to the attention.
            :arg key: The 4d key input tensor to the attention.
            :arg value: The 4d value input tensor to the attention.
            :arg normOp: The normalization operation to perform.
            :arg causal: The boolean that specifies whether an attention will run casual inference.

            :returns: The new Attention, or :class:`None` if it could not be created.
        """
    def add_cast(self, input: ITensor, to_type: DataType) -> ICastLayer:
        """
            Add a cast layer.
            See :class:`ICastLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg to_type: The data type the output tensor should be cast into.

            :returns: The new cast layer, or :class:`None` if it could not be created.
        """
    def add_concatenation(self, inputs: collections.abc.Sequence[ITensor]) -> IConcatenationLayer:
        """
            Add a concatenation layer to the network. Note that all tensors must have the same dimension except for the Channel dimension.
            See :class:`IConcatenationLayer` for more information.

            :arg inputs: The input tensors to the layer.

            :returns: The new concatenation layer, or :class:`None` if it could not be created.
        """
    def add_constant(self, shape: Dims, weights: Weights) -> IConstantLayer:
        """
            Add a constant layer to the network.
            See :class:`IConstantLayer` for more information.

            :arg shape: The shape of the constant.
            :arg weights: The constant value, represented as weights.

            :returns: The new constant layer, or :class:`None` if it could not be created.
        """
    def add_convolution_nd(self, input: ITensor, num_output_maps: typing.SupportsInt, kernel_shape: Dims, kernel: Weights, bias: Weights = None) -> IConvolutionLayer:
        """
            Add a multi-dimension convolution layer to the network.
            See :class:`IConvolutionLayer` for more information.

            :arg input: The input tensor to the convolution.
            :arg num_output_maps: The number of output feature maps for the convolution.
            :arg kernel_shape: The dimensions of the convolution kernel.
            :arg kernel: The kernel weights for the convolution.
            :arg bias: The optional bias weights for the convolution.

            :returns: The new convolution layer, or :class:`None` if it could not be created.
        """
    def add_cumulative(self, input: ITensor, axis: ITensor, op: CumulativeOperation, exclusive: bool, reverse: bool) -> ICumulativeLayer:
        """
            Add a cumulative layer to the network.
            See :class:`ICumulativeLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg axis: The axis tensor to apply the cumulative operation on. Currently, it must be a build-time constant 0-D shape tensor.
            :arg op: The reduction operation to perform.
            :arg exclusive: The boolean that specifies whether it is an exclusive cumulative or inclusive cumulative.
            :arg reverse: The boolean that specifies whether the cumulative should be applied backward.

            :returns: The new cumulative layer, or :class:`None` if it could not be created.
        """
    def add_deconvolution_nd(self, input: ITensor, num_output_maps: typing.SupportsInt, kernel_shape: Dims, kernel: Weights, bias: Weights = None) -> IDeconvolutionLayer:
        """
            Add a multi-dimension deconvolution layer to the network.
            See :class:`IDeconvolutionLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg num_output_maps: The number of output feature maps.
            :arg kernel_shape: The dimensions of the convolution kernel.
            :arg kernel: The kernel weights for the convolution.
            :arg bias: The optional bias weights for the convolution.

            :returns: The new deconvolution layer, or :class:`None` if it could not be created.
        """
    @typing.overload
    def add_dequantize(self, input: ITensor, scale: ITensor) -> IDequantizeLayer:
        """
            Add a dequantization layer to the network.
            See :class:`IDequantizeLayer` for more information.

            :arg input: A tensor to quantize.
            :arg scale: A tensor with the scale coefficients.
            :arg output_type: The datatype of the output tensor. Specifying output_type is optional (default value tensorrt.float32).

            :returns: The new dequantization layer, or :class:`None` if it could not be created.
        """
    @typing.overload
    def add_dequantize(self, input: ITensor, scale: ITensor, output_type: DataType) -> IDequantizeLayer:
        """
            Add a dequantization layer to the network.
            See :class:`IDequantizeLayer` for more information.

            :arg input: A tensor to quantize.
            :arg scale: A tensor with the scale coefficients.
            :arg output_type: The datatype of the output tensor. Specifying output_type is optional (default value tensorrt.float32).

            :returns: The new dequantization layer, or :class:`None` if it could not be created.
        """
    def add_dist_collective(self, input: ITensor, dist_collective_op: CollectiveOperation, reduce_op: typing.Any, root: typing.Any, groups: typing.Any) -> IDistCollectiveLayer:
        """
            Add a dist collective layer to the network.
            See :class:`IDistCollectiveLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg dist_collective_op: The collective operation to perform.
            :arg reduce_op: The reduction operation to perform when ``dist_collective_op`` is
                :data:`CollectiveOperation.ALL_REDUCE`, or
                :data:`CollectiveOperation.REDUCE`, or
                :data:`CollectiveOperation.REDUCE_SCATTER`.
            :arg root: The root rank of the collective operation.
            :arg group_size: The size of the groups array.
            :arg groups: The groups to perform the collective operation on.

            :returns: The new dist collective layer, or :class:`None` if it could not be created.
        """
    def add_dynamic_quantize(self, input: ITensor, axis: typing.SupportsInt, block_size: typing.SupportsInt, output_type: DataType, scale_type: DataType) -> IDynamicQuantizeLayer:
        """
            Add a dynamic quantization layer to the network.
            See :class:`IDynamicQuantizeLayer` for more information.

            :arg input: A tensor to quantize.
            :arg axis: The axis that is sliced into blocks.
            :arg block_size: The number of elements that are quantized using a shared scale factor.
            :arg output_type: The data type of the quantized output tensor.
            :arg scale_type: The data type of the scale factor used for quantizing the input data.

            :returns: The new DynamicQuantization layer, or :class:`None` if it could not be created.
        """
    def add_dynamic_quantize_v2(self, input: ITensor, block_shape: Dims, output_type: DataType, scale_type: DataType) -> IDynamicQuantizeLayer:
        """
            Add a dynamic quantization layer to the network.
            See :class:`IDynamicQuantizeLayer` for more information.

            :arg input: A tensor to quantize.
            :arg block_shape: The shape of the block.
            :arg output_type: The data type of the quantized output tensor.
            :arg scale_type: The data type of the scale factor used for quantizing the input data.

            :returns: The new DynamicQuantization layer, or :class:`None` if it could not be created.
        """
    def add_einsum(self, inputs: collections.abc.Sequence[ITensor], equation: str) -> IEinsumLayer:
        """
            Adds an Einsum layer to the network.
            See :class:`IEinsumLayer` for more information.

            :arg inputs: The input tensors to the layer.
            :arg equation: The Einsum equation of the layer.

            :returns: the new Einsum layer, or :class:`None` if it could not be created.
        """
    def add_elementwise(self, input1: ITensor, input2: ITensor, op: ElementWiseOperation) -> IElementWiseLayer:
        """
            Add an elementwise layer to the network.
            See :class:`IElementWiseLayer` for more information.

            :arg input1: The first input tensor to the layer.
            :arg input2: The second input tensor to the layer.
            :arg op: The binary operation that the layer applies.

            The input tensors must have the same number of dimensions.
            For each dimension, their lengths must match, or one of them must be one.
            In the latter case, the tensor is broadcast along that axis.

            The output tensor has the same number of dimensions as the inputs.
            For each dimension, its length is the maximum of the lengths of the
            corresponding input dimension.

            :returns: The new element-wise layer, or :class:`None` if it could not be created.
        """
    @typing.overload
    def add_fill(self, shape: Dims, op: FillOperation, output_type: DataType) -> IFillLayer:
        """
            Add a fill layer.
            See :class:`IFillLayer` for more information.

            :arg dimensions: The output tensor dimensions.
            :arg op: The fill operation that the layer applies.
            :arg output_type: The datatype of the output tensor. Specifying output_type is optional (default value tensorrt.float32).

            :returns: The new fill layer, or :class:`None` if it could not be created.
        """
    @typing.overload
    def add_fill(self, shape: Dims, op: FillOperation) -> IFillLayer:
        """
            Add a fill layer.
            See :class:`IFillLayer` for more information.

            :arg dimensions: The output tensor dimensions.
            :arg op: The fill operation that the layer applies.
            :arg output_type: The datatype of the output tensor. Specifying output_type is optional (default value tensorrt.float32).

            :returns: The new fill layer, or :class:`None` if it could not be created.
        """
    def add_gather(self, input: ITensor, indices: ITensor, axis: typing.SupportsInt) -> IGatherLayer:
        """
            Add a gather layer to the network.
            See :class:`IGatherLayer` for more information.

            :arg input: The tensor to gather values from.
            :arg indices: The tensor to get indices from to populate the output tensor.
            :arg axis: The non-batch dimension axis in the data tensor to gather on.

            :returns: The new gather layer, or :class:`None` if it could not be created.
        """
    def add_gather_v2(self, input: ITensor, indices: ITensor, mode: GatherMode) -> IGatherLayer:
        """
            Add a gather layer to the network.
            See :class:`IGatherLayer` for more information.

            :arg input: The tensor to gather values from.
            :arg indices: The tensor to get indices from to populate the output tensor.
            :arg mode: The gather mode.

            :returns: The new gather layer, or :class:`None` if it could not be created.
        """
    def add_grid_sample(self, input: ITensor, grid: ITensor) -> IGridSampleLayer:
        """
            Creates a GridSample layer with a trt.InterpolationMode.LINEAR, unaligned corners, and trt.SampleMode.FILL for 4d-shape input tensors.
            See :class:`IGridSampleLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg grid: The grid tensor to the layer.
            :ivar interpolation_mode: class:`InterpolationMode` The interpolation mode to use in the layer. Default is LINEAR.
            :ivar align_corners: class:`bool` the align mode to use in the layer. Default is False.
            :ivar padding_mode: :class:`SampleMode` The padding mode to use in the layer. Default is FILL.

            :returns: The new grid sample layer, or :class:`None` if it could not be created.
        """
    def add_identity(self, input: ITensor) -> IIdentityLayer:
        """
            Add an identity layer.
            See :class:`IIdentityLayer` for more information.

            :arg input: The input tensor to the layer.

            :returns: The new identity layer, or :class:`None` if it could not be created.
        """
    def add_if_conditional(self) -> IIfConditional:
        """
            Adds an if-conditional to the network, which provides a way to specify subgraphs that will be conditionally executed using lazy evaluation.
            See :class:`IIfConditional` for more information.

            :returns: The new if-condtional, or :class:`None` if it could not be created.
        """
    def add_input(self, name: str, dtype: DataType, shape: Dims) -> ITensor:
        """
            Adds an input to the network.

            :arg name: The name of the tensor. Each input and output tensor must have a unique name.
            :arg dtype: The data type of the tensor.
            :arg shape: The dimensions of the tensor.

            :returns: The newly added Tensor.
        """
    def add_kv_cache_update(self, cache: ITensor, update: ITensor, write_indices: ITensor, cache_mode: KVCacheMode) -> IKVCacheUpdateLayer:
        """
            Add a KVCacheUpdate layer to the network.
            See :class:`IKVCacheUpdateLayer` for more information.

            :arg cache: The key/value cache tensor for the layer. The user is responsible for properly allocating and binding the tensor memory.
            :arg update: The newly updated key/value tensor for the layer.
            :arg write_indices: The write indices tensor for key/value cache updates.
            :arg cache_mode: The mode of the KVCacheUpdate layer. For TensorRT 10.15, only `LINEAR` mode is supported.

            :returns: The new KVCacheUpdate layer, or :class:`None` if it could not be created.
        """
    def add_loop(self) -> ILoop:
        """
            Adds a loop to the network, which provides a way to specify a recurrent subgraph.
            See :class:`ILoop` for more information.

            :returns: The new loop layer, or :class:`None` if it could not be created.
        """
    def add_lrn(self, input: ITensor, window: typing.SupportsInt, alpha: typing.SupportsFloat, beta: typing.SupportsFloat, k: typing.SupportsFloat) -> ILRNLayer:
        """
            Add a LRN layer to the network.
            See :class:`ILRNLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg window: The size of the window.
            :arg alpha: The alpha value for the LRN computation.
            :arg beta: The beta value for the LRN computation.
            :arg k: The k value for the LRN computation.

            :returns: The new LRN layer, or :class:`None` if it could not be created.
        """
    def add_matrix_multiply(self, input0: ITensor, op0: MatrixOperation, input1: ITensor, op1: MatrixOperation) -> IMatrixMultiplyLayer:
        """
            Add a matrix multiply layer to the network.
            See :class:`IMatrixMultiplyLayer` for more information.

            :arg input0: The first input tensor (commonly A).
            :arg op0: Whether to treat input0 as matrices, transposed matrices, or vectors.
            :arg input1: The second input tensor (commonly B).
            :arg op1:  Whether to treat input1 as matrices, transposed matrices, or vectors.

            :returns: The new matrix multiply layer, or :class:`None` if it could not be created.
        """
    def add_moe(self, hidden_states: ITensor, selected_experts_for_tokens: ITensor, scores_for_selected_experts: ITensor) -> IMoELayer:
        """
            Add a MoE layer to the network.
            See :class:`IMoELayer` for more information.

            :arg hidden_states: The hidden states tensor input to the MoE layer.
            :arg selected_experts_for_tokens: The tensor containing expert indices selected for each token.
            :arg scores_for_selected_experts: The tensor containing scores computed for the selected experts.

            :returns: The new MoE layer, or :class:`None` if it could not be created.
        """
    @typing.overload
    def add_nms(self, boxes: ITensor, scores: ITensor, max_output_boxes_per_class: ITensor) -> INMSLayer:
        """
            Add a non-maximum suppression layer to the network.
            See :class:`INMSLayer` for more information.

            :arg boxes: The input boxes tensor to the layer.
            :arg scores: The input scores tensor to the layer.
            :arg max_output_boxes_per_class: The maxOutputBoxesPerClass tensor to the layer.
            :ivar bounding_box_format: :class:`BoundingBoxFormat` The bounding box format used by the layer. Default is CORNER_PAIRS.
            :ivar topk_box_limit: :class:`int` The maximum number of filtered boxes considered for selection per batch item. Default is 2000 for SM 5.3 and 6.2 devices, and 5000 otherwise. The TopK box limit must be less than or equal to {2000 for SM 5.3 and 6.2 devices, 5000 otherwise}.
            :arg indices_type: The datatype of the output indices tensor. Specifying indices_type is optional (default value tensorrt.int32).

            :returns: The new NMS layer, or :class:`None` if it could not be created.
        """
    @typing.overload
    def add_nms(self, boxes: ITensor, scores: ITensor, max_output_boxes_per_class: ITensor, indices_type: DataType) -> INMSLayer:
        """
            Add a non-maximum suppression layer to the network.
            See :class:`INMSLayer` for more information.

            :arg boxes: The input boxes tensor to the layer.
            :arg scores: The input scores tensor to the layer.
            :arg max_output_boxes_per_class: The maxOutputBoxesPerClass tensor to the layer.
            :ivar bounding_box_format: :class:`BoundingBoxFormat` The bounding box format used by the layer. Default is CORNER_PAIRS.
            :ivar topk_box_limit: :class:`int` The maximum number of filtered boxes considered for selection per batch item. Default is 2000 for SM 5.3 and 6.2 devices, and 5000 otherwise. The TopK box limit must be less than or equal to {2000 for SM 5.3 and 6.2 devices, 5000 otherwise}.
            :arg indices_type: The datatype of the output indices tensor. Specifying indices_type is optional (default value tensorrt.int32).

            :returns: The new NMS layer, or :class:`None` if it could not be created.
        """
    @typing.overload
    def add_non_zero(self, input: ITensor) -> INonZeroLayer:
        """
            Adds an NonZero layer to the network.
            See :class:`INonZeroLayer` for more information.

            :arg input: The input tensor to the layer.

            :arg indices_type: The datatype of the output indices tensor. Specifying indices_type is optional (default value tensorrt.int32).

            :returns: the new NonZero layer, or :class:`None` if it could not be created.
        """
    @typing.overload
    def add_non_zero(self, input: ITensor, indices_type: DataType) -> INonZeroLayer:
        """
            Adds an NonZero layer to the network.
            See :class:`INonZeroLayer` for more information.

            :arg input: The input tensor to the layer.

            :arg indices_type: The datatype of the output indices tensor. Specifying indices_type is optional (default value tensorrt.int32).

            :returns: the new NonZero layer, or :class:`None` if it could not be created.
        """
    def add_normalization(self, input: ITensor, scale: ITensor, bias: ITensor, axesMask: typing.SupportsInt) -> INormalizationLayer:
        """
            [DEPRECATED] Deprecated in TensorRT 10.15. Superseded by add_normalization_v2.

            Adds a Normalization layer to the network.
            See :class:`Normalization` for more information.

            :arg input: The input tensor to the layer.
            :arg scale: The scale tensor used to scale the normalized output.
            :arg bias: The bias tensor used to scale the normalized output.
            :arg axesMask: The axes on which to perform mean calculations.
                The bit in position i of bitmask axes corresponds to explicit dimension i of the result.
                E.g., the least significant bit corresponds to the first explicit dimension and the next to least
                significant bit corresponds to the second explicit dimension.

            The normalization layer works by performing normalization of the tensor input on the specified axesMask.
            The result is then scaled by multiplying with scale and adding bias.

            The shape of scale and bias must be the same, and must have the same rank and be
            unidirectionally broadcastable to the shape of input. Given a 4D NCHW input tensor, the expected shapes
            for scale and bias are:
            * [1, C, 1, 1] for InstanceNormalization
            * [1, G, 1, 1] for GroupNormalization. Use :func:`INetworkDefinition.add_normalization_v2` instead if [1, C, 1, 1] shapes for scale and bias are required.

            :returns: the new Normalization layer, or :class:`None` if it could not be created.
        """
    def add_normalization_v2(self, input: ITensor, scale: ITensor, bias: ITensor, axesMask: typing.SupportsInt) -> INormalizationLayer:
        """
            Adds a Normalization layer to the network.
            See :class:`Normalization` for more information.

            :arg input: The input tensor to the layer.
            :arg scale: The scale tensor used to scale the normalized output.
            :arg bias: The bias tensor used to scale the normalized output.
            :arg axesMask: The axes on which to perform mean calculations.
                The bit in position i of bitmask axes corresponds to explicit dimension i of the result.
                E.g., the least significant bit corresponds to the first explicit dimension and the next to least
                significant bit corresponds to the second explicit dimension.

            The normalization layer works by performing normalization of the tensor input on the specified axesMask.
            The result is then scaled by multiplying with scale and adding bias.

            The shapes of scale and bias must be the same, and must have the same rank and be
            unidirectionally broadcastable to the shape of input. In the case of InstanceNorm or GroupNorm,
            the shapes of scale and bias are expected to be [1, C, 1, 1] in the case of a 4D NCHW input tensor.

            :returns: the new Normalization layer, or :class:`None` if it could not be created.
        """
    def add_one_hot(self, indices: ITensor, values: ITensor, depth: ITensor, axis: typing.SupportsInt) -> IOneHotLayer:
        """
            Add a OneHot layer to the network.
            See :class:`IOneHotLayer` for more information.

            :arg indices: The tensor to get indices from to populate the output tensor.
            :arg values: The tensor to get off (cold) value and on (hot) value
            :arg depth: The tensor to get depth (number of classes) of one-hot encoding
            :arg axis: The axis to append the one-hot encoding to

            :returns: The new OneHot layer, or :class:`None` if it could not be created.
        """
    def add_padding_nd(self, input: ITensor, pre_padding: Dims, post_padding: Dims) -> IPaddingLayer:
        """
            Add a multi-dimensional padding layer to the network.
            See :class:`IPaddingLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg pre_padding: The padding to apply to the start of the tensor.
            :arg post_padding: The padding to apply to the end of the tensor.

            :returns: The new padding layer, or :class:`None` if it could not be created.
        """
    def add_parametric_relu(self, input: ITensor, slopes: ITensor) -> IParametricReLULayer:
        """
                Add a parametric ReLU layer.
                See :class:`IParametricReLULayer` for more information.

                :arg input: The input tensor to the layer.
                :arg slopes: The slopes tensor (input elements are multiplied with the slopes where the input is negative).

                :returns: The new parametric ReLU layer, or :class:`None` if it could not be created.
        """
    @typing.overload
    def add_plugin(self, tuple: tuple) -> IPluginV3Layer:
        """
            Add a plugin layer to the network using an :class:`IPluginV3` interface.
            See :class:`IPluginV3` for more information.

            :arg inputs: The input tensors to the layer.
            :arg shape_inputs: The shape input tensors to the layer.
            :arg plugin: The layer plugin.

            :returns: The new plugin layer, or :class:`None` if it could not be created.
        """
    @typing.overload
    def add_plugin(self, func: collections.abc.Callable) -> IPluginV3Layer:
        ...
    @typing.overload
    def add_plugin(self, func: collections.abc.Callable, aot: bool) -> IPluginV3Layer:
        ...
    def add_plugin_v2(self, inputs: collections.abc.Sequence[ITensor], plugin: IPluginV2) -> IPluginV2Layer:
        """
            Add a plugin layer to the network using an :class:`IPluginV2` interface.
            See :class:`IPluginV2` for more information.

            :arg inputs: The input tensors to the layer.
            :arg plugin: The layer plugin.

            :returns: The new plugin layer, or :class:`None` if it could not be created.
        """
    def add_plugin_v3(self, inputs: collections.abc.Sequence[ITensor], shape_inputs: collections.abc.Sequence[ITensor], plugin: IPluginV3) -> IPluginV3Layer:
        """
            Add a plugin layer to the network using an :class:`IPluginV3` interface.
            See :class:`IPluginV3` for more information.

            :arg inputs: The input tensors to the layer.
            :arg shape_inputs: The shape input tensors to the layer.
            :arg plugin: The layer plugin.

            :returns: The new plugin layer, or :class:`None` if it could not be created.
        """
    def add_pooling_nd(self, input: ITensor, type: PoolingType, window_size: Dims) -> IPoolingLayer:
        """
            Add a multi-dimension pooling layer to the network.
            See :class:`IPoolingLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg type: The type of pooling to apply.
            :arg window_size: The size of the pooling window.

            :returns: The new pooling layer, or :class:`None` if it could not be created.
        """
    @typing.overload
    def add_quantize(self, input: ITensor, scale: ITensor) -> IQuantizeLayer:
        """
            Add a quantization layer to the network.
            See :class:`IQuantizeLayer` for more information.

            :arg input: A tensor to quantize.
            :arg scale: A tensor with the scale coefficients.
            :arg output_type: The datatype of the output tensor. Specifying output_type is optional (default value tensorrt.int8).

            :returns: The new quantization layer, or :class:`None` if it could not be created.
        """
    @typing.overload
    def add_quantize(self, input: ITensor, scale: ITensor, output_type: DataType) -> IQuantizeLayer:
        """
            Add a quantization layer to the network.
            See :class:`IQuantizeLayer` for more information.

            :arg input: A tensor to quantize.
            :arg scale: A tensor with the scale coefficients.
            :arg output_type: The datatype of the output tensor. Specifying output_type is optional (default value tensorrt.int8).

            :returns: The new quantization layer, or :class:`None` if it could not be created.
        """
    def add_ragged_softmax(self, input: ITensor, bounds: ITensor) -> IRaggedSoftMaxLayer:
        """
            Add a ragged softmax layer to the network.
            See :class:`IRaggedSoftMaxLayer` for more information.

            :arg input: The ZxS input tensor.
            :arg bounds: The Zx1 bounds tensor.

            :returns: The new ragged softmax layer, or :class:`None` if it could not be created.
        """
    def add_reduce(self, input: ITensor, op: ReduceOperation, axes: typing.SupportsInt, keep_dims: bool) -> IReduceLayer:
        """
            Add a reduce layer to the network.
            See :class:`IReduceLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg op: The reduction operation to perform.
            :arg axes: The reduction dimensions.
                The bit in position i of bitmask axes corresponds to explicit dimension i of the result.
                E.g., the least significant bit corresponds to the first explicit dimension and the next to least
                significant bit corresponds to the second explicit dimension.
            :arg keep_dims: The boolean that specifies whether or not to keep the reduced dimensions in the output of the layer.

            :returns: The new reduce layer, or :class:`None` if it could not be created.
        """
    def add_resize(self, input: ITensor) -> IResizeLayer:
        """
            Add a resize layer.
            See :class:`IResizeLayer` for more information.

            :arg input: The input tensor to the layer.

            :returns: The new resize layer, or :class:`None` if it could not be created.
        """
    def add_reverse_sequence(self, input: ITensor, sequence_lens: ITensor) -> IReverseSequenceLayer:
        """
            Adds a ReverseSequence layer to the network.
            See :class:`IReverseSequenceLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg sequence_lens: 1D tensor specifying lengths of sequences to reverse in a batch. The length of ``sequence_lens`` must be equal to the size of the dimension in ``input`` specified by ``batch_axis``.

            :returns: the new ReverseSequence layer, or :class:`None` if it could not be created.
        """
    def add_rotary_embedding(self, input: ITensor, cos_cache: ITensor, sin_cache: ITensor, interleaved: bool, rotary_embedding_dim: typing.SupportsInt) -> IRotaryEmbeddingLayer:
        """
            Add a RotaryEmbedding layer to the network.
            See :class:`IRotaryEmbeddingLayer` for more information.

            :arg input: The input activation tensor to the layer.
            :arg cos_cache: The cosine cache tensor for use in RoPE computation.
            :arg sin_cache: The sine cache tensor for use in RoPE computation.
            :arg interleaved: Whether the input tensor is in interleaved format, i.e., whether the 2-d vectors rotated are taken from adjacent 2 elements in the hidden dimension.
            :arg rotary_embedding_dim: The hidden dimension that participates in RoPE.

            An optional input, position_ids, can be provided using :func:`set_input` with index 3. If provided, it is used to index into cos_cache and sin_cache.

            :returns: The new RotaryEmbedding layer, or :class:`None` if it could not be created.
        """
    def add_scale(self, input: ITensor, mode: ScaleMode, shift: Weights = None, scale: Weights = None, power: Weights = None) -> IScaleLayer:
        """
            Add a scale layer to the network.
            See :class:`IScaleLayer` for more information.

            :arg input: The input tensor to the layer. This tensor is required to have a minimum of 3 dimensions.
            :arg mode: The scaling mode.
            :arg shift: The shift value.
            :arg scale: The scale value.
            :arg power: The power value.

            If the weights are available, then the size of weights are dependent on the ScaleMode.
            For UNIFORM, the number of weights is equal to 1.
            For CHANNEL, the number of weights is equal to the channel dimension.
            For ELEMENTWISE, the number of weights is equal to the volume of the input.

            :returns: The new scale layer, or :class:`None` if it could not be created.
        """
    def add_scale_nd(self, input: ITensor, mode: ScaleMode, shift: Weights = None, scale: Weights = None, power: Weights = None, channel_axis: typing.SupportsInt) -> IScaleLayer:
        """
            Add a multi-dimension scale layer to the network.
            See :class:`IScaleLayer` for more information.

            :arg input: The input tensor to the layer. This tensor is required to have a minimum of 3 dimensions.
            :arg mode: The scaling mode.
            :arg shift: The shift value.
            :arg scale: The scale value.
            :arg power: The power value.
            :arg channel_axis: The channel dimension axis.

            If the weights are available, then the size of weights are dependent on the ScaleMode.
            For UNIFORM, the number of weights is equal to 1.
            For CHANNEL, the number of weights is equal to the channel dimension.
            For ELEMENTWISE, the number of weights is equal to the volume of the input.

            :returns: The new scale layer, or :class:`None` if it could not be created.
        """
    def add_scatter(self, data: ITensor, indices: ITensor, updates: ITensor, mode: ScatterMode) -> IScatterLayer:
        """
            Add a scatter layer to the network.
            See :class:`IScatterLayer` for more information.

            :arg data: The tensor to get default values from.
            :arg indices: The tensor to get indices from to populate the output tensor.
            :arg updates: The tensor to get values from to populate the output tensor.
            :arg mode: operation mode see IScatterLayer for more info

            :returns: The new Scatter layer, or :class:`None` if it could not be created.
        """
    def add_select(self, condition: ITensor, then_input: ITensor, else_input: ITensor) -> ISelectLayer:
        """
            Add a select layer.
            See :class:`ISelectLayer` for more information.

            :arg condition: The condition tensor to the layer.
            :arg then_input: The then input tensor to the layer.
            :arg else_input: The else input tensor to the layer.

            :returns: The new select layer, or :class:`None` if it could not be created.
        """
    def add_shape(self, input: ITensor) -> IShapeLayer:
        """
            Add a shape layer to the network.
            See :class:`IShapeLayer` for more information.

            :arg input: The input tensor to the layer.

            :returns: The new shape layer, or :class:`None` if it could not be created.
        """
    def add_shuffle(self, input: ITensor) -> IShuffleLayer:
        """
            Add a shuffle layer to the network.
            See :class:`IShuffleLayer` for more information.

            :arg input: The input tensor to the layer.

            :returns: The new shuffle layer, or :class:`None` if it could not be created.
        """
    def add_slice(self, input: ITensor, start: Dims, shape: Dims, stride: Dims) -> ISliceLayer:
        """
            Add a slice layer to the network.
            See :class:`ISliceLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg start: The start offset.
            :arg shape: The output shape.
            :arg stride: The slicing stride. Positive, negative, zero stride values, and combinations of them in different dimensions are allowed.

            :returns: The new slice layer, or :class:`None` if it could not be created.
        """
    def add_softmax(self, input: ITensor) -> ISoftMaxLayer:
        """
            Add a softmax layer to the network.
            See :class:`ISoftMaxLayer` for more information.

            :arg input: The input tensor to the layer.

            :returns: The new softmax layer, or :class:`None` if it could not be created.
        """
    def add_squeeze(self, input: ITensor, axes: ITensor) -> ISqueezeLayer:
        """
            Adds a Squeeze layer to the network.
            See :class:`ISqueezeLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg axes: The tensor containing axes to remove. Must be resolvable to a constant Int32 or Int64 1D shape tensor.

            :returns: the new Squeeze layer, or :class:`None` if it could not be created.
        """
    @typing.overload
    def add_topk(self, input: ITensor, op: TopKOperation, k: typing.SupportsInt, axes: typing.SupportsInt) -> ITopKLayer:
        """
            Add a TopK layer to the network.
            See :class:`ITopKLayer` for more information.

            The TopK layer has two outputs of the same dimensions. The first contains data values, the second contains index positions for the values. Output values are sorted, largest first for operation :const:`TopKOperation.MAX` and smallest first for operation :const:`TopKOperation.MIN` .

            Currently only values of K up to 3840 are supported.

            :arg input: The input tensor to the layer.
            :arg op: Operation to perform.
            :arg k: Number of elements to keep.

            :arg axes: The reduction dimensions.
                The bit in position i of bitmask axes corresponds to explicit dimension i of the result.
                E.g., the least significant bit corresponds to the first explicit dimension and the next to least
                significant bit corresponds to the second explicit dimension.
                Currently axes must specify exactly one dimension, and it must be one of the last four dimensions.

            :arg indices_type: The datatype of the output indices tensor. Specifying indices_type is optional (default value tensorrt.int32).

            :returns: The new TopK layer, or :class:`None` if it could not be created.
        """
    @typing.overload
    def add_topk(self, input: ITensor, op: TopKOperation, k: typing.SupportsInt, axes: typing.SupportsInt, indices_type: DataType) -> ITopKLayer:
        """
            Add a TopK layer to the network.
            See :class:`ITopKLayer` for more information.

            The TopK layer has two outputs of the same dimensions. The first contains data values, the second contains index positions for the values. Output values are sorted, largest first for operation :const:`TopKOperation.MAX` and smallest first for operation :const:`TopKOperation.MIN` .

            Currently only values of K up to 3840 are supported.

            :arg input: The input tensor to the layer.
            :arg op: Operation to perform.
            :arg k: Number of elements to keep.

            :arg axes: The reduction dimensions.
                The bit in position i of bitmask axes corresponds to explicit dimension i of the result.
                E.g., the least significant bit corresponds to the first explicit dimension and the next to least
                significant bit corresponds to the second explicit dimension.
                Currently axes must specify exactly one dimension, and it must be one of the last four dimensions.

            :arg indices_type: The datatype of the output indices tensor. Specifying indices_type is optional (default value tensorrt.int32).

            :returns: The new TopK layer, or :class:`None` if it could not be created.
        """
    def add_unary(self, input: ITensor, op: UnaryOperation) -> IUnaryLayer:
        """
            Add a unary layer to the network.
            See :class:`IUnaryLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg op: The operation to apply.

            :returns: The new unary layer, or :class:`None` if it could not be created.
        """
    def add_unsqueeze(self, input: ITensor, axes: ITensor) -> IUnsqueezeLayer:
        """
            Adds an Unsqueeze layer to the network.
            See :class:`IUnsqueezeLayer` for more information.

            :arg input: The input tensor to the layer.
            :arg axes: The tensor containing axes to add. Must be resolvable to a constant Int32 or Int64 1D shape tensor.

            :returns: the new Unsqueeze layer, or :class:`None` if it could not be created.
        """
    def are_weights_marked_refittable(self, name: str) -> bool:
        """
            Whether the weight has been marked as refittable.

            :arg name: The name of the weights to check.
        """
    def get_flag(self, flag: ...) -> bool:
        """
            Returns true if the specified ``NetworkDefinitionCreationFlag`` is set.

            :arg flag: The ``NetworkDefinitionCreationFlag`` .

            :returns: Whether the flag is set.
        """
    def get_input(self, index: typing.SupportsInt) -> ITensor:
        """
            Get the input tensor specified by the given index.

            :arg index: The index of the input tensor.

            :returns: The tensor, or :class:`None` if it is out of range.
        """
    def get_layer(self, index: typing.SupportsInt) -> ILayer:
        """
            Get the layer specified by the given index.

            :arg index: The index of the layer.

            :returns: The layer, or :class:`None` if it is out of range.
        """
    def get_output(self, index: typing.SupportsInt) -> ITensor:
        """
            Get the output tensor specified by the given index.

            :arg index: The index of the output tensor.

            :returns: The tensor, or :class:`None` if it is out of range.
        """
    def is_debug_tensor(self, tensor: ITensor) -> bool:
        """
            Check if a tensor is marked as debug.

            :arg tensor: The tensor to be checked.
        """
    def mark_debug(self, tensor: ITensor) -> bool:
        """
            Mark a tensor as a debug tensor in the network.

            :arg tensor: The tensor to be marked as debug tensor.

            :returns: True on success, False otherwise.
        """
    def mark_output(self, tensor: ITensor) -> None:
        """
            Mark a tensor as an output.

            :arg tensor: The tensor to mark.
        """
    def mark_output_for_shapes(self, tensor: ITensor) -> bool:
        """
            Enable tensor's value to be computed by :func:`IExecutionContext.get_shape_binding`.

            :arg tensor: The tensor to unmark as an output tensor. The tensor must be of type :class:`int32` and have no more than one dimension.

            :returns: :class:`True` if successful, :class:`False` if tensor is already marked as an output.
        """
    def mark_unfused_tensors_as_debug_tensors(self) -> bool:
        """
            Mark unfused tensors as debug tensors.

            Debug tensors can be optionally emitted at runtime.
            Tensors that are fused by the optimizer will not be emitted.
            Tensors marked this way will not prevent fusion like mark_debug() does, thus preserving performance.

            Tensors marked this way cannot be detected by is_debug_tensor().
            DebugListener can only get internal tensor names instead of the original tensor names in the NetworkDefinition for tensors marked this way.
            But the names correspond to the names obtained by IEngineInspector.
            There is no guarantee that all unfused tensors are marked.

            :returns: True if tensors were successfully marked (or were already marked), false otherwise.
        """
    def mark_weights_refittable(self, name: str) -> bool:
        """
            Mark a weight as refittable.

            :arg name: The weight to mark.
        """
    def remove_tensor(self, tensor: ITensor) -> None:
        """
                Remove a tensor from the network.

                :arg tensor: The tensor to remove

                It is illegal to remove a tensor that is the input or output of a layer.
                if this method is called with such a tensor, a warning will be emitted on the log
                and the call will be ignored.
        """
    def set_weights_name(self, weights: Weights, name: str) -> bool:
        """
            Associate a name with all current uses of the given weights.

            The name must be set after the Weights are used in the network.
            Lookup is associative. The name applies to all Weights with matching
            type, value pointer, and count. If Weights with a matching value
            pointer, but different type or count exists in the network, an
            error message is issued, the name is rejected, and return false.
            If the name has already been used for other weights,
            return false. None causes the weights to become unnamed,
            i.e. clears any previous name.

            :arg weights: The weights to be named.
            :arg name: The name to associate with the weights.

            :returns: true on success.
        """
    def unmark_debug(self, tensor: ITensor) -> bool:
        """
            Unmark a tensor as a debug tensor in the network.

            :arg tensor: The tensor to be unmarked as debug tensor.

            :returns: True on success, False otherwise.
        """
    def unmark_output(self, tensor: ITensor) -> None:
        """
                Unmark a tensor as a network output.

                :arg tensor: The tensor to unmark as an output tensor.
        """
    def unmark_output_for_shapes(self, tensor: ITensor) -> bool:
        """
            Undo :func:`mark_output_for_shapes` .

            :arg tensor: The tensor to unmark as an output tensor.

            :returns: :class:`True` if successful, :class:`False` if tensor is not marked as an output.
        """
    def unmark_unfused_tensors_as_debug_tensors(self) -> bool:
        """
            Undo the marking of unfused tensor as debug tensors.

            This has no effect on tensors marked by mark_debug().

            :returns: True if tensor successfully unmarked (or was already unmarked), false otherwise.
        """
    def unmark_weights_refittable(self, name: str) -> bool:
        """
            Unmark a weight as refittable.

            :arg name: The weight to unmark.
        """
    @property
    def builder(self) -> ...:
        """
            The builder from which this INetworkDefinition was created.

            See :class:`IBuilder` for more information.
        """
    @property
    def flags(self) -> int:
        ...
    @property
    def has_implicit_batch_dimension(self) -> bool:
        ...
    @property
    def num_inputs(self) -> int:
        ...
    @property
    def num_layers(self) -> int:
        ...
    @property
    def num_outputs(self) -> int:
        ...
class INonZeroLayer(ILayer):
    """

        A NonZero layer in an :class:`INetworkDefinition` .

        Computes the indices of the input tensor where the value is non-zero. The returned indices are in row-major order.

        The output shape is always `{D, C}`, where `D` is the number of dimensions of the input and `C` is the number of non-zero values.

        :ivar indices_type: :class:`DataType` The specified data type of the output indices tensor. Must be tensorrt.int32 or tensorrt.int64.
    """
    indices_type: DataType
class INormalizationLayer(ILayer):
    """

        A Normalization layer in an :class:`INetworkDefinition` .

        The normalization layer performs the following operation:

        X - input Tensor
        Y - output Tensor
        S - scale Tensor
        B - bias Tensor

        Y = (X - Mean(X, axes)) / Sqrt(Variance(X) + epsilon) * S + B

        Where Mean(X, axes) is a reduction over a set of axes, and Variance(X) = Mean((X - Mean(X, axes)) ^ 2, axes).

        :ivar epsilon: :class:`float` The epsilon value used for the normalization calculation. Default: 1e-5F.
        :ivar axes: :class:`int` The reduction axes for the normalization calculation.
        :ivar num_groups: :class:`int` The number of groups to split the channels into for the normalization calculation. Default: 1.
        :ivar compute_precision: :class:`DataType` The datatype used for the compute precision of this layer. By default TensorRT will run the normalization computation in DataType.kFLOAT32 even in mixed precision mode regardless of any set builder flags to avoid overflow errors. ILayer.precision and ILayer.set_output_type can still be set to control input and output types of this layer. Only DataType.kFLOAT32 and DataType.kHALF are valid for this member. Default: Datatype.FLOAT.
    """
    compute_precision: DataType
    @property
    def axes(self) -> int:
        ...
    @axes.setter
    def axes(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg1: typing.SupportsFloat) -> None:
        ...
    @property
    def is_v2(self) -> bool:
        ...
    @property
    def num_groups(self) -> int:
        ...
    @num_groups.setter
    def num_groups(self, arg1: typing.SupportsInt) -> None:
        ...
class IOneHotLayer(ILayer):
    """

        A OneHot layer in a network definition.

        The OneHot layer has three input tensors: Indices, Values, and Depth, one output tensor,
        Output, and an axis attribute.
        :ivar indices: is an Int32 tensor that determines which locations in Output to set as on_value.
        :ivar values: is a two-element (rank=1) tensor that consists of [off_value, on_value]
        :ivar depth: is an Int32 shape tensor of rank 0, which contains the depth (number of classes) of the one-hot encoding.
        The depth tensor must be a build-time constant, and its value should be positive.
        :returns: a tensor with rank = rank(indices)+1, where the added dimension contains the one-hot encoding.
        :param axis: specifies to which dimension of the output one-hot encoding is added.

        The data types of Output shall be equal to the Values data type.
        The output is computed by copying off_values to all output elements, then setting on_value on the indices
        specified by the indices tensor.

        when axis = 0:
        output[indices[i, j, k], i, j, k] = on_value for all i, j, k and off_value otherwise.

        when axis = -1:
        output[i, j, k, indices[i, j, k]] = on_value for all i, j, k and off_value otherwise.

    """
    @property
    def axis(self) -> int:
        ...
    @axis.setter
    def axis(self, arg1: typing.SupportsInt) -> None:
        ...
class IOptimizationProfile:
    """

        Optimization profile for dynamic input dimensions and shape tensors.

        When building an :class:`ICudaEngine` from an :class:`INetworkDefinition` that has dynamically resizable inputs (at least
        one input tensor has one or more of its dimensions specified as -1) or shape input tensors, users need to specify
        at least one optimization profile. Optimization profiles are numbered 0, 1, ...

        The first optimization profile that has been defined (with index 0) will be used by the :class:`ICudaEngine` whenever no
        optimization profile has been selected explicitly. If none of the inputs are dynamic, the default optimization
        profile will be generated automatically unless it is explicitly provided by the user (this is possible but not
        required in this case). If more than a single optimization profile is defined, users may set a target how
        much additional weight space should be maximally allocated to each additional profile (as a fraction of the
        maximum, unconstrained memory).

        Users set optimum input tensor dimensions, as well as minimum and maximum input tensor dimensions. The builder
        selects the kernels that result in the lowest runtime for the optimum input tensor dimensions, and are valid for
        all input tensor sizes in the valid range between minimum and maximum dimensions. A runtime error will be raised
        if the input tensor dimensions fall outside the valid range for this profile. Likewise, users provide minimum,
        optimum, and maximum values for all shape tensor input values.

        :class:`IOptimizationProfile` implements :func:`__nonzero__` and :func:`__bool__` such that evaluating a profile as a :class:`bool` (e.g. ``if profile:``) will check whether the optimization profile can be passed to an IBuilderConfig object. This will perform partial validation, by e.g. checking that the maximum dimensions are at least as large as the optimum dimensions, and that the optimum dimensions are always as least as large as the minimum dimensions. Some validation steps require knowledge of the network definition and are deferred to engine build time.

        :ivar extra_memory_target: Additional memory that the builder should aim to maximally allocate for this profile, as a fraction of the memory it would use if the user did not impose any constraints on memory. This unconstrained case is the default; it corresponds to ``extra_memory_target`` == 1.0. If ``extra_memory_target`` == 0.0, the builder aims to create the new optimization profile without allocating any additional weight memory. Valid inputs lie between 0.0 and 1.0. This parameter is only a hint, and TensorRT does not guarantee that the ``extra_memory_target`` will be reached. This parameter is ignored for the first (default) optimization profile that is defined.
    """
    def __bool__(self) -> bool:
        ...
    def __nonzero__(self) -> bool:
        ...
    def get_shape(self, input: str) -> list[Dims]:
        """
            Get the minimum/optimum/maximum dimensions for a dynamic input tensor.
            If the dimensions have not been previously set via :func:`set_shape`, return an invalid :class:`Dims` with a length of -1.

            :returns: A ``List[Dims]`` of length 3, containing the minimum, optimum, and maximum shapes, in that order. If the shapes have not been set yet, an empty list is returned.
        """
    def get_shape_input(self, input: str) -> list[list[int]]:
        """
            Get the minimum/optimum/maximum values for a shape input tensor.

            :returns: A ``List[List[int]]`` of length 3, containing the minimum, optimum, and maximum values, in that order. If the values have not been set yet, an empty list is returned.
        """
    def set_shape(self, input: str, min: Dims, opt: Dims, max: Dims) -> None:
        """
            Set the minimum/optimum/maximum dimensions for a dynamic input tensor.

            This function must be called for any network input tensor that has dynamic dimensions. If ``min``, ``opt``, and ``max`` are the minimum, optimum, and maximum dimensions, and ``real_shape`` is the shape for this input tensor provided to the :class:`INetworkDefinition` ,then the following conditions must hold:

            (1) ``len(min)`` == ``len(opt)`` == ``len(max)`` == ``len(real_shape)``
            (2) 0 <= ``min[i]`` <= ``opt[i]`` <= ``max[i]`` for all ``i``
            (3) if ``real_shape[i]`` != -1, then ``min[i]`` == ``opt[i]`` == ``max[i]`` == ``real_shape[i]``

            This function may (but need not be) called for an input tensor that does not have dynamic dimensions. In this
            case, all shapes must equal ``real_shape``.

            :arg input: The name of the input tensor.
            :arg min: The minimum dimensions for this input tensor.
            :arg opt: The optimum dimensions for this input tensor.
            :arg max: The maximum dimensions for this input tensor.

            :raises: :class:`ValueError` if an inconsistency was detected. Note that inputs can be validated only partially; a full validation is performed at engine build time.
        """
    def set_shape_input(self, input: str, min: collections.abc.Sequence[typing.SupportsInt], opt: collections.abc.Sequence[typing.SupportsInt], max: collections.abc.Sequence[typing.SupportsInt]) -> None:
        """
            Set the minimum/optimum/maximum values for a shape input tensor.

            This function must be called for every input tensor ``t`` that is a shape tensor (``t.is_shape`` == ``True``).
            This implies that the datatype of ``t`` is ``int64`` or ``int32``, the rank is either 0 or 1, and the dimensions of ``t``
            are fixed at network definition time. This function must NOT be called for any input tensor that is not a
            shape tensor.

            If ``min``, ``opt``, and ``max`` are the minimum, optimum, and maximum values, it must be true that ``min[i]`` <= ``opt[i]`` <= ``max[i]`` for
            all ``i``.

            :arg input: The name of the input tensor.
            :arg min: The minimum values for this shape tensor.
            :arg opt: The optimum values for this shape tensor.
            :arg max: The maximum values for this shape tensor.

            :raises: :class:`ValueError` if an inconsistency was detected. Note that inputs can be validated only partially; a full validation is performed at engine build time.
        """
    @property
    def extra_memory_target(self) -> float:
        ...
    @extra_memory_target.setter
    def extra_memory_target(self, arg1: typing.SupportsFloat) -> bool:
        ...
class IOutputAllocator:
    """

    Application-implemented class for controlling output tensor allocation.

    To implement a custom output allocator, ensure that you explicitly instantiate the base class in :func:`__init__` :
    ::

        class MyOutputAllocator(trt.IOutputAllocator):
            def __init__(self):
                trt.IOutputAllocator.__init__(self)

            def reallocate_output(self, tensor_name, memory, size, alignment):
                ... # Your implementation here

            def reallocate_output_async(self, tensor_name, memory, size, alignment, stream):
                ... # Your implementation here

            def notify_shape(self, tensor_name, shape):
                ... # Your implementation here

    """
    def __init__(self) -> None:
        ...
class IPaddingLayer(ILayer):
    """

        A padding layer in an :class:`INetworkDefinition` .

        :ivar pre_padding_nd: :class:`Dims` The padding that is applied at the start of the tensor. Negative padding results in trimming the edge by the specified amount. Only 2 dimensions currently supported.
        :ivar post_padding_nd: :class:`Dims` The padding that is applied at the end of the tensor. Negative padding results in trimming the edge by the specified amount. Only 2 dimensions currently supported.
    """
    post_padding_nd: Dims
    pre_padding_nd: Dims
class IParametricReLULayer(ILayer):
    """

        A parametric ReLU layer in an :class:`INetworkDefinition` .

        This layer applies a parametric ReLU activation to an input tensor (first input), with slopes taken from a
        slopes tensor (second input). This can be viewed as a leaky ReLU operation where the negative slope differs
        from element to element (and can in fact be learned).

        The slopes tensor must be unidirectional broadcastable to the input tensor: the rank of the two tensors must
        be the same, and all dimensions of the slopes tensor must either equal the input tensor or be 1.
        The output tensor has the same shape as the input tensor.
    """
class IPluginCapability(IVersionedInterface):
    """

        Base class for plugin capability interfaces

        IPluginCapability represents a split in TensorRT V3 plugins to sub-objects that expose different types of capabilites a plugin may have,
        as opposed to a single interface which defines all capabilities and behaviors of a plugin.
    """
class IPluginCreator(IPluginCreatorInterface, IVersionedInterface):
    """

        Plugin creator class for user implemented layers

        :ivar name: :class:`str` Plugin name.
        :ivar plugin_version: :class:`str` Plugin version.
        :ivar field_names: :class:`list` List of fields that needs to be passed to :func:`create_plugin` .
        :ivar plugin_namespace: :class:`str` The namespace of the plugin creator based on the plugin library it belongs to. This can be set while registering the plugin creator.
    """
    field_names: PluginFieldCollection_
    name: str
    plugin_namespace: str
    plugin_version: str
    def __init__(self) -> None:
        ...
    def create_plugin(self, name: str, field_collection: PluginFieldCollection_) -> IPluginV2:
        """
            Creates a new plugin.

            :arg name: The name of the plugin.
            :arg field_collection: The :class:`PluginFieldCollection` for this plugin.

            :returns: :class:`IPluginV2` or :class:`None` on failure.
        """
    @typing.overload
    def deserialize_plugin(self, name: str, serialized_plugin: collections.abc.Buffer) -> IPluginV2:
        """
            Creates a plugin object from a serialized plugin.

            .. warning::
                This API only applies when called on a C++ plugin from a Python program.

            `serialized_plugin` will contain a Python bytes object containing the serialized representation of the plugin.

            :arg name: Name of the plugin.
            :arg serialized_plugin: A buffer containing a serialized plugin.

            :returns: A new :class:`IPluginV2`
        """
    @typing.overload
    def deserialize_plugin(self: IPluginV2DynamicExt, name: str, serialized_plugin: bytes) -> IPluginV2DynamicExt:
        """
            Creates a plugin object from a serialized plugin.

            .. warning::
                This API only applies when implementing a Python-based plugin.

            `serialized_plugin` contains a serialized representation of the plugin.

            :arg name: Name of the plugin.
            :arg serialized_plugin: A string containing a serialized plugin.

            :returns: A new :class:`IPluginV2`
        """
class IPluginCreatorInterface(IVersionedInterface):
    """

        Base class for for plugin sub-interfaces.
    """
class IPluginCreatorV3One(IPluginCreatorInterface, IVersionedInterface):
    """

        Plugin creator class for user implemented layers

        :ivar name: :class:`str` Plugin name.
        :ivar plugin_version: :class:`str` Plugin version.
        :ivar field_names: :class:`list` List of fields that needs to be passed to :func:`create_plugin` .
        :ivar plugin_namespace: :class:`str` The namespace of the plugin creator based on the plugin library it belongs to. This can be set while registering the plugin creator.
    """
    field_names: PluginFieldCollection_
    name: str
    plugin_namespace: str
    plugin_version: str
    def __init__(self) -> None:
        ...
    def create_plugin(self, name: str, field_collection: PluginFieldCollection_, phase: ...) -> IPluginV3:
        """
            Creates a new plugin.

            :arg name: The name of the plugin.
            :arg field_collection: The :class:`PluginFieldCollection` for this plugin.

            :returns: :class:`IPluginV2` or :class:`None` on failure.
        """
class IPluginCreatorV3Quick(IPluginCreatorInterface, IVersionedInterface):
    field_names: PluginFieldCollection_
    name: str
    plugin_namespace: str
    plugin_version: str
    def __init__(self) -> None:
        ...
    def create_plugin(self, name: str, namespace: str, field_collection: PluginFieldCollection_, phase: ..., quickPluginType: ...) -> IPluginV3:
        ...
class IPluginRegistry:
    """

        Registers plugin creators.

        :ivar plugin_creator_list: [DEPRECATED] Deprecated in TensorRT 10.0. List of IPluginV2-descendent plugin creators in current registry.
        :ivar all_creators: List of all registered plugin creators of current registry.
        :ivar all_creators_recursive: List of all registered plugin creators of current registry and its parents (if :attr:`parent_search_enabled` is True).
        :ivar error_recorder: :class:`IErrorRecorder` Application-implemented error reporting interface for TensorRT objects.
        :ivar parent_search_enabled: bool variable indicating whether parent search is enabled. Default is True.
    """
    error_recorder: ...
    parent_search_enabled: bool
    def acquire_plugin_resource(self, key: str, resource: IPluginResource) -> IPluginResource:
        """
            Get a handle to a plugin resource registered against the provided key.

            :arg: key: Key for identifying the resource.
            :arg: resource: A plugin resource object. The object will only need to be valid until this method returns, as only a clone of this object will be registered by TRT. Cannot be null.
        """
    @typing.overload
    def deregister_creator(self, creator: IPluginCreator) -> bool:
        """
            Deregister a previously registered plugin creator inheriting from IPluginCreator.

            Since there may be a desire to limit the number of plugins,
            this function provides a mechanism for removing plugin creators registered in TensorRT.
            The plugin creator that is specified by ``creator`` is removed from TensorRT and no longer tracked.

            :arg creator: The IPluginCreator instance.

            :returns: ``True`` if the plugin creator was deregistered, ``False`` if it was not found in the registry
                    or otherwise could not be deregistered.
        """
    @typing.overload
    def deregister_creator(self, creator: IPluginCreatorInterface) -> bool:
        """
            Deregister a previously registered plugin creator.

            Since there may be a desire to limit the number of plugins,
            this function provides a mechanism for removing plugin creators registered in TensorRT.
            The plugin creator that is specified by ``creator`` is removed from TensorRT and no longer tracked.

            :arg creator: The plugin creator instance.

            :returns: ``True`` if the plugin creator was deregistered, ``False`` if it was not found in the registry
                    or otherwise could not be deregistered.
        """
    def deregister_library(self, handle: typing_extensions.CapsuleType) -> None:
        """
            Deregister plugins associated with a library. Any resources acquired when the library was loaded will be released.

            :arg: handle: the plugin library handle to deregister.
        """
    def get_creator(self, type: str, version: str, plugin_namespace: str = '') -> typing.Any:
        """
            Return plugin creator based on type, version and namespace

            :arg type: The type of the plugin.
            :arg version: The version of the plugin.
            :arg plugin_namespace: The namespace of the plugin.

            :returns: An :class:`IPluginCreator` .
        """
    def get_plugin_creator(self, type: str, version: str, plugin_namespace: str = '') -> IPluginCreator:
        """
            Return plugin creator based on type, version and namespace

            .. warning::
                Returns None if a plugin creator with matching name, version, and namespace is found, but is not a
                descendent of IPluginCreator

            :arg type: The type of the plugin.
            :arg version: The version of the plugin.
            :arg plugin_namespace: The namespace of the plugin.

            :returns: An :class:`IPluginCreator` .
        """
    def load_library(self, plugin_path: str) -> typing_extensions.CapsuleType:
        """
            Load and register a shared library of plugins.

            :arg: plugin_path: the plugin library path.

            :returns: The loaded plugin library handle. The call will fail and return None if any of the plugins are already registered.
        """
    @typing.overload
    def register_creator(self, creator: IPluginCreator, plugin_namespace: str = '') -> bool:
        """
            Register a plugin creator implementing IPluginCreator.

            :arg creator: The IPluginCreator instance.
            :arg plugin_namespace: The namespace of the plugin creator.

            :returns: False if any plugin creator with the same name, version and namespace is already registered.
        """
    @typing.overload
    def register_creator(self, creator: IPluginCreatorInterface, plugin_namespace: str = '') -> bool:
        """
            Register a plugin creator.

            :arg creator: The plugin creator instance.
            :arg plugin_namespace: The namespace of the plugin creator.

            :returns: False if any plugin creator with the same name, version and namespace is already registered..
        """
    def release_plugin_resource(self, key: str) -> int:
        """
            Decrement reference count for the resource with this key. If reference count goes to zero after decrement, release() will be invoked on the resource,
            and the key will be deregistered.

            :arg: key: Key that was used to register the resource.
        """
    @property
    def all_creators(self) -> list[typing.Any]:
        ...
    @property
    def all_creators_recursive(self) -> list[typing.Any]:
        ...
    @property
    def plugin_creator_list(self) -> list[IPluginCreator]:
        ...
class IPluginResource(IVersionedInterface):
    """

        Interface for plugins to define custom resources that could be shared through the plugin registry
    """
    def __init__(self) -> None:
        ...
    def clone(self) -> IPluginResource:
        """
            Resource initialization (if any) may be skipped for non-cloned objects since only clones will be
            registered by TensorRT.
        """
    def release(self) -> None:
        """
            This will only be called for IPluginResource objects that were produced from IPluginResource::clone().

            The IPluginResource object on which release() is called must still be in a clone-able state
            after release() returns.
        """
class IPluginResourceContext:
    """

        Interface for plugins to access per context resources provided by TensorRT

        There is no public way to construct an IPluginResourceContext. It appears as an argument to trt.IPluginV3OneRuntime.attach_to_context().
    """
    @property
    def error_recorder(self) -> ...:
        ...
    @property
    def gpu_allocator(self) -> ...:
        ...
class IPluginV2:
    """

        Plugin class for user-implemented layers.

        Plugins are a mechanism for applications to implement custom layers. When
        combined with IPluginCreator it provides a mechanism to register plugins and
        look up the Plugin Registry during de-serialization.


        :ivar num_outputs: :class:`int` The number of outputs from the layer. This is used by the implementations of :class:`INetworkDefinition` and :class:`Builder` . In particular, it is called prior to any call to :func:`initialize` .
        :ivar tensorrt_version: :class:`int` The API version with which this plugin was built.
        :ivar plugin_type: :class:`str` The plugin type. Should match the plugin name returned by the corresponding plugin creator
        :ivar plugin_version: :class:`str` The plugin version. Should match the plugin version returned by the corresponding plugin creator.
        :ivar plugin_namespace: :class:`str` The namespace that this plugin object belongs to. Ideally, all plugin objects from the same plugin library should have the same namespace.
        :ivar serialization_size: :class:`int` The size of the serialization buffer required.
    """
    plugin_namespace: str
    plugin_type: str
    plugin_version: str
    def clone(self) -> IPluginV2:
        """
            Clone the plugin object. This copies over internal plugin parameters and returns a new plugin object with these parameters.
        """
    def configure_with_format(self, input_shapes: collections.abc.Sequence[Dims], output_shapes: collections.abc.Sequence[Dims], dtype: DataType, format: ..., max_batch_size: typing.SupportsInt) -> None:
        """
            Configure the layer.

            This function is called by the :class:`Builder` prior to :func:`initialize` . It provides an opportunity for the layer to make algorithm choices on the basis of its weights, dimensions, and maximum batch size.

            The dimensions passed here do not include the outermost batch size (i.e. for 2D image networks, they will be 3D CHW dimensions).

            :arg input_shapes: The shapes of the input tensors.
            :arg output_shapes: The shapes of the output tensors.
            :arg dtype: The data type selected for the engine.
            :arg format: The format selected for the engine.
            :arg max_batch_size: The maximum batch size.
        """
    def destroy(self) -> None:
        """
            Destroy the plugin object. This will be called when the :class:`INetworkDefinition` , :class:`Builder` or :class:`ICudaEngine` is destroyed.
        """
    def execute_async(self, batch_size: typing.SupportsInt, inputs: collections.abc.Sequence[typing_extensions.CapsuleType], outputs: collections.abc.Sequence[typing_extensions.CapsuleType], workspace: typing_extensions.CapsuleType, stream_handle: typing.SupportsInt) -> int:
        """
            Execute the layer asynchronously.

            :arg batch_size: The number of inputs in the batch.
            :arg inputs: The memory for the input tensors.
            :arg outputs: The memory for the output tensors.
            :arg workspace: Workspace for execution.
            :arg stream_handle: The stream in which to execute the kernels.

            :returns: 0 for success, else non-zero (which will cause engine termination).
        """
    def get_output_shape(self, index: typing.SupportsInt, input_shapes: collections.abc.Sequence[Dims]) -> Dims:
        """
            Get the dimension of an output tensor.

            :arg index: The index of the output tensor.
            :arg input_shapes: The shapes of the input tensors.

            This function is called by the implementations of :class:`INetworkDefinition` and :class:`Builder` . In particular, it is called prior to any call to :func:`initialize` .
        """
    def get_workspace_size(self, max_batch_size: typing.SupportsInt) -> int:
        """
            Find the workspace size required by the layer.

            This function is called during engine startup, after :func:`initialize` . The workspace size returned should be sufficient for any batch size up to the maximum.

            :arg max_batch_size: :class:`int` The maximum possible batch size during inference.

            :returns: The workspace size.
        """
    def initialize(self) -> int:
        """
            Initialize the layer for execution. This is called when the engine is created.

            :returns: 0 for success, else non-zero (which will cause engine termination).
        """
    def serialize(self) -> memoryview:
        """
            Serialize the plugin.

            .. warning::
                This API only applies when called on a C++ plugin from a Python program.
        """
    def supports_format(self, dtype: DataType, format: ...) -> bool:
        """
            Check format support.

            This function is called by the implementations of :class:`INetworkDefinition` , :class:`Builder` , and :class:`ICudaEngine` . In particular, it is called when creating an engine and when deserializing an engine.

            :arg dtype: Data type requested.
            :arg format: TensorFormat requested.

            :returns: True if the plugin supports the type-format combination.
        """
    def terminate(self) -> None:
        """
            Release resources acquired during plugin layer initialization. This is called when the engine is destroyed.
        """
    @property
    def num_outputs(self) -> int:
        ...
    @num_outputs.setter
    def num_outputs(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def serialization_size(self) -> int:
        ...
    @property
    def tensorrt_version(self) -> int:
        ...
class IPluginV2DynamicExt(IPluginV2DynamicExtBase, IPluginV2):
    """

        Plugin class for user-implemented layers.

        Plugins are a mechanism for applications to implement custom layers.

        Similar to `IPluginV2Ext` (including capability to support different output data types), but with support for dynamic shapes.

        This class is made available for the purpose of implementing `IPluginV2DynamicExt` plugins with Python. Inherited
        Python->C++ bindings from `IPluginV2` and `IPluginV2Ext` will continue to work on C++-based `IPluginV2DynamicExt` plugins.

        .. note::
            Every attribute except `tensorrt_version` must be explicitly initialized on Python-based plugins. Except `plugin_namespace`,
            these attributes will be read-only when accessed through a C++-based plugin.

        :ivar num_outputs: :class:`int` The number of outputs from the plugin. This is used by the implementations of :class:`INetworkDefinition` and :class:`Builder`. In particular, it is called prior to any call to :func:`initialize`.
        :ivar tensorrt_version: :class:`int` [READ ONLY] The API version with which this plugin was built.
        :ivar plugin_type: :class:`str` The plugin type. Should match the plugin name returned by the corresponding plugin creator.
        :ivar plugin_version: :class:`str` The plugin version. Should match the plugin version returned by the corresponding plugin creator.
        :ivar plugin_namespace: :class:`str` The namespace that this plugin object belongs to. Ideally, all plugin objects from the same plugin library should have the same namespace.
        :ivar serialization_size: :class:`int` [READ ONLY] The size of the serialization buffer required.
    """
    FORMAT_COMBINATION_LIMIT: typing.ClassVar[int] = 100
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: IPluginV2DynamicExt) -> None:
        ...
    def clone(self) -> IPluginV2DynamicExt:
        """
            Clone the plugin object. This copies over internal plugin parameters as well and returns a new plugin object with these parameters.

            If the source plugin is pre-configured with `configure_plugin()`, the returned object should also be pre-configured.
            Cloned plugin objects can share the same per-engine immutable resource (e.g. weights) with the source object to avoid duplication.
        """
    def configure_plugin(self, pos: collections.abc.Sequence[DynamicPluginTensorDesc], in_out: collections.abc.Sequence[DynamicPluginTensorDesc]) -> None:
        """
            Configure the plugin.

            This function can be called multiple times in both the build and execution phases. The build phase happens before `initialize()` is called and only occurs during creation of an engine by `IBuilder`. The execution phase happens after `initialize()` is called and occurs during both creation of an engine by `IBuilder` and execution of an engine by `IExecutionContext`.

            Build phase: `configure_plugin()` is called when a plugin is being prepared for profiling but not for any specific input size. This provides an opportunity for the plugin to make algorithmic choices on the basis of input and output formats, along with the bound of possible dimensions. The min and max value of the `DynamicPluginTensorDesc` correspond to the `kMIN` and `kMAX` value of the current optimization profile that the plugin is being profiled for, with the `desc.dims` field corresponding to the dimensions of plugin specified at network creation. Wildcard dimensions will exist during this phase in the `desc.dims` field.

            Execution phase: `configure_plugin()` is called when a plugin is being prepared for executing the plugin for specific dimensions. This provides an opportunity for the plugin to change algorithmic choices based on the explicit input dimensions stored in `desc.dims` field.

            .. warning::
                This `configure_plugin()` method is not available to be called from Python on C++-based plugins

            :arg in: The input tensors attributes that are used for configuration.
            :arg out: The output tensors attributes that are used for configuration.
        """
    def destroy(self) -> None:
        """
            Destroy the plugin object. This will be called when the :class:`INetworkDefinition` , :class:`Builder` or :class:`ICudaEngine` is destroyed.

            .. note::
                When implementing a Python-based plugin, implementing this method is optional. The default behavior is a `pass`.
        """
    def enqueue(self, input_desc: collections.abc.Sequence[PluginTensorDesc], output_desc: collections.abc.Sequence[PluginTensorDesc], inputs: collections.abc.Sequence[typing.SupportsInt], outputs: collections.abc.Sequence[typing.SupportsInt], workspace: typing.SupportsInt, stream: typing.SupportsInt) -> None:
        """
            Execute the layer.

            `inputs` and `outputs` contains pointers to the corresponding input and output device buffers as their `intptr_t` casts. `stream` also represents an `intptr_t` cast of the CUDA stream in which enqueue should be executed.

            .. warning::
                Since input, output, and workspace buffers are created and owned by TRT, care must be taken when writing to them from the Python side.

            .. warning::
                In contrast to the C++ API for `enqueue()`, this method must not return an error code. The expected behavior is to throw an appropriate exception.
                if an error occurs.

            .. warning::
                This `enqueue()` method is not available to be called from Python on C++-based plugins.

            :arg input_desc:	how to interpret the memory for the input tensors.
            :arg output_desc:	how to interpret the memory for the output tensors.
            :arg inputs:	The memory for the input tensors.
            :arg outputs:   The memory for the output tensors.
            :arg workspace: Workspace for execution.
            :arg stream:	The stream in which to execute the kernels.
        """
    def get_output_datatype(self, index: typing.SupportsInt, input_types: collections.abc.Sequence[DataType]) -> DataType:
        """
            Return the `DataType` of the plugin output at the requested index.
            The default behavior should be to return the type of the first input, or `DataType::kFLOAT` if the layer has no inputs.
            The returned data type must have a format that is supported by the plugin.

            :arg index: Index of the output for which the data type is requested.
            :arg input_types: Data types of the inputs.

            :returns: `DataType` of the plugin output at the requested `index`.
        """
    def get_output_dimensions(self, output_index: typing.SupportsInt, inputs: collections.abc.Sequence[DimsExprs], expr_builder: IExprBuilder) -> DimsExprs:
        """
            Get expressions for computing dimensions of an output tensor from dimensions of the input tensors.

            This function is called by the implementations of `IBuilder` during analysis of the network.

            .. warning::
                This `get_output_dimensions()` method is not available to be called from Python on C++-based plugins

            :arg output_index:	The index of the output tensor
            :arg inputs:	Expressions for dimensions of the input tensors
            :arg expr_builder:	Object for generating new expressions

            :returns: Expression for the output dimensions at the given `output_index`.
        """
    def get_serialization_size(self) -> int:
        """
            Return the serialization size (in bytes) required by the plugin.

            .. note::
                When implementing a Python-based plugin, implementing this method is optional. The default behavior is equivalent to `return len(serialize())`.
        """
    def get_workspace_size(self, in: collections.abc.Sequence[PluginTensorDesc], out: collections.abc.Sequence[PluginTensorDesc]) -> int:
        """
            Return the workspace size (in bytes) required by the plugin.

            This function is called after the plugin is configured, and possibly during execution. The result should be a sufficient workspace size to deal with inputs and outputs of the given size or any smaller problem.

            .. note::
                When implementing a Python-based plugin, implementing this method is optional. The default behavior is equivalent to `return 0`.

            .. warning::
                This `get_workspace_size()` method is not available to be called from Python on C++-based plugins

            :arg input_desc: How to interpret the memory for the input tensors.
            :arg output_desc: How to interpret the memory for the output tensors.

            :returns: The workspace size (in bytes).
        """
    def initialize(self) -> int:
        """
            Initialize the plugin for execution. This is called when the engine is created.

            .. note::
                When implementing a Python-based plugin, implementing this method is optional. The default behavior is equivalent to `pass`.

            .. warning::
                In contrast to the C++ API for `initialize()`, this method must not return an error code. The expected behavior is to throw an appropriate exception
                if an error occurs.

            .. warning::
                This `initialize()` method is not available to be called from Python on C++-based plugins.
        """
    def serialize(self) -> bytes:
        """
            Serialize the plugin.

            .. warning::
                This API only applies when implementing a Python-based plugin.

            :returns: A bytes object containing the serialized representation of the plugin.
        """
    def supports_format_combination(self, pos: typing.SupportsInt, in_out: collections.abc.Sequence[PluginTensorDesc], num_inputs: typing.SupportsInt) -> bool:
        """
            Return true if plugin supports the format and datatype for the input/output indexed by pos.

            For this method, inputs are indexed from `[0, num_inputs-1]` and outputs are indexed from `[num_inputs, (num_inputs + num_outputs - 1)]`. `pos` is an index into `in_ou`t, where `0 <= pos < (num_inputs + num_outputs - 1)`.

            TensorRT invokes this method to query if the input/output tensor indexed by `pos` supports the format and datatype specified by `in_out[pos].format` and `in_out[pos].type`. The override shall return true if that format and datatype at `in_out[pos]` are supported by the plugin. It is undefined behavior to examine the format or datatype or any tensor that is indexed by a number greater than `pos`.

            .. warning::
                This `supports_format_combination()` method is not available to be called from Python on C++-based plugins

            :arg pos: The input or output tensor index being queried.
            :arg in_out: The combined input and output tensor descriptions.
            :arg num_inputs: The number of inputs.

            :returns: boolean indicating whether the format combination is supported or not.
        """
    def terminate(self) -> None:
        """
            Release resources acquired during plugin layer initialization. This is called when the engine is destroyed.

            .. note::
                When implementing a Python-based plugin, implementing this method is optional. The default behavior is equivalent to `pass`.
        """
class IPluginV2DynamicExtBase(IPluginV2):
    pass
class IPluginV2Ext(IPluginV2):
    """

        Plugin class for user-implemented layers.

        Plugins are a mechanism for applications to implement custom layers. This interface provides additional capabilities to the IPluginV2 interface by supporting different output data types.

        :ivar tensorrt_version: :class:`int` The API version with which this plugin was built.
    """
    def attach_to_context(self, cudnn: typing_extensions.CapsuleType, cublas: typing_extensions.CapsuleType, allocator: typing_extensions.CapsuleType) -> None:
        """
            Attach the plugin object to an execution context and grant the plugin the access to some context resource.

            :arg cudnn: The cudnn context handle of the execution context
            :arg cublas: The cublas context handle of the execution context
            :arg allocator: The allocator used by the execution context

            This function is called automatically for each plugin when a new execution context is created. If the plugin needs per-context resource, it can be allocated here. The plugin can also get context-owned CUDNN and CUBLAS context here.
        """
    def clone(self) -> IPluginV2Ext:
        """
            Clone the plugin object. This copies over internal plugin parameters as well and returns a new plugin object with these parameters.

            If the source plugin is pre-configured with `configure_plugin()`, the returned object should also be pre-configured. The returned object should allow attach_to_context() with a new execution context.
            Cloned plugin objects can share the same per-engine immutable resource (e.g. weights) with the source object (e.g. via ref-counting) to avoid duplication.
        """
    def configure_plugin(self, input_shapes: collections.abc.Sequence[Dims], output_shapes: collections.abc.Sequence[Dims], input_types: collections.abc.Sequence[DataType], output_types: collections.abc.Sequence[DataType], input_is_broadcasted: collections.abc.Sequence[bool], output_is_broacasted: collections.abc.Sequence[bool], format: ..., max_batch_size: typing.SupportsInt) -> None:
        """
            Configure the layer.

            This function is called by the :class:`Builder` prior to :func:`initialize` . It provides an opportunity for the layer to make algorithm choices on the basis of its weights, dimensions, and maximum batch size.

            The dimensions passed here do not include the outermost batch size (i.e. for 2D image networks, they will be 3D CHW dimensions).

            :arg input_shapes: The shapes of the input tensors.
            :arg output_shapes: The shapes of the output tensors.
            :arg input_types: The data types of the input tensors.
            :arg output_types: The data types of the output tensors.
            :arg input_is_broadcasted: Whether an input is broadcasted across the batch.
            :arg output_is_broadcasted: Whether an output is broadcasted across the batch.
            :arg format: The format selected for floating-point inputs and outputs of the engine.
            :arg max_batch_size: The maximum batch size.
        """
    def detach_from_context(self) -> None:
        """
            Detach the plugin object from its execution context.

            This function is called automatically for each plugin when a execution context is destroyed. If the plugin owns per-context resource, it can be released here.
        """
    def get_output_data_type(self, index: typing.SupportsInt, input_types: collections.abc.Sequence[DataType]) -> DataType:
        """
            Return the DataType of the plugin output at the requested index.
            The default behavior should be to return the type of the first input, or `DataType::kFLOAT` if the layer has no inputs.
            The returned data type must have a format that is supported by the plugin.

            :arg index: Index of the output for which data type is requested.
            :arg input_types: Data types of the inputs.

            :returns: DataType of the plugin output at the requested index.
        """
class IPluginV2Layer(ILayer):
    """

            A plugin layer in an :class:`INetworkDefinition` .

            :ivar plugin: :class:`IPluginV2` The plugin for the layer.
    """
    @property
    def plugin(self) -> IPluginV2:
        ...
class IPluginV3(IVersionedInterface):
    """

        Plugin class for the V3 generation of user-implemented layers.

        IPluginV3 acts as a wrapper around the plugin capability interfaces that define the actual behavior of the plugin.

        This class is made available for the purpose of implementing `IPluginV3` plugins with Python.

        .. note::
            Every attribute must be explicitly initialized on Python-based plugins.
            These attributes will be read-only when accessed through a C++-based plugin.

    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: IPluginV3) -> None:
        ...
    def clone(self) -> IPluginV3:
        """
            Clone the plugin object. This copies over internal plugin parameters as well and returns a new plugin object with these parameters.

            If the source plugin is pre-configured with `configure_plugin()`, the returned object should also be pre-configured.
            Cloned plugin objects can share the same per-engine immutable resource (e.g. weights) with the source object to avoid duplication.
        """
    def destroy(self) -> None:
        """
            Perform any cleanup or resource release(s) needed before plugin object is destroyed. This will be called when the :class:`INetworkDefinition` , :class:`Builder` or :class:`ICudaEngine` is destroyed.

            .. note::
                There is no direct equivalent to this method in the C++ API.

            .. note::
                Implementing this method is optional. The default behavior is a `pass`.
        """
    def get_capability_interface(self, type: ...) -> typing.Any:
        """
            Return a plugin object implementing the specified PluginCapabilityType.

            .. note::
                IPluginV3 objects added for the build phase (through add_plugin_v3()) must return valid objects for PluginCapabilityType.CORE, PluginCapabilityType.BUILD and PluginCapabilityType.RUNTIME.

            .. note::
                IPluginV3 objects added for the runtime phase must return valid objects for PluginCapabilityType.CORE and PluginCapabilityType.RUNTIME.
        """
class IPluginV3Layer(ILayer):
    """

            A plugin layer in an :class:`INetworkDefinition` .

            :ivar plugin: :class:`IPluginV3` The plugin for the layer.
    """
    @property
    def plugin(self) -> IPluginV3:
        ...
class IPluginV3OneBuild(IPluginCapability, IVersionedInterface):
    """

        A plugin capability interface that enables the build capability (PluginCapabilityType.BUILD).

        Exposes methods that allow the expression of the build time properties and behavior of a plugin.

        .. note::
            Every attribute must be explicitly initialized on Python-based plugins.
            These attributes will be read-only when accessed through a C++-based plugin.

        :ivar num_outputs: :class:`int` The number of outputs from the plugin. This is used by the implementations of :class:`INetworkDefinition` and :class:`Builder`.
        :ivar format_combination_limit: :class:`int` The maximum number of format combinations that the plugin supports.
        :ivar metadata_string: :class:`str` The metadata string for the plugin.
        :ivar timing_cache_id: :class:`str` The timing cache ID for the plugin.

    """
    DEFAULT_FORMAT_COMBINATION_LIMIT: typing.ClassVar[int] = 100
    metadata_string: str
    timing_cache_id: str
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: IPluginV3OneBuild) -> None:
        ...
    def configure_plugin(self: IPluginV3, in: collections.abc.Sequence[DynamicPluginTensorDesc], out: collections.abc.Sequence[DynamicPluginTensorDesc]) -> None:
        """
            Configure the plugin.

            This function can be called multiple times in the build phase during creation of an engine by IBuilder.

            Build phase: `configure_plugin()` is called when a plugin is being prepared for profiling but not for any specific input size. This provides an opportunity for the plugin to make algorithmic choices on the basis of input and output formats, along with the bound of possible dimensions. The min, opt and max value of the
            `DynamicPluginTensorDesc` correspond to the `MIN`, `OPT` and `MAX` value of the current profile that the plugin is
            being profiled for, with the desc.dims field corresponding to the dimensions of plugin specified at network
            creation. Wildcard dimensions may exist during this phase in the desc.dims field.

            .. warning::
                In contrast to the C++ API for `configurePlugin()`, this method must not return an error code. The expected behavior is to throw an appropriate exception
                if an error occurs.

            .. warning::
                This `configure_plugin()` method is not available to be called from Python on C++-based plugins

            :arg in: The input tensors attributes that are used for configuration.
            :arg out: The output tensors attributes that are used for configuration.
        """
    def get_output_data_types(self: IPluginV3, input_types: collections.abc.Sequence[DataType]) -> list[DataType]:
        """
            Return `DataType` s of the plugin outputs.

            Provide `DataType.FLOAT` s if the layer has no inputs. The data type for any size tensor outputs must be
            `DataType.INT32`. The returned data types must each have a format that is supported by the plugin.

            :arg input_types: Data types of the inputs.

            :returns: `DataType` of the plugin output at the requested `index`.
        """
    def get_output_shapes(self: IPluginV3, inputs: collections.abc.Sequence[DimsExprs], shape_inputs: collections.abc.Sequence[DimsExprs], expr_builder: IExprBuilder) -> list[DimsExprs]:
        """
            Get expressions for computing shapes of an output tensor from shapes of the input tensors.

            This function is called by the implementations of `IBuilder` during analysis of the network.

            .. warning::
                This get_output_shapes() method is not available to be called from Python on C++-based plugins

            :arg inputs:	Expressions for shapes of the input tensors
            :arg shape_inputs:	Expressions for shapes of the shape inputs
            :arg expr_builder:	Object for generating new expressions

            :returns: Expressions for the output shapes.
        """
    def get_valid_tactics(self: IPluginV3) -> list[int]:
        """
            Return any custom tactics that the plugin intends to use.

            .. note::
                The provided tactic values must be unique and positive

            .. warning::
                This `get_valid_tactics()` method is not available to be called from Python on C++-based plugins.
        """
    def get_workspace_size(self: IPluginV3, in: collections.abc.Sequence[DynamicPluginTensorDesc], out: collections.abc.Sequence[DynamicPluginTensorDesc]) -> int:
        """
            Return the workspace size (in bytes) required by the plugin.

            This function is called after the plugin is configured, and possibly during execution. The result should be a sufficient workspace size to deal with inputs and outputs of the given size or any smaller problem.

            .. note::
                When implementing a Python-based plugin, implementing this method is optional. The default behavior is equivalent to `return 0`.

            .. warning::
                This `get_workspace_size()` method is not available to be called from Python on C++-based plugins

            :arg input_desc: How to interpret the memory for the input tensors.
            :arg output_desc: How to interpret the memory for the output tensors.

            :returns: The workspace size (in bytes).
        """
    def supports_format_combination(self: IPluginV3, pos: typing.SupportsInt, in_out: collections.abc.Sequence[DynamicPluginTensorDesc], num_inputs: typing.SupportsInt) -> bool:
        """
            Return true if plugin supports the format and datatype for the input/output indexed by pos.

            For this method, inputs are indexed from `[0, num_inputs-1]` and outputs are indexed from `[num_inputs, (num_inputs + num_outputs - 1)]`. `pos` is an index into `in_ou`t, where `0 <= pos < (num_inputs + num_outputs - 1)`.

            TensorRT invokes this method to query if the input/output tensor indexed by `pos` supports the format and datatype specified by `in_out[pos].format` and `in_out[pos].type`. The override shall return true if that format and datatype at `in_out[pos]` are supported by the plugin. It is undefined behavior to examine the format or datatype or any tensor that is indexed by a number greater than `pos`.

            .. warning::
                This `supports_format_combination()` method is not available to be called from Python on C++-based plugins

            :arg pos: The input or output tensor index being queried.
            :arg in_out: The combined input and output tensor descriptions.
            :arg num_inputs: The number of inputs.

            :returns: boolean indicating whether the format combination is supported or not.
        """
    @property
    def format_combination_limit(self) -> int:
        ...
    @format_combination_limit.setter
    def format_combination_limit(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def num_outputs(self) -> int:
        ...
    @num_outputs.setter
    def num_outputs(self, arg1: typing.SupportsInt) -> None:
        ...
class IPluginV3OneBuildV2(IPluginV3OneBuild, IPluginCapability, IVersionedInterface):
    """

        A plugin capability interface that extends IPluginV3OneBuild by providing I/O aliasing functionality.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: IPluginV3OneBuildV2) -> None:
        ...
    def get_aliased_input(self: typing.SupportsInt) -> int:
        """
            Return any custom tactics that the plugin intends to use.

            .. note::
                The provided tactic values must be unique and positive

            .. warning::
                This `get_valid_tactics()` method is not available to be called from Python on C++-based plugins.
        """
class IPluginV3OneCore(IPluginCapability, IVersionedInterface):
    """

        A plugin capability interface that enables the core capability (PluginCapabilityType.CORE).

        .. note::
            Every attribute must be explicitly initialized on Python-based plugins.
            These attributes will be read-only when accessed through a C++-based plugin.

        :ivar plugin_name: :class:`str` The plugin name. Should match the plugin name returned by the corresponding plugin creator.
        :ivar plugin_version: :class:`str` The plugin version. Should match the plugin version returned by the corresponding plugin creator.
        :ivar plugin_namespace: :class:`str` The namespace that this plugin object belongs to. Ideally, all plugin objects from the same plugin library should have the same namespace.
    """
    plugin_name: str
    plugin_namespace: str
    plugin_version: str
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: IPluginV3OneCore) -> None:
        ...
class IPluginV3OneRuntime(IPluginCapability, IVersionedInterface):
    """

        A plugin capability interface that enables the runtime capability (PluginCapabilityType.RUNTIME).

        Exposes methods that allow the expression of the runtime properties and behavior of a plugin.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: IPluginV3OneRuntime) -> None:
        ...
    def attach_to_context(self: IPluginV3, resource_context: ...) -> IPluginV3:
        """
            Clone the plugin, attach the cloned plugin object to a execution context and grant the cloned plugin access to some context resources.

            This function is called automatically for each plugin when a new execution context is created.

            The plugin may use resources provided by the resource_context until the plugin is deleted by TensorRT.

            :arg resource_context: A resource context that exposes methods to get access to execution context specific resources. A different resource context is guaranteed for each different execution context to which the plugin is attached.

            .. note::
                This method should clone the entire IPluginV3 object, not just the runtime interface
        """
    def enqueue(self: IPluginV3, input_desc: collections.abc.Sequence[PluginTensorDesc], output_desc: collections.abc.Sequence[PluginTensorDesc], inputs: collections.abc.Sequence[typing.SupportsInt], outputs: collections.abc.Sequence[typing.SupportsInt], workspace: typing.SupportsInt, stream: typing.SupportsInt) -> None:
        """
            Execute the layer.

            `inputs` and `outputs` contains pointers to the corresponding input and output device buffers as their `intptr_t` casts. `stream` also represents an `intptr_t` cast of the CUDA stream in which enqueue should be executed.

            .. warning::
                Since input, output, and workspace buffers are created and owned by TRT, care must be taken when writing to them from the Python side.

            .. warning::
                In contrast to the C++ API for `enqueue()`, this method must not return an error code. The expected behavior is to throw an appropriate exception.
                if an error occurs.

            .. warning::
                This `enqueue()` method is not available to be called from Python on C++-based plugins.

            :arg input_desc:	how to interpret the memory for the input tensors.
            :arg output_desc:	how to interpret the memory for the output tensors.
            :arg inputs:	The memory for the input tensors.
            :arg outputs:   The memory for the output tensors.
            :arg workspace: Workspace for execution.
            :arg stream:	The stream in which to execute the kernels.
        """
    def get_fields_to_serialize(self: IPluginV3) -> PluginFieldCollection_:
        """
            Return the plugin fields which should be serialized.

            .. note::
                The set of plugin fields returned does not necessarily need to match that advertised through get_field_names() of the corresponding plugin creator.

            .. warning::
                This `get_fields_to_serialize()` method is not available to be called from Python on C++-based plugins.
        """
    def on_shape_change(self: IPluginV3, in: collections.abc.Sequence[PluginTensorDesc], out: collections.abc.Sequence[PluginTensorDesc]) -> None:
        """
            Called when a plugin is being prepared for execution for specific dimensions. This could happen multiple times in the execution phase, both during creation of an engine by IBuilder and execution of an
            engine by IExecutionContext.

             * IBuilder will call this function once per profile, with `in` resolved to the values specified by the kOPT field of the current profile.
             * IExecutionContext will call this during the next subsequent instance of enqueue_v2() or execute_v3() if: (1) The optimization profile is changed (2). An input binding is changed.

            .. warning::
                In contrast to the C++ API for `onShapeChange()`, this method must not return an error code. The expected behavior is to throw an appropriate exception
                if an error occurs.

            .. warning::
                This `on_shape_change()` method is not available to be called from Python on C++-based plugins

            :arg in: The input tensors attributes that are used for configuration.
            :arg out: The output tensors attributes that are used for configuration.
        """
    def set_tactic(self: IPluginV3, tactic: typing.SupportsInt) -> None:
        """
            Set the tactic to be used in the subsequent call to enqueue().

            If no custom tactics were advertised, this will have a value of 0, which is designated as the default tactic.

            .. warning::
                In contrast to the C++ API for `setTactic()`, this method must not return an error code. The expected behavior is to throw an appropriate exception
                if an error occurs.

            .. warning::
                This `set_tactic()` method is not available to be called from Python on C++-based plugins.
        """
class IPluginV3QuickAOTBuild(IPluginV3QuickBuild, IPluginCapability, IVersionedInterface):
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: IPluginV3QuickAOTBuild) -> None:
        ...
class IPluginV3QuickBuild(IPluginCapability, IVersionedInterface):
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: IPluginV3QuickBuild) -> None:
        ...
class IPluginV3QuickCore(IPluginCapability, IVersionedInterface):
    plugin_name: str
    plugin_namespace: str
    plugin_version: str
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: IPluginV3QuickCore) -> None:
        ...
class IPluginV3QuickRuntime(IPluginCapability, IVersionedInterface):
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: IPluginV3QuickRuntime) -> None:
        ...
class IPoolingLayer(ILayer):
    """

        A Pooling layer in an :class:`INetworkDefinition` . The layer applies a reduction operation within a window over the input.

        :ivar type: :class:`PoolingType` The type of pooling to be performed.
        :ivar pre_padding: :class:`DimsHW` The pre-padding. The start of input will be zero-padded by this number of elements in the height and width directions. Default: (0, 0)
        :ivar post_padding: :class:`DimsHW` The post-padding. The end of input will be zero-padded by this number of elements in the height and width directions. Default: (0, 0)
        :ivar padding_mode: :class:`PaddingMode` The padding mode. Padding mode takes precedence if both :attr:`IPoolingLayer.padding_mode` and either :attr:`IPoolingLayer.pre_padding` or :attr:`IPoolingLayer.post_padding` are set.
        :ivar blend_factor: :class:`float` The blending factor for the max_average_blend mode: :math:`max_average_blendPool = (1-blendFactor)*maxPool + blendFactor*avgPool` . ``blend_factor`` is a user value in [0,1] with the default value of 0.0. This value only applies for the :const:`PoolingType.MAX_AVERAGE_BLEND` mode.
        :ivar average_count_excludes_padding: :class:`bool` Whether average pooling uses as a denominator the overlap area between the window and the unpadded input. If this is not set, the denominator is the overlap between the pooling window and the padded input. Default: True
        :ivar window_size_nd: :class:`Dims` The multi-dimension window size for pooling.
        :ivar stride_nd: :class:`Dims` The multi-dimension stride for pooling. Default: (1, ..., 1)
        :ivar padding_nd: :class:`Dims` The multi-dimension padding for pooling. Default: (0, ..., 0)
    """
    average_count_excludes_padding: bool
    padding_mode: PaddingMode
    padding_nd: Dims
    post_padding: Dims
    pre_padding: Dims
    stride_nd: Dims
    type: PoolingType
    window_size_nd: Dims
    @property
    def blend_factor(self) -> float:
        ...
    @blend_factor.setter
    def blend_factor(self, arg1: typing.SupportsFloat) -> None:
        ...
class IProfiler:
    """

        Abstract base Profiler class.

        To implement a custom profiler, ensure that you explicitly instantiate the base class in :func:`__init__` :
        ::

            class MyProfiler(trt.IProfiler):
                def __init__(self):
                    trt.IProfiler.__init__(self)

                def report_layer_time(self, layer_name, ms):
                    ... # Your implementation here

        When this class is added to an :class:`IExecutionContext`, the profiler will be called once per layer for each invocation of :func:`IExecutionContext.execute_v2()`.

        It is not recommended to run inference with profiler enabled when the inference execution time is critical since the profiler may affect execution time negatively.
    """
    def __init__(self) -> None:
        ...
    def report_layer_time(self, layer_name: str, ms: typing.SupportsFloat) -> None:
        """
            Reports time in milliseconds for each layer. This function must be overriden a derived class.

            :arg layer_name: The name of the layer, set when constructing the :class:`INetworkDefinition` . If the engine is built with profiling verbosity set to NONE, the layerName is the decimal index of the layer.
            :arg ms: The time in milliseconds to execute the layer.
        """
class IProgressMonitor:
    """

        Application-implemented progress reporting interface for TensorRT.

        The IProgressMonitor is a user-defined object that TensorRT uses to report back when an internal algorithm has
        started or finished a phase to help provide feedback on the progress of the optimizer.

        The IProgressMonitor will trigger its start function when a phase is entered and will trigger its finish function
        when that phase is exited. Each phase consists of one or more steps. When each step is completed, the step_complete
        function is triggered. This will allow an application using the builder to communicate progress relative to when the
        optimization step is expected to complete.

        The implementation of IProgressMonitor must be thread-safe so that it can be called from multiple internal threads.
        The lifetime of the IProgressMonitor must exceed the lifetime of all TensorRT objects that use it.
    """
    def __init__(self) -> None:
        ...
    def phase_finish(self, phase_name: str) -> None:
        """
            Signal that a phase of the optimizer has finished.

            :arg phase_name: The name of the phase that has finished.

            The phase_finish function signals to the application that the phase is complete. This function may be called before
            all steps in the range [0, num_steps) have been reported to step_complete. This scenario can be triggered by error
            handling, internal optimizations, or when step_complete returns False to request cancellation of the build.
        """
    def phase_start(self, phase_name: str, parent_phase: str, num_steps: typing.SupportsInt) -> None:
        """
            Signal that a phase of the optimizer has started.

            :arg phase_name: The name of this phase for tracking purposes.
            :arg parent_phase: The parent phase that this phase belongs to, None if there is no parent.
            :arg num_steps: The number of steps that are involved in this phase.

            The phase_start function signals to the application that the current phase is beginning, and that it has a
            certain number of steps to perform. If phase_parent is None, then the phase_start is beginning an
            independent phase, and if phase_parent is specified, then the current phase, specified by phase_name, is
            within the scope of the parent phase. num_steps will always be a positive number. The phase_start function
            implies that the first step is being executed. TensorRT will signal when each step is complete.

            Phase names are human readable English strings which are unique within a single phase hierarchy but which can be
            reused once the previous instance has completed. Phase names and their hierarchies may change between versions
            of TensorRT.
        """
    def step_complete(self, phase_name: str, step: typing.SupportsInt) -> bool:
        """
            Signal that a step of an optimizer phase has finished.

            :arg phase_name: The name of the innermost phase being executed.
            :arg step: The step number that was completed.

            The step_complete function signals to the application that TensorRT has finished the current step for the phase
            ``phase_name`` , and will move on to the next step if there is one. The application can return False for TensorRT to exit
            the build early. The step value will increase on subsequent calls in the range [0, num_steps).

            :returns: True to continue to the next step or False to stop the build.
        """
class IQuantizeLayer(ILayer):
    """

        A Quantize layer in an :class:`INetworkDefinition` .

        This layer accepts a floating-point data input tensor, and uses the scale and zeroPt inputs to

        quantize the data to an 8-bit signed integer according to:

        :math:`output = clamp(round(input / scale) + zeroPt)`

        Rounding type is rounding-to-nearest ties-to-even (https://en.wikipedia.org/wiki/Rounding#Round_half_to_even).

        Clamping is in the range [-128, 127].

        The first input (index 0) is the tensor to be quantized.
        The second (index 1) and third (index 2) are the scale and zero point respectively.
        Each of scale and zeroPt must be either a scalar, or a 1D tensor.

        The zeroPt tensor is optional, and if not set, will be assumed to be zero.  Its data type must be
        tensorrt.int8. zeroPt must only contain zero-valued coefficients, because only symmetric quantization is
        supported.
        The scale value must be either a scalar for per-tensor quantization, or a 1D tensor for per-axis
        quantization. The size of the 1-D scale tensor must match the size of the quantization axis. The size of the
        scale must match the size of the zeroPt.

        The subgraph which terminates with the scale tensor must be a build-time constant.  The same restrictions apply
        to the zeroPt.
        The output type, if constrained, must be constrained to tensorrt.int8 or tensorrt.fp8. The input type, if constrained, must be
        constrained to tensorrt.float32, tensorrt.float16 or tensorrt.bfloat16.
        The output size is the same as the input size.

        IQuantizeLayer supports tensorrt.float32, tensorrt.float16 and tensorrt.bfloat16 precision and will default to tensorrt.float32 precision during instantiation.
        IQuantizeLayer supports tensorrt.int8, tensorrt.float8, tensorrt.int4 and tensorrt.fp4 output.

        :ivar axis: :class:`int` The axis along which quantization occurs. The quantization axis is in reference to the input tensor's dimensions.

        :ivar to_type: :class:`DataType` The specified data type of the output tensor. Must be tensorrt.int8 or tensorrt.float8.
    """
    block_shape: Dims
    to_type: DataType
    @property
    def axis(self) -> int:
        ...
    @axis.setter
    def axis(self, arg1: typing.SupportsInt) -> None:
        ...
class IRaggedSoftMaxLayer(ILayer):
    """

        A ragged softmax layer in an :class:`INetworkDefinition` .

        This layer takes a ZxS input tensor and an additional Zx1 bounds tensor holding the lengths of the Z sequences.

        This layer computes a softmax across each of the Z sequences.

        The output tensor is of the same size as the input tensor.
    """
class IRecurrenceLayer(ILoopBoundaryLayer):
    """
    """
    def set_input(self: ILayer, index: typing.SupportsInt, tensor: ITensor) -> None:
        """
            Set the first or second input.
            If index==1 and the number of inputs is one, the input is appended.
            The first input specifies the initial output value, and must come from outside the loop.
            The second input specifies the next output value, and must come from inside the loop.
            The two inputs must have the same dimensions.

            :param index: The index of the input to set.
            :param tensor: The input tensor.
        """
class IReduceLayer(ILayer):
    """

        A reduce layer in an :class:`INetworkDefinition` .

        :ivar op: :class:`ReduceOperation` The reduce operation for the layer.
        :ivar axes: :class:`int` The axes over which to reduce.
        :ivar keep_dims: :class:`bool` Specifies whether or not to keep the reduced dimensions for the layer.
    """
    keep_dims: bool
    op: ReduceOperation
    @property
    def axes(self) -> int:
        ...
    @axes.setter
    def axes(self, arg1: typing.SupportsInt) -> None:
        ...
class IResizeLayer(ILayer):
    """

        A resize layer in an :class:`INetworkDefinition` .

        Resize layer can be used for resizing a N-D tensor.

        Resize layer currently supports the following configurations:

        * InterpolationMode.NEAREST - resizes innermost `m` dimensions of N-D, where 0 < m <= min(3, N) and N > 0.
        * InterpolationMode.LINEAR - resizes innermost `m` dimensions of N-D, where 0 < m <= min(3, N) and N > 0.
        * InterpolationMode.CUBIC - resizes innermost `2` dimensions of N-D, N >= 2.

        Default resize mode is InterpolationMode.NEAREST.

        Resize layer provides two ways to resize tensor dimensions:

        * Set output dimensions directly. It can be done for static as well as dynamic resize layer.
            Static resize layer requires output dimensions to be known at build-time.
            Dynamic resize layer requires output dimensions to be set as one of the input tensors.

        * Set scales for resize. Each output dimension is calculated as floor(input dimension * scale).
            Only static resize layer allows setting scales where the scales are known at build-time.

        If executing this layer on DLA, the following combinations of parameters are supported:

        - In NEAREST mode:

           * (ResizeCoordinateTransformation.ASYMMETRIC, ResizeSelector.FORMULA, ResizeRoundMode.FLOOR)
           * (ResizeCoordinateTransformation.HALF_PIXEL, ResizeSelector.FORMULA, ResizeRoundMode.HALF_DOWN)
           * (ResizeCoordinateTransformation.HALF_PIXEL, ResizeSelector.FORMULA, ResizeRoundMode.HALF_UP)

        - In LINEAR and CUBIC mode:

           * (ResizeCoordinateTransformation.HALF_PIXEL, ResizeSelector.FORMULA)
           * (ResizeCoordinateTransformation.HALF_PIXEL, ResizeSelector.UPPER)


        :ivar shape: :class:`Dims` The output dimensions. Must to equal to input dimensions size.
        :ivar scales: :class:`List[float]` List of resize scales.
            If executing this layer on DLA, there are three restrictions:
            1. ``len(scales)`` has to be exactly 4.
            2. The first two elements in scales need to be exactly 1 (for unchanged batch and channel dimensions).
            3. The last two elements in scales, representing the scale values along height and width dimensions,
            respectively, need to be integer values in the range of [1, 32] for NEAREST mode and [1, 4] for LINEAR.
            Example of DLA-supported scales: [1, 1, 2, 2].
        :ivar resize_mode: :class:`InterpolationMode` Resize mode can be Linear, Cubic or Nearest.
        :ivar coordinate_transformation: :class:`ResizeCoordinateTransformationDoc` Supported resize coordinate transformation modes are ALIGN_CORNERS, ASYMMETRIC and HALF_PIXEL.
        :ivar selector_for_single_pixel: :class:`ResizeSelector` Supported resize selector modes are FORMULA and UPPER.
        :ivar nearest_rounding: :class:`ResizeRoundMode` Supported resize Round modes are HALF_UP, HALF_DOWN, FLOOR and CEIL.
        :ivar exclude_outside: :class:`int` If set to 1, the weight of sampling locations outside the input tensor will be set to 0, and the weight will be renormalized so that their sum is 1.0.
        :ivar cubic_coeff: :class:`float` coefficient 'a' used in cubic interpolation.
    """
    coordinate_transformation: ResizeCoordinateTransformation
    exclude_outside: bool
    nearest_rounding: ResizeRoundMode
    resize_mode: InterpolationMode
    selector_for_single_pixel: ResizeSelector
    shape: Dims
    def set_input(self: ILayer, index: typing.SupportsInt, tensor: ITensor) -> None:
        """
            Sets the input tensor for the given index.

            If index == 1 and num_inputs == 1, num_inputs changes to 2.
            Once such additional input is set, resize layer works in dynamic mode.
            When index == 1 and num_inputs == 1, the output dimensions are used from
            the input tensor, overriding the dimensions supplied by `shape`.

            :arg index: The index of the input tensor.
            :arg tensor: The input tensor.
        """
    @property
    def cubic_coeff(self) -> float:
        ...
    @cubic_coeff.setter
    def cubic_coeff(self, arg1: typing.SupportsFloat) -> None:
        ...
    @property
    def scales(self) -> list[float]:
        ...
    @scales.setter
    def scales(self, arg1: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        ...
class IReverseSequenceLayer(ILayer):
    """

        A ReverseSequence layer in an :class:`INetworkDefinition` .

        This layer performs batch-wise reversal, which slices the input tensor along the axis ``batch_axis``. For the
        ``i``-th slice, the operation reverses the first ``N`` elements, specified by the corresponding ``i``-th value
        in ``sequence_lens``, along ``sequence_axis`` and keeps the remaining elements unchanged. The output tensor will
        have the same shape as the input tensor.

        :ivar batch_axis: :class:`int` The batch axis. Default: 1.
        :ivar sequence_axis: :class:`int` The sequence axis. Default: 0.
    """
    @property
    def batch_axis(self) -> int:
        ...
    @batch_axis.setter
    def batch_axis(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def sequence_axis(self) -> int:
        ...
    @sequence_axis.setter
    def sequence_axis(self, arg1: typing.SupportsInt) -> None:
        ...
class IRotaryEmbeddingLayer(ILayer):
    """

        A RotaryEmbedding layer in :class:`INetworkDefinition`.

        :ivar interleaved: :class:`bool` Specifies whether the input tensor is in interleaved format, i.e., whether the 2-d vectors rotated are taken from adjacent 2 elements in the hidden dimension.
        :ivar rotary_embedding_dim: :class:`int` Specifies the hidden dimension that participates in RoPE.
    """
    interleaved: bool
    def set_input(self: ILayer, index: typing.SupportsInt, tensor: ITensor) -> None:
        """
            Set the input tensor specified by the given index.

            :arg index: The index of the input tensor.
            :arg tensor: The input tensor.

            The indices are as follows:

            Input 0 is the input activation tensor.
            Input 1 is the cosine cache tensor.
            Input 2 is the sine cache tensor.
            Input 3 (optional) is the positionIds tensor, which is used for indexing into the cosine and sine caches.
        """
    @property
    def rotary_embedding_dim(self) -> int:
        ...
    @rotary_embedding_dim.setter
    def rotary_embedding_dim(self, arg1: typing.SupportsInt) -> bool:
        ...
class IRuntimeConfig:
    """

        A runtime configuration for an :class:`ICudaEngine` .
    """
    def get_execution_context_allocation_strategy(self) -> ExecutionContextAllocationStrategy:
        """
            Get the execution context allocation strategy.

            :returns: The execution context allocation strategy.
        """
    def set_execution_context_allocation_strategy(self, strategy: ExecutionContextAllocationStrategy = ...) -> None:
        """
            Set the execution context allocation strategy.

            :arg strategy: The execution context allocation strategy.
        """
class IScaleLayer(ILayer):
    """

        A Scale layer in an :class:`INetworkDefinition` .

        This layer applies a per-element computation to its input:

        :math:`output = (input * scale + shift) ^ {power}`

        The coefficients can be applied on a per-tensor, per-channel, or per-element basis.

        **Note**
        If the number of weights is 0, then a default value is used for shift, power, and scale. The default shift is 0, the default power is 1, and the default scale is 1.

        The output size is the same as the input size.

        **Note**
        The input tensor for this layer is required to have a minimum of 3 dimensions.

        :ivar mode: :class:`ScaleMode` The scale mode.
        :ivar shift: :class:`Weights` The shift value.
        :ivar scale: :class:`Weights` The scale value.
        :ivar power: :class:`Weights` The power value.
        :ivar channel_axis: :class:`int` The channel axis.
    """
    mode: ScaleMode
    @property
    def channel_axis(self) -> int:
        ...
    @channel_axis.setter
    def channel_axis(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def power(self) -> typing.Any:
        ...
    @power.setter
    def power(self, arg1: Weights) -> None:
        ...
    @property
    def scale(self) -> typing.Any:
        ...
    @scale.setter
    def scale(self, arg1: Weights) -> None:
        ...
    @property
    def shift(self) -> typing.Any:
        ...
    @shift.setter
    def shift(self, arg1: Weights) -> None:
        ...
class IScatterLayer(ILayer):
    """

        A Scatter layer as in :class:`INetworkDefinition`.
        :ivar axis: axis to scatter on when using Scatter Element mode (ignored in ND mode)
        :ivar mode: :class:`ScatterMode` The operation mode of the scatter.
    """
    mode: ScatterMode
    @property
    def axis(self) -> int:
        ...
    @axis.setter
    def axis(self, arg1: typing.SupportsInt) -> None:
        ...
class ISelectLayer(ILayer):
    """

        A select layer in an :class:`INetworkDefinition` .

        This layer implements an element-wise ternary conditional operation. Wherever ``condition`` is ``True``, elements are taken from the first input, and wherever ``condition`` is ``False``, elements are taken from the second input.
    """
class ISerializationConfig:
    """
    Class to hold properties for configuring an engine to serialize the binary.
    """
    def clear_flag(self, flag: SerializationFlag) -> bool:
        """
        Clears the serialization flag from the config.
        """
    def get_flag(self, flag: SerializationFlag) -> bool:
        """
        Check if a serialization flag is set.
        """
    def set_flag(self, flag: SerializationFlag) -> bool:
        """
        Add the input serialization flag to the already enabled flags.
        """
    @property
    def flags(self) -> int:
        ...
    @flags.setter
    def flags(self, arg1: typing.SupportsInt) -> None:
        ...
class IShapeLayer(ILayer):
    """

        A shape layer in an :class:`INetworkDefinition` . Used for getting the shape of a tensor.
        This class sets the output to a one-dimensional tensor with the dimensions of the input tensor.

        For example, if the input is a four-dimensional tensor (of any type) with
        dimensions [2,3,5,7], the output tensor is a one-dimensional :class:`int64` tensor
        of length 4 containing the sequence 2, 3, 5, 7.
    """
class IShuffleLayer(ILayer):
    """

        A shuffle layer in an :class:`INetworkDefinition` .

        This class shuffles data by applying in sequence: a transpose operation, a reshape operation and a second transpose operation. The dimension types of the output are those of the reshape dimension.

        :ivar first_transpose: :class:`Permutation` The permutation applied by the first transpose operation. Default: Identity Permutation
        :ivar reshape_dims: :class:`Dims` The reshaped dimensions.
            Two special values can be used as dimensions.
            Value 0 copies the corresponding dimension from input. This special value can be used more than once in the dimensions. If number of reshape dimensions is less than input, 0s are resolved by aligning the most significant dimensions of input.
            Value -1 infers that particular dimension by looking at input and rest of the reshape dimensions. Note that only a maximum of one dimension is permitted to be specified as -1.
            The product of the new dimensions must be equal to the product of the old.
        :ivar second_transpose: :class:`Permutation` The permutation applied by the second transpose operation. Default: Identity Permutation
        :ivar zero_is_placeholder: :class:`bool` The meaning of 0 in reshape dimensions.
            If true, then a 0 in the reshape dimensions denotes copying the corresponding
            dimension from the first input tensor.  If false, then a 0 in the reshape
            dimensions denotes a zero-length dimension.
    """
    first_transpose: Permutation
    reshape_dims: Dims
    second_transpose: Permutation
    zero_is_placeholder: bool
    def set_input(self: ILayer, index: typing.SupportsInt, tensor: ITensor) -> None:
        """
            Sets the input tensor for the given index. The index must be 0 for a static shuffle layer.
            A static shuffle layer is converted to a dynamic shuffle layer by calling :func:`set_input` with an index 1.
            A dynamic shuffle layer cannot be converted back to a static shuffle layer.

            For a dynamic shuffle layer, the values 0 and 1 are valid.
            The indices in the dynamic case are as follows:

            ======= ========================================================================
             Index   Description
            ======= ========================================================================
                0     Data or Shape tensor to be shuffled.
                1     The dimensions for the reshape operation, as a 1D :class:`int32` shape tensor.
            ======= ========================================================================

            If this function is called with a value 1, then :attr:`num_inputs` changes
            from 1 to 2.

            :arg index: The index of the input tensor.
            :arg tensor: The input tensor.
        """
class ISliceLayer(ILayer):
    """

        A slice layer in an :class:`INetworkDefinition` .

        The slice layer has two variants, static and dynamic.
        Static slice specifies the start, size, and stride dimensions at layer creation time via :class:`Dims` and can use the get/set accessor functions of the :class:`ISliceLayer` .
        Dynamic slice specifies one or more of start, size, stride, or axes as :class:`ITensor`s, by using :func:`ILayer.set_input` to add a second, third, fourth, or sixth input respectively.
        The corresponding :class:`Dims` are used if an input is missing or null.

        An application can determine if the :class:`ISliceLayer` has a dynamic output shape based on whether the size or axes input is present and non-null.

        The slice layer selects for each dimension a start location from within the input tensor, and copies elements to the output tensor using the specified stride across the input tensor.
        Start, size, and stride tensors must be 1-D integer-typed shape tensors if not specified via :class:`Dims` .

        An example of using slice on a tensor:
        input = {{0, 2, 4}, {1, 3, 5}}
        start = {1, 0}
        size = {1, 2}
        stride = {1, 2}
        output = {{1, 5}}

        If axes is provided then starts, ends, and strides must have the same length as axes and specifies a subset of dimensions to slice. If axes is not provided, starts, ends, and strides
        must be of the same length as the rank of the input tensor.

        An example of using slice on a tensor with axes specified:
        input = {{0, 2, 4}, {1, 3, 5}}
        start = {1}
        size = {2}
        stride = {1}
        axes = {1}
        output = {{2, 4}, {3, 5}}

        When the sampleMode is :const:`SampleMode.CLAMP` or :const:`SampleMode.REFLECT` , for each input dimension, if its size is 0 then the corresponding output dimension must be 0 too.

        When the sampleMode is :const:`SampleMode.FILL`, the fifth input to the slice layer is used to determine the value to fill in out-of-bound indices. It is an error to specify the fifth input in any other sample mode.

        A slice layer can produce a shape tensor if the following conditions are met:

        * ``start``, ``size``, and ``stride`` are build time constants, either as static :class:`Dims` or as constant input tensors.
        * ``axes``, if provided, is a build time constant, either as static :class:`Dims` or as a constant input tensor.
        * The number of elements in the output tensor does not exceed 2 * :const:`Dims.MAX_DIMS` .

        The input tensor is a shape tensor if the output is a shape tensor.

        The following constraints must be satisfied to execute this layer on DLA:
        * ``start``, ``size``, and ``stride`` are build time constants, either as static :class:`Dims` or as constant input tensors.
        * ``axes``, if provided, is a build time constant, either as static :class:`Dims` or as a constant input tensor.
        * sampleMode is :const:`SampleMode.DEFAULT` , :const:`SampleMode.WRAP` , or :const:`SampleMode.FILL` .
        * Strides are 1 for all dimensions.
        * Slicing is not performed on the first dimension.
        * The input tensor has four dimensions.
        * For :const:`SliceMode.FILL` , the fill value input is a scalar output of an :class:`IConstantLayer` with value 0 that is not consumed by any other layer.

        :ivar start: :class:`Dims` The start offset.
        :ivar shape: :class:`Dims` The output dimensions.
        :ivar stride: :class:`Dims` The slicing stride.
        :ivar mode: :class:`SampleMode` Controls how :class:`ISliceLayer` handles out of bounds coordinates.
        :ivar axes: :class:`Dims` The axes that starts, sizes, and strides correspond to.
    """
    axes: Dims
    mode: ...
    shape: Dims
    start: Dims
    stride: Dims
    def set_input(self: ILayer, index: typing.SupportsInt, tensor: ITensor) -> None:
        """
            Sets the input tensor for the given index. The index must be 0 or 4 for a static slice layer.
            A static slice layer is converted to a dynamic slice layer by calling :func:`set_input` with an index between 1 and 3.
            A dynamic slice layer cannot be converted back to a static slice layer.

            The indices are as follows:

            =====   ==================================================================================
            Index   Description
            =====   ==================================================================================
                0     Data or Shape tensor to be sliced.
                1     The start tensor to begin slicing, N-dimensional for Data, and 1-D for Shape.
                2     The size tensor of the resulting slice, N-dimensional for Data, and 1-D for Shape.
                3     The stride of the slicing operation, N-dimensional for Data, and 1-D for Shape.
                4     Value for the :const:`SampleMode.FILL` slice mode. Disallowed for other modes.
                5     The axes tensor indicating the axes that starts, sizes, and strides correspond to. Must be a 1-D tensor.
            =====   ==================================================================================

            If this function is called with a value greater than 0, then :attr:`num_inputs` changes
            from 1 to index + 1.

            :arg index: The index of the input tensor.
            :arg tensor: The input tensor.
        """
class ISoftMaxLayer(ILayer):
    """

        A Softmax layer in an :class:`INetworkDefinition` .

        This layer applies a per-channel softmax to its input.

        The output size is the same as the input size.

        :ivar axes: :class:`int` The axis along which softmax is computed. Currently, only one axis can be set.

        The axis is specified by setting the bit corresponding to the axis to 1, as a bit mask.

        For example, consider an NCHW tensor as input (three non-batch dimensions).

        By default, softmax is performed on the axis which is the number of axes minus three. It is 0 if there are fewer than 3 non-batch axes. For example, if the input is NCHW, the default axis is C. If the input is NHW, then the default axis is H.

        |  Bit 0 corresponds to the N dimension boolean.
        |  Bit 1 corresponds to the C dimension boolean.
        |  Bit 2 corresponds to the H dimension boolean.
        |  Bit 3 corresponds to the W dimension boolean.
        |  By default, softmax is performed on the axis which is the number of axes minus three. It is 0 if
        |  there are fewer than 3 axes. For example, if the input is NCHW, the default axis is C. If the input
        |  is NHW, then the default axis is N.
        |
        |  For example, to perform softmax on axis R of a NPQRCHW input, set bit 3.

        The following constraints must be satisfied to execute this layer on DLA:

        - Axis must be one of the channel or spatial dimensions.
        - There are two classes of supported input sizes:

           * Non-axis, non-batch dimensions are all 1 and the axis dimension is at most 8192. This is the recommended case for using softmax since it is the most accurate.
           * At least one non-axis, non-batch dimension greater than 1 and the axis dimension is at most 1024. Note that in this case, there may be some approximation error as the axis dimension size approaches the upper bound. See the TensorRT Developer Guide for more details on the approximation error.
    """
    @property
    def axes(self) -> int:
        ...
    @axes.setter
    def axes(self, arg1: typing.SupportsInt) -> None:
        ...
class ISqueezeLayer(ILayer):
    """

        A Squeeze layer in an :class:`INetworkDefinition` .

        This layer represents a squeeze operation, removing unit dimensions of the input tensor on a set of axes.

        Axes must be resolvable to a constant Int32 or Int64 1D shape tensor.
        Values in axes must be unique and in the range of [-r, r-1], where r is the rank of the input tensor.
        For each axis value, the corresponding dimension in the input tensor must be one.

    """
    def set_input(self: ILayer, index: typing.SupportsInt, tensor: ITensor) -> None:
        """
            Sets the input tensor for the given index. The index must be 0 or 1 for a Squeeze layer.

            The indices are as follows:

            =====   ==================================================================================
            Index   Description
            =====   ==================================================================================
                0     Input data tensor.
                1     The axes to remove. Must be resolvable to a constant Int32 or Int64 1D shape tensor.
            =====   ==================================================================================

            :arg index: The index of the input tensor.
            :arg tensor: The input tensor.
        """
class IStreamReader:
    """

    Application-implemented class for reading data from a stream.

    To implement a custom stream reader, ensure that you explicitly instantiate the base class in :func:`__init__` :
    ::

        class MyStreamReader(trt.IStreamReader):
            def __init__(self):
                trt.IStreamReader.__init__(self)

            def read(self, size: int) -> bytes:
                ... # Your implementation here

    """
    def __init__(self) -> None:
        ...
    def read(self, destination: typing_extensions.CapsuleType, size: typing.SupportsInt) -> int:
        """
            A callback implemented by the application to read a particular chunk of memory.

            If an allocation request cannot be satisfied, ``0`` should be returned.

            :arg size: The number of bytes required.

            :returns: A buffer containing the bytes read.
        """
class IStreamReaderV2:
    """

        Application-implemented class for asynchronously reading data from a stream. Implementation does not need to be
        asynchronous or use the provided cuda stream. Python users are unlikely to see performance gains over IStreamReader
        or deserialization from a glob.

        To implement a custom stream reader, ensure that you explicitly instantiate the base class in :func:`__init__` :
        ::
            class MyStreamReader(trt.IStreamReaderV2):
                def __init__(self):
                    trt.IStreamReaderV2.__init__(self)

                def read(self, num_bytes: int, stream: int) -> bytes:
                    ... # Your implementation here

                def seek(self, offset: int, where: SeekPosition) -> bool:
                    ... # Your implementation here

        To implement a read operation that retrieves the entire buffer, use the following approach:
        ::
            def read(self, num_bytes: int, stream: int) -> bytes:
                data = read_file_with_size(num_bytes)
                return data

        To read the buffer in chunks to reduce peak host memory usage (e.g., splitting the read into 200MiB chunks):
        ::
            def read(self, num_bytes: int, stream: int) -> bytes:
                chunk_size = 200 * 1024 * 1024  # 200MiB
                num_bytes = min(num_bytes, chunk_size)
                data = read_file_with_size(num_bytes)
                return data
    """
    def __init__(self) -> None:
        ...
    def read(self, destination: typing_extensions.CapsuleType, num_bytes: typing.SupportsInt, stream: typing.SupportsInt) -> int:
        """
            A callback implemented by the application to set the stream location.

            :arg offset: The offset within the stream to seek to.
            :arg where: A `SeekPosition` enum specifying where the offset is relative to.

            :returns: A buffer containing the bytes read.
        """
    def seek(self, offset: typing.SupportsInt, where: SeekPosition) -> bool:
        """
            A callback implemented by the application to read a particular chunk of memory.

            :arg num_bytes: The number of bytes to read. If N bytes are requested but only M bytes (M < N) are read, additional read requests will be made until all N bytes are retrieved.
            :arg stream: A handle to the cudaStream your implementation can use for reading.

            :returns: A buffer containing the bytes read.
        """
class IStreamWriter:
    """

    Application-implemented class for writing data to a stream.

    To implement a custom stream writer, ensure that you explicitly instantiate the base class in :func:`__init__` :
    ::

        class MyStreamWriter(trt.IStreamWriter):
            def __init__(self):
                trt.IStreamWriter.__init__(self)

            def write(self, data: bytes) -> int:
                ... # Your implementation here

    """
    def __init__(self) -> None:
        ...
    def write(self, data: typing_extensions.CapsuleType, size: typing.SupportsInt) -> int:
        """
            A callback implemented by the application to write a particular chunk of memory.

            :arg data: The data to be written out in bytes.

            :returns: The total bytes actually be written.
        """
class ISymExpr:
    def __init__(self) -> None:
        ...
class ISymExprs:
    def __getitem__(self, arg0: typing.SupportsInt) -> ISymExpr:
        ...
    def __len__(self) -> int:
        ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: ISymExpr) -> bool:
        ...
    @property
    def nbSymExprs(self) -> int:
        ...
    @nbSymExprs.setter
    def nbSymExprs(self, arg1: typing.SupportsInt) -> bool:
        ...
class ITensor:
    """

        A tensor in an :class:`INetworkDefinition` .

        :ivar name: :class:`str` The tensor name. For a network input, the name is assigned by the application. For tensors which are layer outputs, a default name is assigned consisting of the layer name followed by the index of the output in brackets. Each network input and output tensor must have a unique name.

        :ivar shape: :class:`Dims` The shape of a tensor. For a network input the shape is assigned by the application. For a network output it is computed based on the layer parameters and the inputs to the layer. If a tensor size or a parameter is modified in the network, the shape of all dependent tensors will be recomputed. This call is only legal for network input tensors, since the shape of layer output tensors are inferred based on layer inputs and parameters.

        :ivar dtype: :class:`DataType` The data type of a tensor. The type is unchanged if the type is invalid for the given tensor.

        :ivar broadcast_across_batch: :class:`bool` [DEPRECATED] Deprecated in TensorRT 10.0. Always false since the implicit batch dimensions support has been removed.

        :ivar location: :class:`TensorLocation` The storage location of a tensor.
        :ivar is_network_input: :class:`bool` Whether the tensor is a network input.
        :ivar is_network_output: :class:`bool` Whether the tensor is a network output.
        :ivar dynamic_range: :class:`Tuple[float, float]` [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization. A tuple containing the [minimum, maximum] of the dynamic range, or :class:`None` if the range was not set.
        :ivar is_shape: :class:`bool` Whether the tensor is a shape tensor.
        :ivar allowed_formats: :class:`int32` The allowed set of TensorFormat candidates. This should be an integer consisting of one or more :class:`TensorFormat` s, combined via bitwise OR after bit shifting. For example, ``1 << int(TensorFormat.CHW4) | 1 << int(TensorFormat.CHW32)``.
    """
    broadcast_across_batch: bool
    dtype: DataType
    location: ...
    name: str
    shape: Dims
    def get_dimension_name(self, index: typing.SupportsInt) -> str:
        """
            Get the name of an input dimension.

            :arg index: index of the dimension.
            :returns: name of the dimension, or null if dimension is unnamed.
        """
    def reset_dynamic_range(self) -> None:
        """
            [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization.
            Undo the effect of setting the dynamic range.
        """
    def set_dimension_name(self, index: typing.SupportsInt, name: str) -> None:
        """
            Name a dimension of an input tensor.

            Associate a runtime dimension of an input tensor with a symbolic name.
            Dimensions with the same non-empty name must be equal at runtime.
            Knowing this equality for runtime dimensions may help the TensorRT optimizer.
            Both runtime and build-time dimensions can be named.
            If the function is called again, with the same index, it will overwrite the previous name.
            If None is passed as name, it will clear the name of the dimension.

            For example, setDimensionName(0, "n") associates the symbolic name "n" with the leading dimension.

            :arg index: index of the dimension.
            :arg name: name of the dimension.
        """
    def set_dynamic_range(self, min: typing.SupportsFloat, max: typing.SupportsFloat) -> bool:
        """
            [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization.
            Set dynamic range for the tensor.
            NOTE: It is suggested to use ``tensor.dynamic_range = (min, max)`` instead.

            :arg min: Minimum of the dynamic range.
            :arg max: Maximum of the dyanmic range.
            :returns: true if succeed in setting range. Otherwise false.
        """
    @property
    def allowed_formats(self) -> int:
        ...
    @allowed_formats.setter
    def allowed_formats(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def dynamic_range(self) -> typing.Any:
        ...
    @dynamic_range.setter
    def dynamic_range(self, arg1: collections.abc.Sequence[typing.SupportsFloat]) -> None:
        ...
    @property
    def is_execution_tensor(self) -> bool:
        ...
    @property
    def is_network_input(self) -> bool:
        ...
    @property
    def is_network_output(self) -> bool:
        ...
    @property
    def is_shape_tensor(self) -> bool:
        ...
class ITimingCache:
    """

            Class to handle tactic timing info collected from builder.

    """
    def combine(self, input_cache: ITimingCache, ignore_mismatch: bool) -> bool:
        """
                Combine input timing cache into local instance.

                Append entries in input cache to local cache. Conflicting entries will be skipped. The input
                cache must be generated by a TensorRT build of exact same version, otherwise combine will be
                skipped and return false. ``bool(ignore_mismatch) == True`` if combining a timing cache
                created from a different device.

                :arg input_cache: The input timing cache
                :arg ignore_mismatch: Whether or not to allow cache verification header mismatch

                :returns: A `bool` indicating whether the combine operation is done successfully.
        """
    def query(self, key: TimingCacheKey) -> TimingCacheValue:
        """
                Query value in a cache entry.

                If the key exists, write the value out, otherwise return an invalid value.

                :arg key: The query key.
                :cache cache: value if the key exists, otherwise an invalid value.

                :returns A :class:`TimingCacheValue` object.
        """
    def queryKeys(self) -> list[TimingCacheKey]:
        """
            Query cache keys from Timing Cache.

            If an error occurs, a RuntimeError will be raised.

            :returns A list containing the cache keys.
        """
    def reset(self) -> bool:
        """
                Empty the timing cache

                :returns: A `bool` indicating whether the reset operation is done successfully.
        """
    def serialize(self) -> IHostMemory:
        """
                Serialize a timing cache to a :class:`IHostMemory` object.

                :returns: An :class:`IHostMemory` object that contains a serialized timing cache.
        """
    def update(self, key: TimingCacheKey, value: TimingCacheValue) -> bool:
        """
                Update values in a cache entry.

                Update the value of the given cache key. If the key does not exist, return False.
                If the key exists and the new tactic timing is NaN, delete the cache entry and
                return True. If tactic timing is not NaN and the new value is valid, override the
                cache value and return True. False is returned when the new value is invalid.
                If this layer cannot use the new tactic, build errors will be reported when
                building the next engine.

                :arg key: The key to the entry to be updated.
                :arg value: New cache value.

                :returns True if update succeeds, otherwise False.
        """
class ITopKLayer(ILayer):
    """

        A TopK layer in an :class:`INetworkDefinition` .

        :ivar op: :class:`TopKOperation` The operation for the layer.
        :ivar k: :class:`TopKOperation` the k value for the layer. Currently only values up to 3840 are supported.
            Use the set_input() method with index 1 to pass in dynamic k as a tensor.
        :ivar axes: :class:`TopKOperation` The axes along which to reduce.
        :ivar indices_type: :class:`DataType` The specified data type of the output indices tensor. Must be tensorrt.int32 or tensorrt.int64.

    """
    indices_type: DataType
    op: TopKOperation
    def set_input(self: ILayer, index: typing.SupportsInt, tensor: ITensor) -> None:
        """
            Sets the input tensor for the given index. The index must be 0 or 1 for a TopK layer.

            The indices are as follows:

            =====   ==================================================================================
            Index   Description
            =====   ==================================================================================
                0     Input data tensor.
                1     A scalar Int32 tensor containing a positive value corresponding to the number
                        of top elements to retrieve. Values larger than 3840 will result in a runtime
                        error. If provided, this will override the static k value in calculations.
            =====   ==================================================================================

            :arg index: The index of the input tensor.
            :arg tensor: The input tensor.
        """
    @property
    def axes(self) -> int:
        ...
    @axes.setter
    def axes(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def k(self) -> int:
        ...
    @k.setter
    def k(self, arg1: typing.SupportsInt) -> None:
        ...
class ITripLimitLayer(ILoopBoundaryLayer):
    """

        :ivar kind: The kind of trip limit. See :class:`TripLimit`
    """
    @property
    def kind(self) -> TripLimit:
        ...
class IUnaryLayer(ILayer):
    """

        A unary layer in an :class:`INetworkDefinition` .

        :ivar op: :class:`UnaryOperation` The unary operation for the layer. When running this layer on DLA, only ``UnaryOperation.ABS`` is supported.
    """
    op: UnaryOperation
class IUnsqueezeLayer(ILayer):
    """

        An Unsqueeze layer in an :class:`INetworkDefinition` .

        This layer represents an unsqueeze operation, which reshapes the input tensor by inserting unit-length dimensions at specified axes of the output.

        Axes must be resolvable to a constant Int32 or Int64 shape tensor.
        Values in axes must be unique and in the range of [-r_final, r_final-1], where r_final is the sum of rank(input) and len(axes).

        r_final must be less than Dims.MAX_DIMS.

    """
    def set_input(self: ILayer, index: typing.SupportsInt, tensor: ITensor) -> None:
        """
            Sets the input tensor for the given index. The index must be 0 or 1 for an Unsqueeze layer.

            The indices are as follows:

            =====   ==================================================================================
            Index   Description
            =====   ==================================================================================
                0     Input data tensor.
                1     The axes to add. Must be resolvable to a constant Int32 or Int64 1D shape tensor.
            =====   ==================================================================================

            :arg index: The index of the input tensor.
            :arg tensor: The input tensor.
        """
class IVersionedInterface:
    """

        Base class for all versioned interfaces.
    """
    @property
    def api_language(self) -> APILanguage:
        ...
    @property
    def interface_info(self) -> InterfaceInfo:
        ...
class InterfaceInfo:
    """

        Version information for a TensorRT interface.
    """
    kind: str
    @property
    def major(self) -> int:
        ...
    @major.setter
    def major(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def minor(self) -> int:
        ...
    @minor.setter
    def minor(self, arg0: typing.SupportsInt) -> None:
        ...
class InterpolationMode:
    """
    Various modes of interpolation, used in resize and grid_sample layers.

    Members:

      NEAREST : 1D, 2D, and 3D nearest neighbor interpolation.

      LINEAR : Supports linear, bilinear, trilinear interpolation.

      CUBIC : Supports bicubic interpolation.
    """
    CUBIC: typing.ClassVar[InterpolationMode]  # value = <InterpolationMode.CUBIC: 2>
    LINEAR: typing.ClassVar[InterpolationMode]  # value = <InterpolationMode.LINEAR: 1>
    NEAREST: typing.ClassVar[InterpolationMode]  # value = <InterpolationMode.NEAREST: 0>
    __members__: typing.ClassVar[dict[str, InterpolationMode]]  # value = {'NEAREST': <InterpolationMode.NEAREST: 0>, 'LINEAR': <InterpolationMode.LINEAR: 1>, 'CUBIC': <InterpolationMode.CUBIC: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class KVCacheMode:
    """
    The cache modes supported by a KVCacheUpdate layer

    Members:

      LINEAR :
    """
    LINEAR: typing.ClassVar[KVCacheMode]  # value = <KVCacheMode.LINEAR: 0>
    __members__: typing.ClassVar[dict[str, KVCacheMode]]  # value = {'LINEAR': <KVCacheMode.LINEAR: 0>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class KernelLaunchParams:
    block_x: ...
    block_y: ...
    block_z: ...
    grid_x: ...
    grid_y: ...
    grid_z: ...
    shared_mem: ...
class LayerInformationFormat:
    """
    The format in which the IEngineInspector prints the layer information.

    Members:

      ONELINE : Print layer information in one line per layer.

      JSON : Print layer information in JSON format.
    """
    JSON: typing.ClassVar[LayerInformationFormat]  # value = <LayerInformationFormat.JSON: 1>
    ONELINE: typing.ClassVar[LayerInformationFormat]  # value = <LayerInformationFormat.ONELINE: 0>
    __members__: typing.ClassVar[dict[str, LayerInformationFormat]]  # value = {'ONELINE': <LayerInformationFormat.ONELINE: 0>, 'JSON': <LayerInformationFormat.JSON: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class LayerType:
    """
    Type of Layer

    Members:

      CONVOLUTION : Convolution layer

      GRID_SAMPLE : Grid sample layer

      NMS : NMS layer

      ACTIVATION : Activation layer

      POOLING : Pooling layer

      LRN : LRN layer

      SCALE : Scale layer

      SOFTMAX : Softmax layer

      DECONVOLUTION : Deconvolution layer

      CONCATENATION : Concatenation layer

      ELEMENTWISE : Elementwise layer

      PLUGIN : Plugin layer

      UNARY : Unary layer

      PADDING : Padding layer

      SHUFFLE : Shuffle layer

      REDUCE : Reduce layer

      TOPK : TopK layer

      GATHER : Gather layer

      MATRIX_MULTIPLY : Matrix multiply layer

      RAGGED_SOFTMAX : Ragged softmax layer

      CONSTANT : Constant layer

      IDENTITY : Identity layer

      CAST : Cast layer

      PLUGIN_V2 : PluginV2 layer

      SLICE : Slice layer

      SHAPE : Shape layer

      PARAMETRIC_RELU : Parametric ReLU layer

      RESIZE : Resize layer

      TRIP_LIMIT : Loop Trip limit layer

      RECURRENCE : Loop Recurrence layer

      ITERATOR : Loop Iterator layer

      LOOP_OUTPUT : Loop output layer

      SELECT : Select layer

      ASSERTION : Assertion layer

      FILL : Fill layer

      QUANTIZE : Quantize layer

      DEQUANTIZE : Dequantize layer

      CONDITION : If-conditional Condition layer

      CONDITIONAL_INPUT : If-conditional input layer

      CONDITIONAL_OUTPUT : If-conditional output layer

      SCATTER : Scatter layer

      EINSUM : Einsum layer

      ONE_HOT : OneHot layer

      NON_ZERO : NonZero layer

      REVERSE_SEQUENCE : ReverseSequence layer

      NORMALIZATION : Normalization layer

      PLUGIN_V3 : PluginV3 layer

      SQUEEZE : Squeeze layer

      UNSQUEEZE : Unsqueeze layer

      CUMULATIVE : Cumulative layer

      DYNAMIC_QUANTIZE : DynamicQuantize layer

      ATTENTION_INPUT : Attention input layer

      ATTENTION_OUTPUT : Attention output layer

      ROTARY_EMBEDDING : Rotary Embedding layer

      KV_CACHE_UPDATE : KVCacheUpdate layer

      MOE : MoE layer

      DIST_COLLECTIVE : DistCollective layer
    """
    ACTIVATION: typing.ClassVar[LayerType]  # value = <LayerType.ACTIVATION: 2>
    ASSERTION: typing.ClassVar[LayerType]  # value = <LayerType.ASSERTION: 39>
    ATTENTION_INPUT: typing.ClassVar[LayerType]  # value = <LayerType.ATTENTION_INPUT: 51>
    ATTENTION_OUTPUT: typing.ClassVar[LayerType]  # value = <LayerType.ATTENTION_OUTPUT: 52>
    CAST: typing.ClassVar[LayerType]  # value = <LayerType.CAST: 1>
    CONCATENATION: typing.ClassVar[LayerType]  # value = <LayerType.CONCATENATION: 8>
    CONDITION: typing.ClassVar[LayerType]  # value = <LayerType.CONDITION: 34>
    CONDITIONAL_INPUT: typing.ClassVar[LayerType]  # value = <LayerType.CONDITIONAL_INPUT: 35>
    CONDITIONAL_OUTPUT: typing.ClassVar[LayerType]  # value = <LayerType.CONDITIONAL_OUTPUT: 36>
    CONSTANT: typing.ClassVar[LayerType]  # value = <LayerType.CONSTANT: 19>
    CONVOLUTION: typing.ClassVar[LayerType]  # value = <LayerType.CONVOLUTION: 0>
    CUMULATIVE: typing.ClassVar[LayerType]  # value = <LayerType.CUMULATIVE: 49>
    DECONVOLUTION: typing.ClassVar[LayerType]  # value = <LayerType.DECONVOLUTION: 7>
    DEQUANTIZE: typing.ClassVar[LayerType]  # value = <LayerType.DEQUANTIZE: 33>
    DIST_COLLECTIVE: typing.ClassVar[LayerType]  # value = <LayerType.DIST_COLLECTIVE: 56>
    DYNAMIC_QUANTIZE: typing.ClassVar[LayerType]  # value = <LayerType.DYNAMIC_QUANTIZE: 50>
    EINSUM: typing.ClassVar[LayerType]  # value = <LayerType.EINSUM: 38>
    ELEMENTWISE: typing.ClassVar[LayerType]  # value = <LayerType.ELEMENTWISE: 9>
    FILL: typing.ClassVar[LayerType]  # value = <LayerType.FILL: 31>
    GATHER: typing.ClassVar[LayerType]  # value = <LayerType.GATHER: 16>
    GRID_SAMPLE: typing.ClassVar[LayerType]  # value = <LayerType.GRID_SAMPLE: 42>
    IDENTITY: typing.ClassVar[LayerType]  # value = <LayerType.IDENTITY: 20>
    ITERATOR: typing.ClassVar[LayerType]  # value = <LayerType.ITERATOR: 28>
    KV_CACHE_UPDATE: typing.ClassVar[LayerType]  # value = <LayerType.KV_CACHE_UPDATE: 54>
    LOOP_OUTPUT: typing.ClassVar[LayerType]  # value = <LayerType.LOOP_OUTPUT: 29>
    LRN: typing.ClassVar[LayerType]  # value = <LayerType.LRN: 4>
    MATRIX_MULTIPLY: typing.ClassVar[LayerType]  # value = <LayerType.MATRIX_MULTIPLY: 17>
    MOE: typing.ClassVar[LayerType]  # value = <LayerType.MOE: 55>
    NMS: typing.ClassVar[LayerType]  # value = <LayerType.NMS: 43>
    NON_ZERO: typing.ClassVar[LayerType]  # value = <LayerType.NON_ZERO: 41>
    NORMALIZATION: typing.ClassVar[LayerType]  # value = <LayerType.NORMALIZATION: 45>
    ONE_HOT: typing.ClassVar[LayerType]  # value = <LayerType.ONE_HOT: 40>
    PADDING: typing.ClassVar[LayerType]  # value = <LayerType.PADDING: 12>
    PARAMETRIC_RELU: typing.ClassVar[LayerType]  # value = <LayerType.PARAMETRIC_RELU: 24>
    PLUGIN: typing.ClassVar[LayerType]  # value = <LayerType.PLUGIN: 10>
    PLUGIN_V2: typing.ClassVar[LayerType]  # value = <LayerType.PLUGIN_V2: 21>
    PLUGIN_V3: typing.ClassVar[LayerType]  # value = <LayerType.PLUGIN_V3: 46>
    POOLING: typing.ClassVar[LayerType]  # value = <LayerType.POOLING: 3>
    QUANTIZE: typing.ClassVar[LayerType]  # value = <LayerType.QUANTIZE: 32>
    RAGGED_SOFTMAX: typing.ClassVar[LayerType]  # value = <LayerType.RAGGED_SOFTMAX: 18>
    RECURRENCE: typing.ClassVar[LayerType]  # value = <LayerType.RECURRENCE: 27>
    REDUCE: typing.ClassVar[LayerType]  # value = <LayerType.REDUCE: 14>
    RESIZE: typing.ClassVar[LayerType]  # value = <LayerType.RESIZE: 25>
    REVERSE_SEQUENCE: typing.ClassVar[LayerType]  # value = <LayerType.REVERSE_SEQUENCE: 44>
    ROTARY_EMBEDDING: typing.ClassVar[LayerType]  # value = <LayerType.ROTARY_EMBEDDING: 53>
    SCALE: typing.ClassVar[LayerType]  # value = <LayerType.SCALE: 5>
    SCATTER: typing.ClassVar[LayerType]  # value = <LayerType.SCATTER: 37>
    SELECT: typing.ClassVar[LayerType]  # value = <LayerType.SELECT: 30>
    SHAPE: typing.ClassVar[LayerType]  # value = <LayerType.SHAPE: 23>
    SHUFFLE: typing.ClassVar[LayerType]  # value = <LayerType.SHUFFLE: 13>
    SLICE: typing.ClassVar[LayerType]  # value = <LayerType.SLICE: 22>
    SOFTMAX: typing.ClassVar[LayerType]  # value = <LayerType.SOFTMAX: 6>
    SQUEEZE: typing.ClassVar[LayerType]  # value = <LayerType.SQUEEZE: 47>
    TOPK: typing.ClassVar[LayerType]  # value = <LayerType.TOPK: 15>
    TRIP_LIMIT: typing.ClassVar[LayerType]  # value = <LayerType.TRIP_LIMIT: 26>
    UNARY: typing.ClassVar[LayerType]  # value = <LayerType.UNARY: 11>
    UNSQUEEZE: typing.ClassVar[LayerType]  # value = <LayerType.UNSQUEEZE: 48>
    __members__: typing.ClassVar[dict[str, LayerType]]  # value = {'CONVOLUTION': <LayerType.CONVOLUTION: 0>, 'GRID_SAMPLE': <LayerType.GRID_SAMPLE: 42>, 'NMS': <LayerType.NMS: 43>, 'ACTIVATION': <LayerType.ACTIVATION: 2>, 'POOLING': <LayerType.POOLING: 3>, 'LRN': <LayerType.LRN: 4>, 'SCALE': <LayerType.SCALE: 5>, 'SOFTMAX': <LayerType.SOFTMAX: 6>, 'DECONVOLUTION': <LayerType.DECONVOLUTION: 7>, 'CONCATENATION': <LayerType.CONCATENATION: 8>, 'ELEMENTWISE': <LayerType.ELEMENTWISE: 9>, 'PLUGIN': <LayerType.PLUGIN: 10>, 'UNARY': <LayerType.UNARY: 11>, 'PADDING': <LayerType.PADDING: 12>, 'SHUFFLE': <LayerType.SHUFFLE: 13>, 'REDUCE': <LayerType.REDUCE: 14>, 'TOPK': <LayerType.TOPK: 15>, 'GATHER': <LayerType.GATHER: 16>, 'MATRIX_MULTIPLY': <LayerType.MATRIX_MULTIPLY: 17>, 'RAGGED_SOFTMAX': <LayerType.RAGGED_SOFTMAX: 18>, 'CONSTANT': <LayerType.CONSTANT: 19>, 'IDENTITY': <LayerType.IDENTITY: 20>, 'CAST': <LayerType.CAST: 1>, 'PLUGIN_V2': <LayerType.PLUGIN_V2: 21>, 'SLICE': <LayerType.SLICE: 22>, 'SHAPE': <LayerType.SHAPE: 23>, 'PARAMETRIC_RELU': <LayerType.PARAMETRIC_RELU: 24>, 'RESIZE': <LayerType.RESIZE: 25>, 'TRIP_LIMIT': <LayerType.TRIP_LIMIT: 26>, 'RECURRENCE': <LayerType.RECURRENCE: 27>, 'ITERATOR': <LayerType.ITERATOR: 28>, 'LOOP_OUTPUT': <LayerType.LOOP_OUTPUT: 29>, 'SELECT': <LayerType.SELECT: 30>, 'ASSERTION': <LayerType.ASSERTION: 39>, 'FILL': <LayerType.FILL: 31>, 'QUANTIZE': <LayerType.QUANTIZE: 32>, 'DEQUANTIZE': <LayerType.DEQUANTIZE: 33>, 'CONDITION': <LayerType.CONDITION: 34>, 'CONDITIONAL_INPUT': <LayerType.CONDITIONAL_INPUT: 35>, 'CONDITIONAL_OUTPUT': <LayerType.CONDITIONAL_OUTPUT: 36>, 'SCATTER': <LayerType.SCATTER: 37>, 'EINSUM': <LayerType.EINSUM: 38>, 'ONE_HOT': <LayerType.ONE_HOT: 40>, 'NON_ZERO': <LayerType.NON_ZERO: 41>, 'REVERSE_SEQUENCE': <LayerType.REVERSE_SEQUENCE: 44>, 'NORMALIZATION': <LayerType.NORMALIZATION: 45>, 'PLUGIN_V3': <LayerType.PLUGIN_V3: 46>, 'SQUEEZE': <LayerType.SQUEEZE: 47>, 'UNSQUEEZE': <LayerType.UNSQUEEZE: 48>, 'CUMULATIVE': <LayerType.CUMULATIVE: 49>, 'DYNAMIC_QUANTIZE': <LayerType.DYNAMIC_QUANTIZE: 50>, 'ATTENTION_INPUT': <LayerType.ATTENTION_INPUT: 51>, 'ATTENTION_OUTPUT': <LayerType.ATTENTION_OUTPUT: 52>, 'ROTARY_EMBEDDING': <LayerType.ROTARY_EMBEDDING: 53>, 'KV_CACHE_UPDATE': <LayerType.KV_CACHE_UPDATE: 54>, 'MOE': <LayerType.MOE: 55>, 'DIST_COLLECTIVE': <LayerType.DIST_COLLECTIVE: 56>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Logger(ILogger):
    """

    Logger for the :class:`Builder`, :class:`ICudaEngine` and :class:`Runtime` .

    :arg min_severity: The initial minimum severity of this Logger.

    :ivar min_severity: :class:`Logger.Severity` This minimum required severity of messages for the logger to log them.

    """
    class Severity:
        """

            Indicates the severity of a message. The values in this enum are also accessible in the :class:`ILogger` directly.
            For example, ``tensorrt.ILogger.INFO`` corresponds to ``tensorrt.ILogger.Severity.INFO`` .


        Members:

          INTERNAL_ERROR :
            Represents an internal error. Execution is unrecoverable.


          ERROR :
            Represents an application error.


          WARNING :
            Represents an application error that TensorRT has recovered from or fallen back to a default.


          INFO :
            Represents informational messages.


          VERBOSE :
            Verbose messages with debugging information.
        """
        ERROR: typing.ClassVar[ILogger.Severity]  # value = <Severity.ERROR: 1>
        INFO: typing.ClassVar[ILogger.Severity]  # value = <Severity.INFO: 3>
        INTERNAL_ERROR: typing.ClassVar[ILogger.Severity]  # value = <Severity.INTERNAL_ERROR: 0>
        VERBOSE: typing.ClassVar[ILogger.Severity]  # value = <Severity.VERBOSE: 4>
        WARNING: typing.ClassVar[ILogger.Severity]  # value = <Severity.WARNING: 2>
        __members__: typing.ClassVar[dict[str, ILogger.Severity]]  # value = {'INTERNAL_ERROR': <Severity.INTERNAL_ERROR: 0>, 'ERROR': <Severity.ERROR: 1>, 'WARNING': <Severity.WARNING: 2>, 'INFO': <Severity.INFO: 3>, 'VERBOSE': <Severity.VERBOSE: 4>}
        def __eq__(self, other: typing.Any) -> bool:
            ...
        def __ge__(self, other: typing.Any) -> bool:
            ...
        def __getstate__(self) -> int:
            ...
        def __gt__(self, other: typing.Any) -> bool:
            ...
        def __hash__(self) -> int:
            ...
        def __index__(self: ILogger.Severity) -> int:
            ...
        def __init__(self: ILogger.Severity, value: typing.SupportsInt) -> None:
            ...
        def __int__(self: ILogger.Severity) -> int:
            ...
        def __le__(self, other: typing.Any) -> bool:
            ...
        def __lt__(self, other: typing.Any) -> bool:
            ...
        def __ne__(self, other: typing.Any) -> bool:
            ...
        def __repr__(self) -> str:
            ...
        def __setstate__(self: ILogger.Severity, state: typing.SupportsInt) -> None:
            ...
        def __str__(self) -> str:
            ...
        @property
        def name(self) -> str:
            ...
        @property
        def value(self) -> int:
            ...
    ERROR: typing.ClassVar[ILogger.Severity]  # value = <Severity.ERROR: 1>
    INFO: typing.ClassVar[ILogger.Severity]  # value = <Severity.INFO: 3>
    INTERNAL_ERROR: typing.ClassVar[ILogger.Severity]  # value = <Severity.INTERNAL_ERROR: 0>
    VERBOSE: typing.ClassVar[ILogger.Severity]  # value = <Severity.VERBOSE: 4>
    WARNING: typing.ClassVar[ILogger.Severity]  # value = <Severity.WARNING: 2>
    min_severity: ILogger.Severity
    def __init__(self, min_severity: ILogger.Severity = ...) -> None:
        ...
    def log(self, severity: ILogger.Severity, msg: str) -> None:
        """
        Logs a message to `stdout`.

        :arg severity: The severity of the message.
        :arg msg: The log message.
        """
class LoopOutput:
    """
    Describes kinds of loop outputs.

    Members:

      LAST_VALUE : Output value is value of tensor for last iteration.

      CONCATENATE : Output value is concatenation of values of tensor for each iteration, in forward order.

      REVERSE : Output value is concatenation of values of tensor for each iteration, in reverse order.
    """
    CONCATENATE: typing.ClassVar[LoopOutput]  # value = <LoopOutput.CONCATENATE: 1>
    LAST_VALUE: typing.ClassVar[LoopOutput]  # value = <LoopOutput.LAST_VALUE: 0>
    REVERSE: typing.ClassVar[LoopOutput]  # value = <LoopOutput.REVERSE: 2>
    __members__: typing.ClassVar[dict[str, LoopOutput]]  # value = {'LAST_VALUE': <LoopOutput.LAST_VALUE: 0>, 'CONCATENATE': <LoopOutput.CONCATENATE: 1>, 'REVERSE': <LoopOutput.REVERSE: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class MatrixOperation:
    """
    The matrix operations that may be performed by a Matrix layer

    Members:

      NONE :

      TRANSPOSE : Transpose each matrix

      VECTOR : Treat operand as collection of vectors
    """
    NONE: typing.ClassVar[MatrixOperation]  # value = <MatrixOperation.NONE: 0>
    TRANSPOSE: typing.ClassVar[MatrixOperation]  # value = <MatrixOperation.TRANSPOSE: 1>
    VECTOR: typing.ClassVar[MatrixOperation]  # value = <MatrixOperation.VECTOR: 2>
    __members__: typing.ClassVar[dict[str, MatrixOperation]]  # value = {'NONE': <MatrixOperation.NONE: 0>, 'TRANSPOSE': <MatrixOperation.TRANSPOSE: 1>, 'VECTOR': <MatrixOperation.VECTOR: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class MemoryPoolType:
    """
    The type for memory pools used by TensorRT.

    Members:

      WORKSPACE :
        WORKSPACE is used by TensorRT to store intermediate buffers within an operation.
        This defaults to max device memory. Set to a smaller value to restrict tactics that use over the threshold en masse.
        For more targeted removal of tactics use the IAlgorithmSelector interface ([DEPRECATED] Deprecated in TensorRT 10.8. Please use editable mode in ITimingCache instead).


      DLA_MANAGED_SRAM :
        DLA_MANAGED_SRAM is a fast software managed RAM used by DLA to communicate within a layer.
        The size of this pool must be at least 4 KiB and must be a power of 2.
        This defaults to 1 MiB.
        Orin has capacity of 1 MiB per core.


      DLA_LOCAL_DRAM :
        DLA_LOCAL_DRAM is host RAM used by DLA to share intermediate tensor data across operations.
        The size of this pool must be at least 4 KiB and must be a power of 2.
        This defaults to 1 GiB.


      DLA_GLOBAL_DRAM :
        DLA_GLOBAL_DRAM is host RAM used by DLA to store weights and metadata for execution.
        The size of this pool must be at least 4 KiB and must be a power of 2.
        This defaults to 512 MiB.


      TACTIC_DRAM :
        TACTIC_DRAM is the host DRAM used by the optimizer to
        run tactics. On embedded devices, where host and device memory are unified, this includes all device
        memory required by TensorRT to build the network up to the point of each memory allocation.
        This defaults to 75% of totalGlobalMem as reported by cudaGetDeviceProperties when
        cudaGetDeviceProperties.embedded is true, and 100% otherwise.


      TACTIC_SHARED_MEMORY :
        TACTIC_SHARED_MEMORY defines the maximum shared memory size utilized for driver reserved and executing
        the backend CUDA kernel implementation. Adjust this value to restrict tactics that exceed
        the specified threshold en masse. The default value is device max capability. This value must
        be less than 1GiB.

        Updating this flag will override the shared memory limit set by \\ref HardwareCompatibilityLevel,
        which defaults to 48KiB.
    """
    DLA_GLOBAL_DRAM: typing.ClassVar[MemoryPoolType]  # value = <MemoryPoolType.DLA_GLOBAL_DRAM: 3>
    DLA_LOCAL_DRAM: typing.ClassVar[MemoryPoolType]  # value = <MemoryPoolType.DLA_LOCAL_DRAM: 2>
    DLA_MANAGED_SRAM: typing.ClassVar[MemoryPoolType]  # value = <MemoryPoolType.DLA_MANAGED_SRAM: 1>
    TACTIC_DRAM: typing.ClassVar[MemoryPoolType]  # value = <MemoryPoolType.TACTIC_DRAM: 4>
    TACTIC_SHARED_MEMORY: typing.ClassVar[MemoryPoolType]  # value = <MemoryPoolType.TACTIC_SHARED_MEMORY: 5>
    WORKSPACE: typing.ClassVar[MemoryPoolType]  # value = <MemoryPoolType.WORKSPACE: 0>
    __members__: typing.ClassVar[dict[str, MemoryPoolType]]  # value = {'WORKSPACE': <MemoryPoolType.WORKSPACE: 0>, 'DLA_MANAGED_SRAM': <MemoryPoolType.DLA_MANAGED_SRAM: 1>, 'DLA_LOCAL_DRAM': <MemoryPoolType.DLA_LOCAL_DRAM: 2>, 'DLA_GLOBAL_DRAM': <MemoryPoolType.DLA_GLOBAL_DRAM: 3>, 'TACTIC_DRAM': <MemoryPoolType.TACTIC_DRAM: 4>, 'TACTIC_SHARED_MEMORY': <MemoryPoolType.TACTIC_SHARED_MEMORY: 5>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class MoEActType:
    """
    The activation type that may be performed by an MoE layer

    Members:

      NONE :

      SILU :
    """
    NONE: typing.ClassVar[MoEActType]  # value = <MoEActType.NONE: 0>
    SILU: typing.ClassVar[MoEActType]  # value = <MoEActType.SILU: 1>
    __members__: typing.ClassVar[dict[str, MoEActType]]  # value = {'NONE': <MoEActType.NONE: 0>, 'SILU': <MoEActType.SILU: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class NetworkDefinitionCreationFlag:
    """
    List of immutable network properties expressed at network creation time. For example, to enable strongly typed mode, pass a value of ``1 << int(NetworkDefinitionCreationFlag.STRONGLY_TYPED)`` to :func:`create_network`

    Members:

      EXPLICIT_BATCH : [DEPRECATED] Ignored because networks are always "explicit batch" in TensorRT 10.0.

      STRONGLY_TYPED : Specify that every tensor in the network has a data type defined in the network following only type inference rules and the inputs/operator annotations. Setting layer precision and layer output types is not allowed, and the network output types will be inferred based on the input types and the type inference rules

      PREFER_AOT_PYTHON_PLUGINS : If set, for a Python plugin with both AOT and JIT implementations, the AOT implementation will be used.

      PREFER_JIT_PYTHON_PLUGINS : If set, for a Python plugin with both AOT and JIT implementations, the JIT implementation will be used.
    """
    EXPLICIT_BATCH: typing.ClassVar[NetworkDefinitionCreationFlag]  # value = <NetworkDefinitionCreationFlag.EXPLICIT_BATCH: 0>
    PREFER_AOT_PYTHON_PLUGINS: typing.ClassVar[NetworkDefinitionCreationFlag]  # value = <NetworkDefinitionCreationFlag.PREFER_AOT_PYTHON_PLUGINS: 3>
    PREFER_JIT_PYTHON_PLUGINS: typing.ClassVar[NetworkDefinitionCreationFlag]  # value = <NetworkDefinitionCreationFlag.PREFER_JIT_PYTHON_PLUGINS: 2>
    STRONGLY_TYPED: typing.ClassVar[NetworkDefinitionCreationFlag]  # value = <NetworkDefinitionCreationFlag.STRONGLY_TYPED: 1>
    __members__: typing.ClassVar[dict[str, NetworkDefinitionCreationFlag]]  # value = {'EXPLICIT_BATCH': <NetworkDefinitionCreationFlag.EXPLICIT_BATCH: 0>, 'STRONGLY_TYPED': <NetworkDefinitionCreationFlag.STRONGLY_TYPED: 1>, 'PREFER_AOT_PYTHON_PLUGINS': <NetworkDefinitionCreationFlag.PREFER_AOT_PYTHON_PLUGINS: 3>, 'PREFER_JIT_PYTHON_PLUGINS': <NetworkDefinitionCreationFlag.PREFER_JIT_PYTHON_PLUGINS: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class NodeIndices:
    __hash__: typing.ClassVar[None] = None
    def __bool__(self: collections.abc.Sequence[typing.SupportsInt]) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self: collections.abc.Sequence[typing.SupportsInt], x: typing.SupportsInt) -> bool:
        """
        Return true the container contains ``x``
        """
    @typing.overload
    def __delitem__(self: collections.abc.Sequence[typing.SupportsInt], arg0: typing.SupportsInt) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self: collections.abc.Sequence[typing.SupportsInt], arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self: collections.abc.Sequence[typing.SupportsInt], arg0: collections.abc.Sequence[typing.SupportsInt]) -> bool:
        ...
    @typing.overload
    def __getitem__(self: collections.abc.Sequence[typing.SupportsInt], s: slice) -> list[int]:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self: collections.abc.Sequence[typing.SupportsInt], arg0: typing.SupportsInt) -> int:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: collections.abc.Sequence[typing.SupportsInt]) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self: collections.abc.Sequence[typing.SupportsInt]) -> collections.abc.Iterator[int]:
        ...
    def __len__(self: collections.abc.Sequence[typing.SupportsInt]) -> int:
        ...
    def __ne__(self: collections.abc.Sequence[typing.SupportsInt], arg0: collections.abc.Sequence[typing.SupportsInt]) -> bool:
        ...
    def __repr__(self: collections.abc.Sequence[typing.SupportsInt]) -> str:
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __setitem__(self: collections.abc.Sequence[typing.SupportsInt], arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None:
        ...
    @typing.overload
    def __setitem__(self: collections.abc.Sequence[typing.SupportsInt], arg0: slice, arg1: collections.abc.Sequence[typing.SupportsInt]) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self: collections.abc.Sequence[typing.SupportsInt], x: typing.SupportsInt) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self: collections.abc.Sequence[typing.SupportsInt]) -> None:
        """
        Clear the contents
        """
    def count(self: collections.abc.Sequence[typing.SupportsInt], x: typing.SupportsInt) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self: collections.abc.Sequence[typing.SupportsInt], L: collections.abc.Sequence[typing.SupportsInt]) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self: collections.abc.Sequence[typing.SupportsInt], L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self: collections.abc.Sequence[typing.SupportsInt], i: typing.SupportsInt, x: typing.SupportsInt) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self: collections.abc.Sequence[typing.SupportsInt]) -> int:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self: collections.abc.Sequence[typing.SupportsInt], i: typing.SupportsInt) -> int:
        """
        Remove and return the item at index ``i``
        """
    def remove(self: collections.abc.Sequence[typing.SupportsInt], x: typing.SupportsInt) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
class OnnxParser:
    """

        This class is used for parsing ONNX models into a TensorRT network definition

        :ivar num_errors: :class:`int` The number of errors that occurred during prior calls to :func:`parse`
    """
    @staticmethod
    def __enter__(this):
        ...
    @staticmethod
    def __exit__(this, exc_type, exc_value, traceback):
        """

            Context managers are deprecated and have no effect. Objects are automatically freed when
            the reference count reaches 0.

        """
    def __del__(self) -> None:
        ...
    def __init__(self, network: INetworkDefinition, logger: ILogger) -> None:
        """
            :arg network: The network definition to which the parser will write.
            :arg logger: The logger to use.
        """
    def clear_errors(self) -> None:
        """
            Clear errors from prior calls to :func:`parse`
        """
    def clear_flag(self, flag: ...) -> None:
        """
            Clears the parser flag from the enabled flags.

            :arg flag: The flag to clear.
        """
    def get_error(self, index: typing.SupportsInt) -> ...:
        """
            Get an error that occurred during prior calls to :func:`parse`

            :arg index: Index of the error
        """
    def get_flag(self, flag: ...) -> bool:
        """
            Check if a build mode flag is set.

            :arg flag: The flag to check.

            :returns: A `bool` indicating whether the flag is set.
        """
    def get_layer_output_tensor(self, name: str, i: typing.SupportsInt) -> ITensor:
        """
            Get the i-th output ITensor object for the ONNX layer "name".

           In the case of multiple nodes sharing the same name this function will return
           the output tensors of the first instance of the node in the ONNX graph.

            :arg name: The name of the ONNX layer.

            :arg i: The index of the output.

            :returns: The output tensor or None if the layer was not found or an invalid index was provided.
        """
    def get_subgraph_nodes(self, index: typing.SupportsInt) -> list:
        """
            Get the nodes of the specified subgraph. Calling before \\p supportsModelV2 is an undefined behavior.
            Will return an empty list by default.

            :arg index: Index of the subgraph.
            :returns: List[int]
                A list of node indices in the subgraph.
        """
    def get_used_vc_plugin_libraries(self) -> list[str]:
        """
            Query the plugin libraries needed to implement operations used by the parser in a version-compatible engine.

            This provides a list of plugin libraries on the filesystem needed to implement operations
            in the parsed network.  If you are building a version-compatible engine using this network,
            provide this list to IBuilderConfig.set_plugins_to_serialize() to serialize these plugins along
            with the version-compatible engine, or, if you want to ship these plugin libraries externally
            to the engine, ensure that IPluginRegistry.load_library() is used to load these libraries in the
            appropriate runtime before deserializing the corresponding engine.

            :returns: List[str] List of plugin libraries found by the parser.

            :raises: :class:`RuntimeError` if an internal error occurred when trying to fetch the list of plugin libraries.
        """
    def is_subgraph_supported(self, index: typing.SupportsInt) -> bool:
        """
            Returns whether the subgraph is supported. Calling before \\p supportsModelV2 is an undefined behavior.
            Will return false by default.

            :arg index: Index of the subgraph to be checked.
            :returns: true if subgraph is supported
        """
    def load_initializer(self, name: str, data: typing.SupportsInt, size: typing.SupportsInt) -> bool:
        """
            Prompt the ONNX parser to load an initializer with user-provided binary data.
            The lifetime of the data must exceed the lifetime of the parser.

            All user-provided initializers must be provided prior to calling parse_model_proto().

            This function can be called multiple times to specify the names of multiple initializers.

            Calling this function with an initializer previously specified will overwrite the previous instance.

            This function will return false if initializer validation fails. Possible validation errors are:
             * This function was called prior to load_model_proto().
             * The requested initializer was not found in the model.
             * The size of the data provided is different from the corresponding initializer in the model.

            :arg name: name of the initializer.
            :arg data: binary data of the initializer.
            :arg size: the size of the binary data.

            :returns: true if the initializer was successfully loaded
        """
    def load_model_proto(self, model: collections.abc.Buffer, path: str = None) -> bool:
        """
            Load a serialized ONNX model into the parser. Unlike the parse(), parse_from_file(), or parse_with_weight_descriptors()
            functions, this function does not immediately convert the model into a TensorRT INetworkDefinition. Using this function
            allows users to provide their own initializers for the ONNX model through the load_initializer() function.

            Only one model can be loaded at a time. Subsequent calls to load_model_proto() will result in an error.

            To begin the conversion of the model into a TensorRT INetworkDefinition, use parse_model_proto().

            :arg model: The serialized ONNX model.
            :arg path: The path to the model file. Only required if the model has externally stored weights.

            :returns: true if the model was loaded successfully
        """
    def parse(self, model: collections.abc.Buffer, path: str = None) -> bool:
        """
            Parse a serialized ONNX model into the TensorRT network.

            :arg model: The serialized ONNX model.
            :arg path: The path to the model file. Only required if the model has externally stored weights.

            :returns: true if the model was parsed successfully
        """
    def parse_from_file(self, model: str) -> bool:
        """
            Parse an ONNX model from file into a TensorRT network.

            :arg model: The path to an ONNX model.

            :returns: true if the model was parsed successfully
        """
    def parse_model_proto(self) -> bool:
        """
            Begin the parsing and conversion process of the loaded ONNX model into a TensorRT INetworkDefinition.

            :returns: true if the model was parsed successfully.
        """
    def parse_with_weight_descriptors(self, model: collections.abc.Buffer) -> bool:
        """
            [DEPRECATED] Deprecated in TensorRT 10.13. See load_initializers.

            Parse a serialized ONNX model into the TensorRT network with consideration of user provided weights.

            :arg model: The serialized ONNX model.

            :returns: true if the model was parsed successfully
        """
    def set_builder_config(self, builder_config: IBuilderConfig) -> bool:
        """
            Set the BuilderConfig for the parser.

            :arg builder_config: The BuilderConfig to set.

            :returns: true if the BuilderConfig was set successfully, false otherwise.
        """
    def set_flag(self, flag: ...) -> None:
        """
            Add the input parser flag to the already enabled flags.

            :arg flag: The flag to set.
        """
    def supports_model(self, model: collections.abc.Buffer, path: str = None) -> tuple[bool, list[tuple[list[int], bool]]]:
        """
            [DEPRECATED] Deprecated in TensorRT 10.1. See supports_model_v2.

            Check whether TensorRT supports a particular ONNX model.

            :arg model: The serialized ONNX model.
            :arg path: The path to the model file. Only required if the model has externally stored weights.

            :returns: Tuple[bool, List[Tuple[NodeIndices, bool]]]
                The first element of the tuple indicates whether the model is supported.
                The second indicates subgraphs (by node index) in the model and whether they are supported.
        """
    def supports_model_v2(self, model: collections.abc.Buffer, path: str = None) -> bool:
        """
            Check whether TensorRT supports a particular ONNX model.
            Query each subgraph with num_subgraphs, is_subgraph_supported, get_subgraph_nodes.

            :arg model: The serialized ONNX model.
            :arg path: The path to the model file. Only required if the model has externally stored weights.
            :returns: true if the model is supported
        """
    def supports_operator(self, op_name: str) -> bool:
        """
            Returns whether the specified operator may be supported by the parser.
            Note that a result of true does not guarantee that the operator will be supported in all cases (i.e., this function may return false-positives).

            :arg op_name:  The name of the ONNX operator to check for support
        """
    @property
    def flags(self) -> int:
        ...
    @flags.setter
    def flags(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def num_errors(self) -> int:
        ...
    @property
    def num_subgraphs(self) -> int:
        ...
class OnnxParserFlag:
    """

        Flags that control how an ONNX model gets parsed.


    Members:

      NATIVE_INSTANCENORM :
       Parse the ONNX model into the INetworkDefinition with the intention of using TensorRT's native layer implementation over the plugin implementation for InstanceNormalization nodes.
       This flag is required when building version-compatible or hardware-compatible engines.
       This flag is ON by default.


      ENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA :
        Enable UINT8 as a quantization data type and asymmetric quantization with non-zero zero-point values in Quantize and Dequantize nodes.
        The resulting engine must be built targeting DLA version >= 3.16.
        This flag is OFF by default.


      REPORT_CAPABILITY_DLA :
        Parse the ONNX model with per-node validation for DLA. If this flag is set, is_subgraph_supported() will
        also return capability in the context of DLA support.
        When this flag is set, a valid BuilderConfig must be provided to the parser via set_builder_config().
        This flag is OFF by default.


      ENABLE_PLUGIN_OVERRIDE :
        Allow a loaded plugin with the same name as an ONNX operator type to override the default ONNX implementation,
        even if the plugin namespace attribute is not set.
        This flag is useful for custom plugins that are intended to replace standard ONNX operators, for example to provide
        alternative implementations or improved performance.
        This flag is OFF by default.


      ADJUST_FOR_DLA :
        Parse the ONNX model with adjustments to make layers more amenable to running on DLA.
        This flag is OFF by default.

    """
    ADJUST_FOR_DLA: typing.ClassVar[OnnxParserFlag]  # value = <OnnxParserFlag.ADJUST_FOR_DLA: 4>
    ENABLE_PLUGIN_OVERRIDE: typing.ClassVar[OnnxParserFlag]  # value = <OnnxParserFlag.ENABLE_PLUGIN_OVERRIDE: 3>
    ENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA: typing.ClassVar[OnnxParserFlag]  # value = <OnnxParserFlag.ENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA: 1>
    NATIVE_INSTANCENORM: typing.ClassVar[OnnxParserFlag]  # value = <OnnxParserFlag.NATIVE_INSTANCENORM: 0>
    REPORT_CAPABILITY_DLA: typing.ClassVar[OnnxParserFlag]  # value = <OnnxParserFlag.REPORT_CAPABILITY_DLA: 2>
    __members__: typing.ClassVar[dict[str, OnnxParserFlag]]  # value = {'NATIVE_INSTANCENORM': <OnnxParserFlag.NATIVE_INSTANCENORM: 0>, 'ENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA': <OnnxParserFlag.ENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA: 1>, 'REPORT_CAPABILITY_DLA': <OnnxParserFlag.REPORT_CAPABILITY_DLA: 2>, 'ENABLE_PLUGIN_OVERRIDE': <OnnxParserFlag.ENABLE_PLUGIN_OVERRIDE: 3>, 'ADJUST_FOR_DLA': <OnnxParserFlag.ADJUST_FOR_DLA: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class OnnxParserRefitter:
    """

        This is an interface designed to refit weights from an ONNX model.
    """
    def __init__(self, refitter: Refitter, logger: ILogger) -> None:
        """
            :arg refitter: The Refitter object used to refit the model.
            :arg logger: The logger to use.
        """
    def clear_errors(self) -> None:
        """
            Clear errors from prior calls to :func:`refitFromBytes` or :func:`refitFromFile`.
        """
    def get_error(self, index: typing.SupportsInt) -> ParserError:
        """
            Get an error that occurred during prior calls to :func:`refitFromBytes` or :func:`refitFromFile`.

            :arg index: Index of the error
        """
    def load_initializer(self, name: str, data: typing.SupportsInt, size: typing.SupportsInt) -> bool:
        """
            Prompt the ONNX refitter to load an initializer with user-provided binary data.
            The lifetime of the data must exceed the lifetime of the refitter.

            All user-provided initializers must be provided prior to calling refit_model_proto().

            This function can be called multiple times to specify the names of multiple initializers.

            Calling this function with an initializer previously specified will overwrite the previous instance.

            This function will return false if initializer validation fails. Possible validation errors are:
             * This function was called prior to load_model_proto().
             * The requested initializer was not found in the model.
             * The size of the data provided is different from the corresponding initializer in the model.

            :arg name: name of the initializer.
            :arg data: binary data of the initializer.
            :arg size: the size of the binary data.

            :returns: true if the initializer was successfully loaded.
        """
    def load_model_proto(self, model: collections.abc.Buffer, path: str = None) -> bool:
        """
            Load a serialized ONNX model into the refitter. Unlike the refit() or refit_from_file()
            functions, this function does not immediately begin the refit process. Using this function
            allows users to provide their own initializers for the ONNX model through the load_initializer() function.

            Only one model can be loaded at a time. Subsequent calls to load_model_proto() will result in an error.

            To begin the refit process, use refit_model_proto().

            :arg model: The serialized ONNX model.
            :arg path: The path to the model file. Only required if the model has externally stored weights.

            :returns: true if the model was loaded successfully.
        """
    def refit_from_bytes(self, model: collections.abc.Buffer, path: str = None) -> bool:
        """
            Load a serialized ONNX model from memory and perform weight refit.

            :arg model: The serialized ONNX model.
            :arg path: The path to the model file. Only required if the model has externally stored weights.

            :returns: true if all the weights in the engine were refit successfully.
        """
    def refit_from_file(self, model: str) -> bool:
        """
            Load and parse a ONNX model from disk and perform weight refit.

            :arg model: The path to an ONNX model.

            :returns: true if the model was loaded successfully, and if all the weights in the engine were refit successfully.
        """
    def refit_model_proto(self) -> bool:
        """
            Begin the refit process from the loaded ONNX model.

            :returns: true if the model was refit successfully.
        """
    @property
    def num_errors(self) -> int:
        ...
class PaddingMode:
    """

        Enumerates types of padding available in convolution, deconvolution and pooling layers.
        Padding mode takes precedence if both :attr:`padding_mode` and :attr:`pre_padding` are set.

        |  EXPLICIT* corresponds to explicit padding.
        |  SAME* implicitly calculates padding such that the output dimensions are the same as the input dimensions. For convolution and pooling,
            output dimensions are determined by ceil(input dimensions, stride).
        |  CAFFE* corresponds to symmetric padding.


    Members:

      EXPLICIT_ROUND_DOWN : Use explicit padding, rounding the output size down

      EXPLICIT_ROUND_UP : Use explicit padding, rounding the output size up

      SAME_UPPER : Use SAME padding, with :attr:`pre_padding` <= :attr:`post_padding`

      SAME_LOWER : Use SAME padding, with :attr:`pre_padding` >= :attr:`post_padding`
    """
    EXPLICIT_ROUND_DOWN: typing.ClassVar[PaddingMode]  # value = <PaddingMode.EXPLICIT_ROUND_DOWN: 0>
    EXPLICIT_ROUND_UP: typing.ClassVar[PaddingMode]  # value = <PaddingMode.EXPLICIT_ROUND_UP: 1>
    SAME_LOWER: typing.ClassVar[PaddingMode]  # value = <PaddingMode.SAME_LOWER: 3>
    SAME_UPPER: typing.ClassVar[PaddingMode]  # value = <PaddingMode.SAME_UPPER: 2>
    __members__: typing.ClassVar[dict[str, PaddingMode]]  # value = {'EXPLICIT_ROUND_DOWN': <PaddingMode.EXPLICIT_ROUND_DOWN: 0>, 'EXPLICIT_ROUND_UP': <PaddingMode.EXPLICIT_ROUND_UP: 1>, 'SAME_UPPER': <PaddingMode.SAME_UPPER: 2>, 'SAME_LOWER': <PaddingMode.SAME_LOWER: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ParserError:
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def code(self) -> ErrorCode:
        """
            :returns: The error code
        """
    def desc(self) -> str:
        """
            :returns: Description of the error
        """
    def file(self) -> str:
        """
            :returns: Source file in which the error occurred
        """
    def func(self) -> str:
        """
            :returns: Source function in which the error occurred
        """
    def line(self) -> int:
        """
            :returns: Source line at which the error occurred
        """
    def local_function_stack(self) -> list[str]:
        """
            :returns: Current stack trace of local functions in which the error occurred
        """
    def local_function_stack_size(self) -> int:
        """
            :returns: Size of the current stack trace of local functions in which the error occurred
        """
    def node(self) -> int:
        """
            :returns: Index of the Onnx model node in which the error occurred
        """
    def node_name(self) -> str:
        """
            :returns: Name of the node in the model in which the error occurred
        """
    def node_operator(self) -> str:
        """
            :returns: Name of the node operation in the model in which the error occurred
        """
class Permutation:
    """

        The elements of the permutation. The permutation is applied as outputDimensionIndex = permutation[inputDimensionIndex], so to permute from CHW order to HWC order, the required permutation is [1, 2, 0], and to permute from HWC to CHW, the required permutation is [2, 0, 1].

        It supports iteration and indexing and is implicitly convertible to/from Python iterables (like :class:`tuple` or :class:`list` ). Therefore, you can use those classes in place of :class:`Permutation` .
    """
    def __getitem__(self, arg0: typing.SupportsInt) -> int:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def __setitem__(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
class PluginArgDataType:
    """
    Members:

      INT8

      INT16

      INT32
    """
    INT16: typing.ClassVar[PluginArgDataType]  # value = <PluginArgDataType.INT16: 1>
    INT32: typing.ClassVar[PluginArgDataType]  # value = <PluginArgDataType.INT32: 2>
    INT8: typing.ClassVar[PluginArgDataType]  # value = <PluginArgDataType.INT8: 0>
    __members__: typing.ClassVar[dict[str, PluginArgDataType]]  # value = {'INT8': <PluginArgDataType.INT8: 0>, 'INT16': <PluginArgDataType.INT16: 1>, 'INT32': <PluginArgDataType.INT32: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PluginArgType:
    """
    Members:

      INT
    """
    INT: typing.ClassVar[PluginArgType]  # value = <PluginArgType.INT: 0>
    __members__: typing.ClassVar[dict[str, PluginArgType]]  # value = {'INT': <PluginArgType.INT: 0>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PluginCapabilityType:
    """

        Enumerates the different capability types a IPluginV3 object may have.


    Members:

      CORE

      BUILD

      RUNTIME
    """
    BUILD: typing.ClassVar[PluginCapabilityType]  # value = <PluginCapabilityType.BUILD: 1>
    CORE: typing.ClassVar[PluginCapabilityType]  # value = <PluginCapabilityType.CORE: 0>
    RUNTIME: typing.ClassVar[PluginCapabilityType]  # value = <PluginCapabilityType.RUNTIME: 2>
    __members__: typing.ClassVar[dict[str, PluginCapabilityType]]  # value = {'CORE': <PluginCapabilityType.CORE: 0>, 'BUILD': <PluginCapabilityType.BUILD: 1>, 'RUNTIME': <PluginCapabilityType.RUNTIME: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PluginCreatorVersion:
    """

        Enum to identify version of the plugin creator.


    Members:

      V1

      V1_PYTHON
    """
    V1: typing.ClassVar[PluginCreatorVersion]  # value = <PluginCreatorVersion.V1: 0>
    V1_PYTHON: typing.ClassVar[PluginCreatorVersion]  # value = <PluginCreatorVersion.V1_PYTHON: 64>
    __members__: typing.ClassVar[dict[str, PluginCreatorVersion]]  # value = {'V1': <PluginCreatorVersion.V1: 0>, 'V1_PYTHON': <PluginCreatorVersion.V1_PYTHON: 64>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PluginField:
    """

        Contains plugin attribute field names and associated data.
        This information can be parsed to decode necessary plugin metadata

        :ivar name: :class:`str` Plugin field attribute name.
        :ivar data: :class:`buffer` Plugin field attribute data.
        :ivar type: :class:`PluginFieldType` Plugin field attribute type.
        :ivar size: :class:`int` Number of data entries in the Plugin attribute.
    """
    type: PluginFieldType
    @typing.overload
    def __init__(self, name: FallbackString = '') -> None:
        ...
    @typing.overload
    def __init__(self, name: FallbackString, data: collections.abc.Buffer, type: PluginFieldType = ...) -> None:
        ...
    @property
    def data(self) -> numpy.ndarray:
        ...
    @data.setter
    def data(self, arg1: collections.abc.Buffer) -> None:
        ...
    @property
    def name(self) -> str:
        ...
    @name.setter
    def name(self, arg1: FallbackString) -> None:
        ...
    @property
    def size(self) -> int:
        ...
    @size.setter
    def size(self, arg0: typing.SupportsInt) -> None:
        ...
class PluginFieldCollection:
    def __bool__(self) -> bool:
        """
        Check whether the list is nonempty
        """
    @typing.overload
    def __delitem__(self, arg0: typing.SupportsInt) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, s: slice) -> PluginFieldCollection:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: typing.SupportsInt) -> ...:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: PluginFieldCollection) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[...]:
        ...
    def __len__(self) -> int:
        ...
    @typing.overload
    def __setitem__(self, arg0: typing.SupportsInt, arg1: ...) -> None:
        ...
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: PluginFieldCollection) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self, x: ...) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self) -> None:
        """
        Clear the contents
        """
    @typing.overload
    def extend(self, L: PluginFieldCollection) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self, i: typing.SupportsInt, x: ...) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> ...:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self, i: typing.SupportsInt) -> ...:
        """
        Remove and return the item at index ``i``
        """
class PluginFieldCollection_:
    """

        Contains plugin attribute field names and associated data.
        This information can be parsed to decode necessary plugin metadata

        The collection behaves like a Python iterable.
    """
    def __getitem__(self, arg0: typing.SupportsInt) -> PluginField:
        ...
    def __init__(self, arg0: PluginFieldCollection) -> None:
        ...
    def __len__(self) -> int:
        ...
class PluginFieldType:
    """

        The possible field types for custom layer.


    Members:

      FLOAT16

      FLOAT32

      FLOAT64

      INT8

      INT16

      INT32

      CHAR

      DIMS

      UNKNOWN

      BF16

      INT64

      FP8

      INT4

      FP4
    """
    BF16: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.BF16: 9>
    CHAR: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.CHAR: 6>
    DIMS: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.DIMS: 7>
    FLOAT16: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.FLOAT16: 0>
    FLOAT32: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.FLOAT32: 1>
    FLOAT64: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.FLOAT64: 2>
    FP4: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.FP4: 13>
    FP8: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.FP8: 11>
    INT16: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.INT16: 4>
    INT32: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.INT32: 5>
    INT4: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.INT4: 12>
    INT64: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.INT64: 10>
    INT8: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.INT8: 3>
    UNKNOWN: typing.ClassVar[PluginFieldType]  # value = <PluginFieldType.UNKNOWN: 8>
    __members__: typing.ClassVar[dict[str, PluginFieldType]]  # value = {'FLOAT16': <PluginFieldType.FLOAT16: 0>, 'FLOAT32': <PluginFieldType.FLOAT32: 1>, 'FLOAT64': <PluginFieldType.FLOAT64: 2>, 'INT8': <PluginFieldType.INT8: 3>, 'INT16': <PluginFieldType.INT16: 4>, 'INT32': <PluginFieldType.INT32: 5>, 'CHAR': <PluginFieldType.CHAR: 6>, 'DIMS': <PluginFieldType.DIMS: 7>, 'UNKNOWN': <PluginFieldType.UNKNOWN: 8>, 'BF16': <PluginFieldType.BF16: 9>, 'INT64': <PluginFieldType.INT64: 10>, 'FP8': <PluginFieldType.FP8: 11>, 'INT4': <PluginFieldType.INT4: 12>, 'FP4': <PluginFieldType.FP4: 13>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PluginTensorDesc:
    """

        Fields that a plugin might see for an input or output.

        `scale` is only valid when the `type` is `DataType.INT8`. TensorRT will set the value to -1.0 if it is invalid.

        :ivar dims: :class:`Dims` 	Dimensions.
        :ivar format: :class:`TensorFormat` Tensor format.
        :ivar type: :class:`DataType` Type.
        :ivar scale: :class:`float` Scale for INT8 data type.
    """
    dims: Dims
    format: ...
    type: DataType
    def __init__(self) -> None:
        ...
    @property
    def scale(self) -> float:
        ...
    @scale.setter
    def scale(self, arg0: typing.SupportsFloat) -> None:
        ...
class PoolingType:
    """
    The type of pooling to perform in a pooling layer.

    Members:

      MAX : Maximum over elements

      AVERAGE : Average over elements. If the tensor is padded, the count includes the padding

      MAX_AVERAGE_BLEND : Blending between the max pooling and average pooling: `(1-blendFactor)*maxPool + blendFactor*avgPool`
    """
    AVERAGE: typing.ClassVar[PoolingType]  # value = <PoolingType.AVERAGE: 1>
    MAX: typing.ClassVar[PoolingType]  # value = <PoolingType.MAX: 0>
    MAX_AVERAGE_BLEND: typing.ClassVar[PoolingType]  # value = <PoolingType.MAX_AVERAGE_BLEND: 2>
    __members__: typing.ClassVar[dict[str, PoolingType]]  # value = {'MAX': <PoolingType.MAX: 0>, 'AVERAGE': <PoolingType.AVERAGE: 1>, 'MAX_AVERAGE_BLEND': <PoolingType.MAX_AVERAGE_BLEND: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PreviewFeature:
    """

        List of Preview Features that can be enabled. Preview Features have been fully tested but are not yet as stable as other features in TensorRT.
        They are provided as opt-in features for at least one release.
        For example, to enable faster dynamic shapes, call :func:`set_preview_feature` with ``PreviewFeature.PROFILE_SHARING_0806``


    Members:

      PROFILE_SHARING_0806 :
        [DEPRECATED] Allows optimization profiles to be shared across execution contexts. The default value for this flag is on in TensorRT 10.0. Turning if off is deprecated.


      ALIASED_PLUGIN_IO_10_03 :
        Allows plugin I/O to be aliased when using IPluginV3OneBuildV2.


      RUNTIME_ACTIVATION_RESIZE_10_10 :
        Allow update_device_memory_size_for_shapes to resize runner internal activation memory by changing the allocation algorithm. Using this feature can reduce runtime memory requirement when the actual input tensor shapes are smaller than the maximum input tensor dimensions.


      MULTIDEVICE_RUNTIME_10_16 :
        Allow using multi-device mode in the builder. This enables running inference on multiple GPUs on supported platforms.
    """
    ALIASED_PLUGIN_IO_10_03: typing.ClassVar[PreviewFeature]  # value = <PreviewFeature.ALIASED_PLUGIN_IO_10_03: 1>
    MULTIDEVICE_RUNTIME_10_16: typing.ClassVar[PreviewFeature]  # value = <PreviewFeature.MULTIDEVICE_RUNTIME_10_16: 3>
    PROFILE_SHARING_0806: typing.ClassVar[PreviewFeature]  # value = <PreviewFeature.PROFILE_SHARING_0806: 0>
    RUNTIME_ACTIVATION_RESIZE_10_10: typing.ClassVar[PreviewFeature]  # value = <PreviewFeature.RUNTIME_ACTIVATION_RESIZE_10_10: 2>
    __members__: typing.ClassVar[dict[str, PreviewFeature]]  # value = {'PROFILE_SHARING_0806': <PreviewFeature.PROFILE_SHARING_0806: 0>, 'ALIASED_PLUGIN_IO_10_03': <PreviewFeature.ALIASED_PLUGIN_IO_10_03: 1>, 'RUNTIME_ACTIVATION_RESIZE_10_10': <PreviewFeature.RUNTIME_ACTIVATION_RESIZE_10_10: 2>, 'MULTIDEVICE_RUNTIME_10_16': <PreviewFeature.MULTIDEVICE_RUNTIME_10_16: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Profiler(IProfiler):
    """

        When this class is added to an :class:`IExecutionContext`, the profiler will be called once per layer for each invocation of :func:`IExecutionContext.execute_v2()`.

        It is not recommended to run inference with profiler enabled when the inference execution time is critical since the profiler may affect execution time negatively.
    """
    def __init__(self) -> None:
        ...
    def report_layer_time(self: IProfiler, layer_name: str, ms: typing.SupportsFloat) -> None:
        """
            Prints time in milliseconds for each layer to stdout.

            :arg layer_name: The name of the layer, set when constructing the :class:`INetworkDefinition` .
            :arg ms: The time in milliseconds to execute the layer.
        """
class ProfilingVerbosity:
    """
    Profiling verbosity in NVTX annotations and the engine inspector

    Members:

      LAYER_NAMES_ONLY : Print only the layer names. This is the default setting.

      DETAILED : Print detailed layer information including layer names and layer parameters.

      NONE : Do not print any layer information.
    """
    DETAILED: typing.ClassVar[ProfilingVerbosity]  # value = <ProfilingVerbosity.DETAILED: 2>
    LAYER_NAMES_ONLY: typing.ClassVar[ProfilingVerbosity]  # value = <ProfilingVerbosity.LAYER_NAMES_ONLY: 0>
    NONE: typing.ClassVar[ProfilingVerbosity]  # value = <ProfilingVerbosity.NONE: 1>
    __members__: typing.ClassVar[dict[str, ProfilingVerbosity]]  # value = {'LAYER_NAMES_ONLY': <ProfilingVerbosity.LAYER_NAMES_ONLY: 0>, 'DETAILED': <ProfilingVerbosity.DETAILED: 2>, 'NONE': <ProfilingVerbosity.NONE: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class QuantizationFlag:
    """
    List of valid flags for quantizing the network to int8.

    Members:

      CALIBRATE_BEFORE_FUSION : Run int8 calibration pass before layer fusion. Only valid for IInt8LegacyCalibrator and IInt8EntropyCalibrator. We always run int8 calibration pass before layer fusion for IInt8MinMaxCalibrator and IInt8EntropyCalibrator2. Disabled by default.
    """
    CALIBRATE_BEFORE_FUSION: typing.ClassVar[QuantizationFlag]  # value = <QuantizationFlag.CALIBRATE_BEFORE_FUSION: 0>
    __members__: typing.ClassVar[dict[str, QuantizationFlag]]  # value = {'CALIBRATE_BEFORE_FUSION': <QuantizationFlag.CALIBRATE_BEFORE_FUSION: 0>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class QuickPluginCreationRequest:
    """
    Members:

      UNKNOWN

      PREFER_JIT

      PREFER_AOT

      STRICT_JIT

      STRICT_AOT
    """
    PREFER_AOT: typing.ClassVar[QuickPluginCreationRequest]  # value = <QuickPluginCreationRequest.PREFER_AOT: 2>
    PREFER_JIT: typing.ClassVar[QuickPluginCreationRequest]  # value = <QuickPluginCreationRequest.PREFER_JIT: 1>
    STRICT_AOT: typing.ClassVar[QuickPluginCreationRequest]  # value = <QuickPluginCreationRequest.STRICT_AOT: 4>
    STRICT_JIT: typing.ClassVar[QuickPluginCreationRequest]  # value = <QuickPluginCreationRequest.STRICT_JIT: 3>
    UNKNOWN: typing.ClassVar[QuickPluginCreationRequest]  # value = <QuickPluginCreationRequest.UNKNOWN: 0>
    __members__: typing.ClassVar[dict[str, QuickPluginCreationRequest]]  # value = {'UNKNOWN': <QuickPluginCreationRequest.UNKNOWN: 0>, 'PREFER_JIT': <QuickPluginCreationRequest.PREFER_JIT: 1>, 'PREFER_AOT': <QuickPluginCreationRequest.PREFER_AOT: 2>, 'STRICT_JIT': <QuickPluginCreationRequest.STRICT_JIT: 3>, 'STRICT_AOT': <QuickPluginCreationRequest.STRICT_AOT: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ReduceOperation:
    """
    The reduce operations that may be performed by a Reduce layer

    Members:

      SUM : Sum of the elements

      PROD : Product of the elements

      MAX : Maximum of the elements

      MIN : Minimum of the elements

      AVG : Average of the elements

      NONE : No reduction
    """
    AVG: typing.ClassVar[ReduceOperation]  # value = <ReduceOperation.AVG: 4>
    MAX: typing.ClassVar[ReduceOperation]  # value = <ReduceOperation.MAX: 2>
    MIN: typing.ClassVar[ReduceOperation]  # value = <ReduceOperation.MIN: 3>
    NONE: typing.ClassVar[ReduceOperation]  # value = <ReduceOperation.NONE: 5>
    PROD: typing.ClassVar[ReduceOperation]  # value = <ReduceOperation.PROD: 1>
    SUM: typing.ClassVar[ReduceOperation]  # value = <ReduceOperation.SUM: 0>
    __members__: typing.ClassVar[dict[str, ReduceOperation]]  # value = {'SUM': <ReduceOperation.SUM: 0>, 'PROD': <ReduceOperation.PROD: 1>, 'MAX': <ReduceOperation.MAX: 2>, 'MIN': <ReduceOperation.MIN: 3>, 'AVG': <ReduceOperation.AVG: 4>, 'NONE': <ReduceOperation.NONE: 5>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Refitter:
    """

        Updates weights in an :class:`ICudaEngine` .

        :ivar error_recorder: :class:`IErrorRecorder` Application-implemented error reporting interface for TensorRT objects.
        :ivar logger: :class:`ILogger` The logger provided when creating the refitter.
        :ivar max_threads: :class:`int` The maximum thread that can be used by the :class:`Refitter`.
        :ivar weights_validation: :class:`bool` The flag to indicate whether to validate weights in the refitting process.
    """
    error_recorder: IErrorRecorder
    weights_validation: bool
    def __del__(self) -> None:
        ...
    def __init__(self, engine: ICudaEngine, logger: ILogger) -> None:
        """
            :arg engine: The engine to refit.
            :arg logger: The logger to use.
        """
    def get_all(self) -> tuple[list[str], list[WeightsRole]]:
        """
            Get description of all weights that could be refitted.

            :returns: The names of layers with refittable weights, and the roles of those weights.
        """
    def get_all_weights(self) -> list[str]:
        """
            Get names of all weights that could be refitted.

            :returns: The names of refittable weights.
        """
    def get_dynamic_range(self, tensor_name: str) -> tuple:
        """
            [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization.

            Gets the dynamic range of a tensor. If the dynamic range was never set, returns the range computed during calibration.

            :arg tensor_name: The name of the tensor whose dynamic range to retrieve.

            :returns: :class:`Tuple[float, float]` A tuple containing the [minimum, maximum] of the dynamic range.
        """
    def get_missing(self) -> tuple[list[str], list[WeightsRole]]:
        """
            Get description of missing weights.

            For example, if some Weights have been set, but the engine was optimized
            in a way that combines weights, any unsupplied Weights in the combination
            are considered missing.

            :returns: The names of layers with missing weights, and the roles of those weights.
        """
    def get_missing_weights(self) -> list[str]:
        """
            Get names of missing weights.

            For example, if some Weights have been set, but the engine was optimized
            in a way that combines weights, any unsupplied Weights in the combination
            are considered missing.

            :returns: The names of missing weights, empty string for unnamed weights.
        """
    def get_named_weights(self, weights_name: str) -> Weights:
        """
            Get weights associated with the given name.

            If the weights were never set, returns null weights and reports an error to the refitter errorRecorder.

            :arg weights_name: The name of the weights to be refitted.

            :returns: Weights associated with the given name.
        """
    def get_tensors_with_dynamic_range(self) -> list[str]:
        """
            [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization.

            Get names of all tensors that have refittable dynamic ranges.

            :returns: The names of tensors with refittable dynamic ranges.
        """
    def get_weights_location(self, weights_name: str) -> TensorLocation:
        """
            Get location for the weights associated with the given name.

            If the weights were never set, returns TensorLocation.HOST and reports an error to the refitter errorRecorder.

            :arg weights_name: The name of the weights to be refitted.

            :returns: Location for the weights associated with the given name.
        """
    def get_weights_prototype(self, weights_name: str) -> Weights:
        """
            Get the weights prototype associated with the given name.

            The dtype and size of weights prototype is the same as weights used for engine building.
            The size of the weights prototype is -1 when the name of the weights is None or does not correspond to any refittable weights.

            :arg weights_name: The name of the weights to be refitted.

            :returns: weights prototype associated with the given name.
        """
    def refit_cuda_engine(self) -> bool:
        """
           Refits associated engine.

           If ``False`` is returned, a subset of weights may have been refitted.

           The behavior is undefined if the engine has pending enqueued work.
           Provided weights on CPU or GPU can be unset and released, or updated after refit_cuda_engine returns.

           IExecutionContexts associated with the engine remain valid for use afterwards. There is no need to set the same
           weights repeatedly for multiple refit calls as the weights memory can be updated directly instead.

           :returns: ``True`` on success, or ``False`` if new weights validation fails or get_missing_weights() != 0 before the call.
        """
    def refit_cuda_engine_async(self, stream_handle: typing.SupportsInt) -> bool:
        """
            Enqueue weights refitting of the associated engine on the given stream.

            If ``False`` is returned, a subset of weights may have been refitted.

            The behavior is undefined if the engine has pending enqueued work on a different stream from the provided one.
            Provided weights on CPU can be unset and released, or updated after refit_cuda_engine_async returns.
            Freeing or updating of the provided weights on GPU can be enqueued on the same stream after refit_cuda_engine_async returns.

            IExecutionContexts associated with the engine remain valid for use afterwards. There is no need to set the same
            weights repeatedly for multiple refit calls as the weights memory can be updated directly instead. The weights
            updating task should use the the same stream as the one used for the refit call.

            :arg stream: The stream to enqueue the weights updating task.

            :returns: ``True`` on success, or ``False`` if new weights validation fails or get_missing_weights() != 0 before the call.
        """
    def set_dynamic_range(self, tensor_name: str, range: collections.abc.Sequence[typing.SupportsFloat]) -> bool:
        """
            [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization.

            Update dynamic range for a tensor.

            :arg tensor_name: The name of the tensor whose dynamic range to update.
            :arg range: The new range.

            :returns: :class:`True` if successful, :class:`False` otherwise.

            Returns false if there is no Int8 engine tensor derived from a network tensor of that name.  If successful, then :func:`get_missing` may report that some weights need to be supplied.
        """
    @typing.overload
    def set_named_weights(self, name: str, weights: Weights) -> bool:
        """
            Specify new weights of given name.
            Possible reasons for rejection are:

            * The name of weights is empty or does not correspond to any refittable weights.
            * The size of the weights is inconsistent with the size returned from calling :func:`get_weights_prototype` with the same name.
            * The dtype of the weights is inconsistent with the dtype returned from calling :func:`get_weights_prototype` with the same name.

            Modifying the weights before :func:`refit_cuda_engine` or :func:`refit_cuda_engine_async` returns
            will result in undefined behavior.

            :arg name: The name of the weights to be refitted.
            :arg weights: The new weights to associate with the name.

            :returns: ``True`` on success, or ``False`` if new weights are rejected.
        """
    @typing.overload
    def set_named_weights(self, name: str, weights: Weights, location: TensorLocation) -> bool:
        """
            Specify new weights on a specified device of given name.
            Possible reasons for rejection are:

            * The name of weights is empty or does not correspond to any refittable weights.
            * The size of the weights is inconsistent with the size returned from calling :func:`get_weights_prototype` with the same name.
            * The dtype of the weights is inconsistent with the dtype returned from calling :func:`get_weights_prototype` with the same name.

            It is allowed to provide some weights on CPU and others on GPU.
            Modifying the weights before :func:`refit_cuda_engine` or :func:`refit_cuda_engine_async` returns
            will result in undefined behavior.

            :arg name: The name of the weights to be refitted.
            :arg weights: The new weights on the specified device.
            :arg location: The location (host vs. device) of the new weights.

            :returns: ``True`` on success, or ``False`` if new weights are rejected.
        """
    def set_weights(self, layer_name: str, role: WeightsRole, weights: Weights) -> bool:
        """
            Specify new weights for a layer of given name.
            Possible reasons for rejection are:

            * There is no such layer by that name.
            * The layer does not have weights with the specified role.
            * The size of weights is inconsistent with the layer’s original specification.

            Modifying the weights before :func:`refit_cuda_engine` or :func:`refit_cuda_engine_async` returns
            will result in undefined behavior.

            :arg layer_name: The name of the layer.
            :arg role: The role of the weights. See :class:`WeightsRole` for more information.
            :arg weights: The weights to refit with.

            :returns: ``True`` on success, or ``False`` if new weights are rejected.
        """
    def unset_named_weights(self, weights_name: str) -> bool:
        """
            Unset weights associated with the given name.

            Unset weights before releasing them.

            :arg weights_name: The name of the weights to be refitted.

            :returns: ``False`` if the weights were never set, returns ``True`` otherwise.
        """
    @property
    def logger(self) -> ILogger:
        ...
    @property
    def max_threads(self) -> int:
        ...
    @max_threads.setter
    def max_threads(self, arg1: typing.SupportsInt) -> bool:
        ...
class ResizeCoordinateTransformation:
    """
    Various modes of how to map the resized coordinate back to the original coordinate.

    Members:

      ALIGN_CORNERS : In this mode, map the resized coordinate back to the original coordinate by the formula: x_original = x_resized * (length_original - 1) / (length_resized - 1).

      ASYMMETRIC : In this mode, map the resized coordinate back to the original coordinate by the formula: x_original = x_resized * (length_original / length_resized).

      HALF_PIXEL : In this mode, map the resized coordinate back to the original coordinate by the formula: x_original = (x_resized + 0.5) * (length_original / length_resized) - 0.5.
    """
    ALIGN_CORNERS: typing.ClassVar[ResizeCoordinateTransformation]  # value = <ResizeCoordinateTransformation.ALIGN_CORNERS: 0>
    ASYMMETRIC: typing.ClassVar[ResizeCoordinateTransformation]  # value = <ResizeCoordinateTransformation.ASYMMETRIC: 1>
    HALF_PIXEL: typing.ClassVar[ResizeCoordinateTransformation]  # value = <ResizeCoordinateTransformation.HALF_PIXEL: 2>
    __members__: typing.ClassVar[dict[str, ResizeCoordinateTransformation]]  # value = {'ALIGN_CORNERS': <ResizeCoordinateTransformation.ALIGN_CORNERS: 0>, 'ASYMMETRIC': <ResizeCoordinateTransformation.ASYMMETRIC: 1>, 'HALF_PIXEL': <ResizeCoordinateTransformation.HALF_PIXEL: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ResizeRoundMode:
    """
    Rounding modes available for the resize layer.

    Members:

      HALF_UP : Round original floating-point coordinate to the nearest integer value, with halfway cases rounded up.

      HALF_DOWN : Round original floating-point coordinate to the nearest integer value, with halfway cases rounded down.

      FLOOR : Round original floating-point coordinate to the nearest integer value less than it.

      CEIL : Round original floating-point coordinate to the nearest integer value larger than it.
    """
    CEIL: typing.ClassVar[ResizeRoundMode]  # value = <ResizeRoundMode.CEIL: 3>
    FLOOR: typing.ClassVar[ResizeRoundMode]  # value = <ResizeRoundMode.FLOOR: 2>
    HALF_DOWN: typing.ClassVar[ResizeRoundMode]  # value = <ResizeRoundMode.HALF_DOWN: 1>
    HALF_UP: typing.ClassVar[ResizeRoundMode]  # value = <ResizeRoundMode.HALF_UP: 0>
    __members__: typing.ClassVar[dict[str, ResizeRoundMode]]  # value = {'HALF_UP': <ResizeRoundMode.HALF_UP: 0>, 'HALF_DOWN': <ResizeRoundMode.HALF_DOWN: 1>, 'FLOOR': <ResizeRoundMode.FLOOR: 2>, 'CEIL': <ResizeRoundMode.CEIL: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ResizeSelector:
    """
    Decides whether the original coordinate is 0 given a resize coordinate less than 2.

    Members:

      FORMULA : Use the transformation formula to calculate the original coordinate.

      UPPER : Return the original coordinate index as 0 given a resize coordinate is less than 2.
    """
    FORMULA: typing.ClassVar[ResizeSelector]  # value = <ResizeSelector.FORMULA: 0>
    UPPER: typing.ClassVar[ResizeSelector]  # value = <ResizeSelector.UPPER: 1>
    __members__: typing.ClassVar[dict[str, ResizeSelector]]  # value = {'FORMULA': <ResizeSelector.FORMULA: 0>, 'UPPER': <ResizeSelector.UPPER: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Runtime:
    """

        Allows a serialized :class:`ICudaEngine` to be deserialized.

        :ivar error_recorder: :class:`IErrorRecorder` Application-implemented error reporting interface for TensorRT objects.
        :ivar gpu_allocator: :class:`IGpuAllocator` The GPU allocator to be used by the :class:`Runtime` . All GPU memory
            acquired will use this allocator. If set to None, the default allocator will be used (Default: cudaMalloc/cudaFree).
        :ivar DLA_core: :class:`int` The DLA core that the engine executes on. Must be between 0 and N-1 where N is the number of available DLA cores.
        :ivar num_DLA_cores: :class:`int` The number of DLA engines available to this builder.
        :ivar logger: :class:`ILogger` The logger provided when creating the refitter.
        :ivar max_threads: :class:`int` The maximum thread that can be used by the :class:`Runtime`.
        :ivar temporary_directory: :class:`str` The temporary directory to use when loading executable code for engines.  If set to None (the default), TensorRT will
                                                attempt to find a suitable directory for use using platform-specific heuristics:
                                                - On UNIX/Linux platforms, TensorRT will first try the TMPDIR environment variable, then fall back to /tmp
                                                - On Windows, TensorRT will try the TEMP environment variable.
        :ivar tempfile_control_flags: :class:`int` Flags which control whether TensorRT is allowed to create in-memory or temporary files.
                                                   See :class:`TempfileControlFlag` for details.
        :ivar engine_host_code_allowed: :class:`bool` Whether this runtime is allowed to deserialize engines that contain host executable code (Default: False).

    """
    engine_host_code_allowed: bool
    error_recorder: IErrorRecorder
    temporary_directory: str
    @staticmethod
    def __enter__(this):
        ...
    @staticmethod
    def __exit__(this, exc_type, exc_value, traceback):
        """

            Context managers are deprecated and have no effect. Objects are automatically freed when
            the reference count reaches 0.

        """
    def __del__(self) -> None:
        ...
    def __init__(self, logger: ILogger) -> None:
        """
            :arg logger: The logger to use.
        """
    @typing.overload
    def deserialize_cuda_engine(self, serialized_engine: collections.abc.Buffer) -> ICudaEngine:
        """
            Deserialize an :class:`ICudaEngine` from host memory.

            :arg serialized_engine: The :class:`buffer` that holds the serialized :class:`ICudaEngine`.

            :returns: The :class:`ICudaEngine`, or None if it could not be deserialized.
        """
    @typing.overload
    def deserialize_cuda_engine(self, stream_reader: IStreamReader) -> ICudaEngine:
        """
            Deserialize an :class:`ICudaEngine` from a stream reader.

            :arg stream_reader: The :class:`PyStreamReader` that will read the serialized :class:`ICudaEngine`. This enables
                deserialization from a file directly.

            :returns: The :class:`ICudaEngine`, or None if it could not be deserialized.
        """
    @typing.overload
    def deserialize_cuda_engine(self, stream_reader_v2: IStreamReaderV2) -> ICudaEngine:
        """
            Deserialize an :class:`ICudaEngine` from a stream reader v2.

            :arg stream_reader: The :class:`PyStreamReaderV2` that will read the serialized :class:`ICudaEngine`. This
                enables deserialization from a file directly, with possible benefits to performance.

            :returns: The :class:`ICudaEngine`, or None if it could not be deserialized.
        """
    def get_plugin_registry(self) -> IPluginRegistry:
        """
            Get the local plugin registry that can be used by the runtime.

            :returns: The local plugin registry that can be used by the runtime.
        """
    def load_runtime(self, path: str) -> Runtime:
        """
            Load IRuntime from the file.

            This method loads a runtime library from a shared library file. The runtime can
            then be used to execute a plan file built with BuilderFlag.VERSION_COMPATIBLE
            and BuilderFlag.EXCLUDE_LEAN_RUNTIME both set and built with the same version
            of TensorRT as the loaded runtime library.

            :ivar path: Path to the runtime lean library.

            :returns: The :class:`IRuntime`, or None if it could not be loaded.
        """
    @property
    def DLA_core(self) -> int:
        ...
    @DLA_core.setter
    def DLA_core(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def logger(self) -> ILogger:
        ...
    @property
    def max_threads(self) -> int:
        ...
    @max_threads.setter
    def max_threads(self, arg1: typing.SupportsInt) -> bool:
        ...
    @property
    def num_DLA_cores(self) -> int:
        ...
    @property
    def tempfile_control_flags(self) -> int:
        ...
    @tempfile_control_flags.setter
    def tempfile_control_flags(self, arg1: typing.SupportsInt) -> None:
        ...
class RuntimePlatform:
    """

        Describes the intended runtime platform for the execution of the TensorRT engine.
        TensorRT provides support for cross-platform engine compatibility when the target runtime platform is different from the build platform.

        **NOTE:** The cross-platform engine will not be able to run on the host platform it was built on.

        **NOTE:** When building a cross-platform engine that also requires version forward compatibility, EXCLUDE_LEAN_RUNTIME must be set to exclude the target platform lean runtime.

        **NOTE:** The cross-platform engine might have performance differences compared to the natively built engine on the target platform.


    Members:

      SAME_AS_BUILD :
        No requirement for cross-platform compatibility. The engine constructed by TensorRT can only run on the identical platform it was built on.


      WINDOWS_AMD64 :
        Designates the target platform for engine execution as Windows AMD64 system. Currently this flag can only be enabled when building engines on Linux AMD64 platforms.
    """
    SAME_AS_BUILD: typing.ClassVar[RuntimePlatform]  # value = <RuntimePlatform.SAME_AS_BUILD: 0>
    WINDOWS_AMD64: typing.ClassVar[RuntimePlatform]  # value = <RuntimePlatform.WINDOWS_AMD64: 1>
    __members__: typing.ClassVar[dict[str, RuntimePlatform]]  # value = {'SAME_AS_BUILD': <RuntimePlatform.SAME_AS_BUILD: 0>, 'WINDOWS_AMD64': <RuntimePlatform.WINDOWS_AMD64: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class SampleMode:
    """
    Controls how ISliceLayer and IGridSample handles out of bounds coordinates

    Members:

      STRICT_BOUNDS : Fail with error when the coordinates are out of bounds.

      WRAP : Coordinates wrap around periodically.

      CLAMP : Out of bounds indices are clamped to bounds

      FILL : Use fill input value when coordinates are out of bounds.

      REFLECT : Coordinates reflect.
    """
    CLAMP: typing.ClassVar[SampleMode]  # value = <SampleMode.CLAMP: 2>
    FILL: typing.ClassVar[SampleMode]  # value = <SampleMode.FILL: 3>
    REFLECT: typing.ClassVar[SampleMode]  # value = <SampleMode.REFLECT: 4>
    STRICT_BOUNDS: typing.ClassVar[SampleMode]  # value = <SampleMode.STRICT_BOUNDS: 0>
    WRAP: typing.ClassVar[SampleMode]  # value = <SampleMode.WRAP: 1>
    __members__: typing.ClassVar[dict[str, SampleMode]]  # value = {'STRICT_BOUNDS': <SampleMode.STRICT_BOUNDS: 0>, 'WRAP': <SampleMode.WRAP: 1>, 'CLAMP': <SampleMode.CLAMP: 2>, 'FILL': <SampleMode.FILL: 3>, 'REFLECT': <SampleMode.REFLECT: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ScaleMode:
    """
    Controls how scale is applied in a Scale layer.

    Members:

      UNIFORM : Identical coefficients across all elements of the tensor.

      CHANNEL : Per-channel coefficients. The channel dimension is assumed to be the third to last dimension.

      ELEMENTWISE : Elementwise coefficients.
    """
    CHANNEL: typing.ClassVar[ScaleMode]  # value = <ScaleMode.CHANNEL: 1>
    ELEMENTWISE: typing.ClassVar[ScaleMode]  # value = <ScaleMode.ELEMENTWISE: 2>
    UNIFORM: typing.ClassVar[ScaleMode]  # value = <ScaleMode.UNIFORM: 0>
    __members__: typing.ClassVar[dict[str, ScaleMode]]  # value = {'UNIFORM': <ScaleMode.UNIFORM: 0>, 'CHANNEL': <ScaleMode.CHANNEL: 1>, 'ELEMENTWISE': <ScaleMode.ELEMENTWISE: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ScatterMode:
    """
    The scatter mode to be done by the scatter layer.

    Members:

      ELEMENT : Scatter Element mode

      ND : Scatter ND mode
    """
    ELEMENT: typing.ClassVar[ScatterMode]  # value = <ScatterMode.ELEMENT: 0>
    ND: typing.ClassVar[ScatterMode]  # value = <ScatterMode.ND: 1>
    __members__: typing.ClassVar[dict[str, ScatterMode]]  # value = {'ELEMENT': <ScatterMode.ELEMENT: 0>, 'ND': <ScatterMode.ND: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class SeekPosition:
    """
    Specifies what the offset is relative to when calling `seek` on an `IStreamReaderV2`.

    Members:

      SET : Offsets forward from the start of the stream.

      CUR : Offsets forward from the current position within the stream.

      END : Offsets backward from the end of the stream.
    """
    CUR: typing.ClassVar[SeekPosition]  # value = <SeekPosition.CUR: 1>
    END: typing.ClassVar[SeekPosition]  # value = <SeekPosition.END: 2>
    SET: typing.ClassVar[SeekPosition]  # value = <SeekPosition.SET: 0>
    __members__: typing.ClassVar[dict[str, SeekPosition]]  # value = {'SET': <SeekPosition.SET: 0>, 'CUR': <SeekPosition.CUR: 1>, 'END': <SeekPosition.END: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class SerializationFlag:
    """
    Valid flags that can be use to creating binary file from engine.

    Members:

      EXCLUDE_WEIGHTS : Exclude weights that can be refitted.

      EXCLUDE_LEAN_RUNTIME : Exclude lean runtime from the plan.

      INCLUDE_REFIT : Remain refittable if originally so.
    """
    EXCLUDE_LEAN_RUNTIME: typing.ClassVar[SerializationFlag]  # value = <SerializationFlag.EXCLUDE_LEAN_RUNTIME: 1>
    EXCLUDE_WEIGHTS: typing.ClassVar[SerializationFlag]  # value = <SerializationFlag.EXCLUDE_WEIGHTS: 0>
    INCLUDE_REFIT: typing.ClassVar[SerializationFlag]  # value = <SerializationFlag.INCLUDE_REFIT: 2>
    __members__: typing.ClassVar[dict[str, SerializationFlag]]  # value = {'EXCLUDE_WEIGHTS': <SerializationFlag.EXCLUDE_WEIGHTS: 0>, 'EXCLUDE_LEAN_RUNTIME': <SerializationFlag.EXCLUDE_LEAN_RUNTIME: 1>, 'INCLUDE_REFIT': <SerializationFlag.INCLUDE_REFIT: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class SubGraphCollection:
    __hash__: typing.ClassVar[None] = None
    def __bool__(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]]) -> bool:
        """
        Check whether the list is nonempty
        """
    def __contains__(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]], x: tuple[collections.abc.Sequence[typing.SupportsInt], bool]) -> bool:
        """
        Return true the container contains ``x``
        """
    @typing.overload
    def __delitem__(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]], arg0: typing.SupportsInt) -> None:
        """
        Delete the list elements at index ``i``
        """
    @typing.overload
    def __delitem__(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]], arg0: slice) -> None:
        """
        Delete list elements using a slice object
        """
    def __eq__(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]], arg0: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]]) -> bool:
        ...
    @typing.overload
    def __getitem__(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]], s: slice) -> list[tuple[list[int], bool]]:
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]], arg0: typing.SupportsInt) -> tuple[list[int], bool]:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]]) -> None:
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: collections.abc.Iterable) -> None:
        ...
    def __iter__(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]]) -> collections.abc.Iterator[tuple[list[int], bool]]:
        ...
    def __len__(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]]) -> int:
        ...
    def __ne__(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]], arg0: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]]) -> bool:
        ...
    @typing.overload
    def __setitem__(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]], arg0: typing.SupportsInt, arg1: tuple[collections.abc.Sequence[typing.SupportsInt], bool]) -> None:
        ...
    @typing.overload
    def __setitem__(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]], arg0: slice, arg1: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]]) -> None:
        """
        Assign list elements using a slice object
        """
    def append(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]], x: tuple[collections.abc.Sequence[typing.SupportsInt], bool]) -> None:
        """
        Add an item to the end of the list
        """
    def clear(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]]) -> None:
        """
        Clear the contents
        """
    def count(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]], x: tuple[collections.abc.Sequence[typing.SupportsInt], bool]) -> int:
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]], L: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]]) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]], L: collections.abc.Iterable) -> None:
        """
        Extend the list by appending all the items in the given list
        """
    def insert(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]], i: typing.SupportsInt, x: tuple[collections.abc.Sequence[typing.SupportsInt], bool]) -> None:
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]]) -> tuple[list[int], bool]:
        """
        Remove and return the last item
        """
    @typing.overload
    def pop(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]], i: typing.SupportsInt) -> tuple[list[int], bool]:
        """
        Remove and return the item at index ``i``
        """
    def remove(self: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], bool]], x: tuple[collections.abc.Sequence[typing.SupportsInt], bool]) -> None:
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
class TacticSource:
    """
    Tactic sources that can provide tactics for TensorRT.

    Members:

      CUBLAS :
            Enables cuBLAS tactics. Disabled by default.
            [DEPRECATED] Deprecated in TensorRT 10.0.
            **NOTE:** Disabling CUBLAS tactic source will cause the cuBLAS handle passed to plugins in attachToContext to be null.


      CUBLAS_LT :
            Enables cuBLAS LT tactics. Disabled by default.
            [DEPRECATED] Deprecated in TensorRT 9.0.


      CUDNN :
            Enables cuDNN tactics. Disabled by default.
            [DEPRECATED] Deprecated in TensorRT 10.0.
            **NOTE:** Disabling CUDNN tactic source will cause the cuDNN handle passed to plugins in attachToContext to be null.


      EDGE_MASK_CONVOLUTIONS :
            Enables convolution tactics implemented with edge mask tables. These tactics tradeoff memory for performance
            by consuming additional memory space proportional to the input size. Enabled by default.


      JIT_CONVOLUTIONS :
            Enables convolution tactics implemented with source-code JIT fusion. The engine building time may increase
            when this is enabled. Enabled by default.

    """
    CUBLAS: typing.ClassVar[TacticSource]  # value = <TacticSource.CUBLAS: 0>
    CUBLAS_LT: typing.ClassVar[TacticSource]  # value = <TacticSource.CUBLAS_LT: 1>
    CUDNN: typing.ClassVar[TacticSource]  # value = <TacticSource.CUDNN: 2>
    EDGE_MASK_CONVOLUTIONS: typing.ClassVar[TacticSource]  # value = <TacticSource.EDGE_MASK_CONVOLUTIONS: 3>
    JIT_CONVOLUTIONS: typing.ClassVar[TacticSource]  # value = <TacticSource.JIT_CONVOLUTIONS: 4>
    __members__: typing.ClassVar[dict[str, TacticSource]]  # value = {'CUBLAS': <TacticSource.CUBLAS: 0>, 'CUBLAS_LT': <TacticSource.CUBLAS_LT: 1>, 'CUDNN': <TacticSource.CUDNN: 2>, 'EDGE_MASK_CONVOLUTIONS': <TacticSource.EDGE_MASK_CONVOLUTIONS: 3>, 'JIT_CONVOLUTIONS': <TacticSource.JIT_CONVOLUTIONS: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TempfileControlFlag:
    """

    Flags used to control TensorRT's behavior when creating executable temporary files.

    On some platforms the TensorRT runtime may need to create files in a temporary directory or use platform-specific
    APIs to create files in-memory to load temporary DLLs that implement runtime code. These flags allow the
    application to explicitly control TensorRT's use of these files. This will preclude the use of certain TensorRT
    APIs for deserializing and loading lean runtimes.

    These should be treated as bit offsets, e.g. in order to allow in-memory files for a given :class:`IRuntime`:

    .. code-block:: python

        runtime.tempfile_control_flags |= (1 << int(TempfileControlFlag.ALLOW_IN_MEMORY_FILES))



    Members:

      ALLOW_IN_MEMORY_FILES : Allow creating and loading files in-memory (or unnamed files).

      ALLOW_TEMPORARY_FILES : Allow creating and loading named files in a temporary directory on the filesystem.
    """
    ALLOW_IN_MEMORY_FILES: typing.ClassVar[TempfileControlFlag]  # value = <TempfileControlFlag.ALLOW_IN_MEMORY_FILES: 0>
    ALLOW_TEMPORARY_FILES: typing.ClassVar[TempfileControlFlag]  # value = <TempfileControlFlag.ALLOW_TEMPORARY_FILES: 1>
    __members__: typing.ClassVar[dict[str, TempfileControlFlag]]  # value = {'ALLOW_IN_MEMORY_FILES': <TempfileControlFlag.ALLOW_IN_MEMORY_FILES: 0>, 'ALLOW_TEMPORARY_FILES': <TempfileControlFlag.ALLOW_TEMPORARY_FILES: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TensorFormat:
    """

        Format of the input/output tensors.

        This enum is used by both plugins and network I/O tensors.

        For more information about data formats, see the topic "Data Format Description" located in the
        TensorRT Developer Guide (https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html).


    Members:

      LINEAR :
        Row major linear format.

        For a tensor with dimensions {N, C, H, W}, the W axis always has unit stride, and the stride of every other axis is at least the product of the next dimension times the next stride. the strides are the same as for a C array with dimensions [N][C][H][W].


      CHW2 :
        Two wide channel vectorized row major format.

        This format is bound to FP16 and BF16. It is only available for dimensions >= 3.

        For a tensor with dimensions {N, C, H, W}, the memory layout is equivalent to a C array with dimensions [N][(C+1)/2][H][W][2], with the tensor coordinates (n, c, h, w) mapping to array subscript [n][c/2][h][w][c%2].


      HWC8 :
        Eight channel format where C is padded to a multiple of 8.

        This format is bound to FP16 and BF16. It is only available for dimensions >= 3.

        For a tensor with dimensions {N, C, H, W}, the memory layout is equivalent to the array with dimensions [N][H][W][(C+7)/8*8], with the tensor coordinates (n, c, h, w) mapping to array subscript [n][h][w][c].


      CHW4 :
        Four wide channel vectorized row major format.
        This format is bound to INT8. It is only available for dimensions >= 3.

        For a tensor with dimensions {N, C, H, W}, the memory layout is equivalent to a C array with dimensions [N][(C+3)/4][H][W][4], with the tensor coordinates (n, c, h, w) mapping to array subscript [n][c/4][h][w][c%4].


      CHW16 :
        Sixteen wide channel vectorized row major format.

        This format is only supported by DLA and requires FP16. It is only available for dimensions >= 3.

        For a tensor with dimensions {N, C, H, W}, the memory layout is equivalent to a C array with dimensions [N][(C+15)/16][H][W][16], with the tensor coordinates (n, c, h, w) mapping to array subscript [n][c/16][h][w][c%16].


      CHW32 :
        Thirty-two wide channel vectorized row major format.

        This format is only available for dimensions >= 3.

        For a tensor with dimensions {N, C, H, W}, the memory layout is equivalent to a C array with dimensions [N][(C+31)/32][H][W][32], with the tensor coordinates (n, c, h, w) mapping to array subscript [n][c/32][h][w][c%32].


      DHWC8 :
        Eight channel format where C is padded to a multiple of 8.

        This format is bound to FP16 and BF16, and it is only available for dimensions >= 4.

        For a tensor with dimensions {N, C, D, H, W}, the memory layout is equivalent to an array with dimensions [N][D][H][W][(C+7)/8*8], with the tensor coordinates (n, c, d, h, w) mapping to array subscript [n][d][h][w][c].


      CDHW32 :
        Thirty-two wide channel vectorized row major format with 3 spatial dimensions.

        This format is bound to FP16 and INT8. It is only available for dimensions >= 4.

        For a tensor with dimensions {N, C, D, H, W}, the memory layout is equivalent to a C array with dimensions [N][(C+31)/32][D][H][W][32], with the tensor coordinates (n, d, c, h, w) mapping to array subscript [n][c/32][d][h][w][c%32].


      HWC :
        Non-vectorized channel-last format.
        This format is bound to FP32, FP16, INT8, INT64 and BF16, and is only available for dimensions >= 3.


      DLA_LINEAR :
        DLA planar format. Row major format. The stride for stepping along the H axis is rounded up to 64 bytes.

        This format is bound to FP16/Int8 and is only available for dimensions >= 3.

        For a tensor with dimensions {N, C, H, W}, the memory layout is equivalent to a C array with dimensions [N][C][H][roundUp(W, 64/elementSize)] where elementSize is 2 for FP16 and 1 for Int8, with the tensor coordinates (n, c, h, w) mapping to array subscript [n][c][h][w].


      DLA_HWC4 :
        DLA image format. channel-last format. C can only be 1, 3, 4. If C == 3 it will be rounded to 4. The stride for stepping along the H axis is rounded up to 32 bytes.

        This format is bound to FP16/Int8 and is only available for dimensions >= 3.

        For a tensor with dimensions {N, C, H, W}, with C’ is 1, 4, 4 when C is 1, 3, 4 respectively, the memory layout is equivalent to a C array with dimensions [N][H][roundUp(W, 32/C'/elementSize)][C'] where elementSize is 2 for FP16 and 1 for Int8, C' is the rounded C. The tensor coordinates (n, c, h, w) maps to array subscript [n][h][w][c].


      HWC16 :
        Sixteen channel format where C is padded to a multiple of 16. This format is bound to FP16/INT8/FP8. It is only available for dimensions >= 3.

        For a tensor with dimensions {N, C, H, W}, the memory layout is equivalent to the array with dimensions [N][H][W][(C+15)/16*16], with the tensor coordinates (n, c, h, w) mapping to array subscript [n][h][w][c].


      DHWC :
        Non-vectorized channel-last format. This format is bound to FP32.  It is only available for dimensions >= 4.
    """
    CDHW32: typing.ClassVar[TensorFormat]  # value = <TensorFormat.CDHW32: 7>
    CHW16: typing.ClassVar[TensorFormat]  # value = <TensorFormat.CHW16: 4>
    CHW2: typing.ClassVar[TensorFormat]  # value = <TensorFormat.CHW2: 1>
    CHW32: typing.ClassVar[TensorFormat]  # value = <TensorFormat.CHW32: 5>
    CHW4: typing.ClassVar[TensorFormat]  # value = <TensorFormat.CHW4: 3>
    DHWC: typing.ClassVar[TensorFormat]  # value = <TensorFormat.DHWC: 12>
    DHWC8: typing.ClassVar[TensorFormat]  # value = <TensorFormat.DHWC8: 6>
    DLA_HWC4: typing.ClassVar[TensorFormat]  # value = <TensorFormat.DLA_HWC4: 10>
    DLA_LINEAR: typing.ClassVar[TensorFormat]  # value = <TensorFormat.DLA_LINEAR: 9>
    HWC: typing.ClassVar[TensorFormat]  # value = <TensorFormat.HWC: 8>
    HWC16: typing.ClassVar[TensorFormat]  # value = <TensorFormat.HWC16: 11>
    HWC8: typing.ClassVar[TensorFormat]  # value = <TensorFormat.HWC8: 2>
    LINEAR: typing.ClassVar[TensorFormat]  # value = <TensorFormat.LINEAR: 0>
    __members__: typing.ClassVar[dict[str, TensorFormat]]  # value = {'LINEAR': <TensorFormat.LINEAR: 0>, 'CHW2': <TensorFormat.CHW2: 1>, 'HWC8': <TensorFormat.HWC8: 2>, 'CHW4': <TensorFormat.CHW4: 3>, 'CHW16': <TensorFormat.CHW16: 4>, 'CHW32': <TensorFormat.CHW32: 5>, 'DHWC8': <TensorFormat.DHWC8: 6>, 'CDHW32': <TensorFormat.CDHW32: 7>, 'HWC': <TensorFormat.HWC: 8>, 'DLA_LINEAR': <TensorFormat.DLA_LINEAR: 9>, 'DLA_HWC4': <TensorFormat.DLA_HWC4: 10>, 'HWC16': <TensorFormat.HWC16: 11>, 'DHWC': <TensorFormat.DHWC: 12>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TensorIOMode:
    """
    IO tensor modes for TensorRT.

    Members:

      NONE : Tensor is not an input or output.

      INPUT : Tensor is input to the engine.

      OUTPUT : Tensor is output to the engine.
    """
    INPUT: typing.ClassVar[TensorIOMode]  # value = <TensorIOMode.INPUT: 1>
    NONE: typing.ClassVar[TensorIOMode]  # value = <TensorIOMode.NONE: 0>
    OUTPUT: typing.ClassVar[TensorIOMode]  # value = <TensorIOMode.OUTPUT: 2>
    __members__: typing.ClassVar[dict[str, TensorIOMode]]  # value = {'NONE': <TensorIOMode.NONE: 0>, 'INPUT': <TensorIOMode.INPUT: 1>, 'OUTPUT': <TensorIOMode.OUTPUT: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TensorLocation:
    """
    The physical location of the data.

    Members:

      DEVICE : Data is stored on the device.

      HOST : Data is stored on the host.
    """
    DEVICE: typing.ClassVar[TensorLocation]  # value = <TensorLocation.DEVICE: 0>
    HOST: typing.ClassVar[TensorLocation]  # value = <TensorLocation.HOST: 1>
    __members__: typing.ClassVar[dict[str, TensorLocation]]  # value = {'DEVICE': <TensorLocation.DEVICE: 0>, 'HOST': <TensorLocation.HOST: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TensorRTPhase:
    """

        Indicates a phase of operation of TensorRT


    Members:

      BUILD

      RUNTIME
    """
    BUILD: typing.ClassVar[TensorRTPhase]  # value = <TensorRTPhase.BUILD: 0>
    RUNTIME: typing.ClassVar[TensorRTPhase]  # value = <TensorRTPhase.RUNTIME: 1>
    __members__: typing.ClassVar[dict[str, TensorRTPhase]]  # value = {'BUILD': <TensorRTPhase.BUILD: 0>, 'RUNTIME': <TensorRTPhase.RUNTIME: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TilingOptimizationLevel:
    """

        Describes the optimization level of tiling strategies. A higher level allows TensorRT to spend more time searching for better tiling strategy.


    Members:

      NONE :
        Do not apply any tiling strategy.


      FAST :
        Use a fast algorithm and heuristic based strategy. Slightly increases engine build time.


      MODERATE :
        Increase search space and use a mixed heuristic/profiling strategy. Moderately increases engine build time.


      FULL :
        Increase search space even wider. Significantly increases engine build time.
    """
    FAST: typing.ClassVar[TilingOptimizationLevel]  # value = <TilingOptimizationLevel.FAST: 1>
    FULL: typing.ClassVar[TilingOptimizationLevel]  # value = <TilingOptimizationLevel.FULL: 3>
    MODERATE: typing.ClassVar[TilingOptimizationLevel]  # value = <TilingOptimizationLevel.MODERATE: 2>
    NONE: typing.ClassVar[TilingOptimizationLevel]  # value = <TilingOptimizationLevel.NONE: 0>
    __members__: typing.ClassVar[dict[str, TilingOptimizationLevel]]  # value = {'NONE': <TilingOptimizationLevel.NONE: 0>, 'FAST': <TilingOptimizationLevel.FAST: 1>, 'MODERATE': <TilingOptimizationLevel.MODERATE: 2>, 'FULL': <TilingOptimizationLevel.FULL: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TimingCacheKey:
    """

            The key to retrieve timing cache entries.

    """
    @staticmethod
    def parse(text: str) -> TimingCacheKey:
        """
                Parse the text into a `TimingCacheKey` object.

                :arg text: The input text.

                :returns: A `TimingCacheKey` object.
        """
    def __str__(self) -> str:
        """
                Convert a `TimingCacheKey` object into text.

                :returns: A `str` object.
        """
class TimingCacheValue:
    """

            The values in the cache entry.

    """
    def __init__(self, arg0: typing.SupportsInt, arg1: typing.SupportsFloat) -> None:
        ...
    @property
    def tacticHash(self) -> int:
        ...
    @tacticHash.setter
    def tacticHash(self, arg1: typing.SupportsInt) -> None:
        ...
    @property
    def timingMSec(self) -> float:
        ...
    @timingMSec.setter
    def timingMSec(self, arg1: typing.SupportsFloat) -> None:
        ...
class TopKOperation:
    """
    The operations that may be performed by a TopK layer

    Members:

      MAX : Maximum of the elements

      MIN : Minimum of the elements
    """
    MAX: typing.ClassVar[TopKOperation]  # value = <TopKOperation.MAX: 0>
    MIN: typing.ClassVar[TopKOperation]  # value = <TopKOperation.MIN: 1>
    __members__: typing.ClassVar[dict[str, TopKOperation]]  # value = {'MAX': <TopKOperation.MAX: 0>, 'MIN': <TopKOperation.MIN: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TripLimit:
    """
    Describes kinds of trip limits.

    Members:

      COUNT : Tensor is a scalar of type :class:`int32` that contains the trip count.

      WHILE : Tensor is a scalar of type :class:`bool`. Loop terminates when its value is false.
    """
    COUNT: typing.ClassVar[TripLimit]  # value = <TripLimit.COUNT: 0>
    WHILE: typing.ClassVar[TripLimit]  # value = <TripLimit.WHILE: 1>
    __members__: typing.ClassVar[dict[str, TripLimit]]  # value = {'COUNT': <TripLimit.COUNT: 0>, 'WHILE': <TripLimit.WHILE: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class UnaryOperation:
    """
    The unary operations that may be performed by a Unary layer.

    Members:

      EXP : Exponentiation

      LOG : Log (base e)

      SQRT : Square root

      RECIP : Reciprocal

      ABS : Absolute value

      NEG : Negation

      SIN : Sine

      COS : Cosine

      TAN : Tangent

      SINH : Hyperbolic sine

      COSH : Hyperbolic cosine

      ASIN : Inverse sine

      ACOS : Inverse cosine

      ATAN : Inverse tangent

      ASINH : Inverse hyperbolic sine

      ACOSH : Inverse hyperbolic cosine

      ATANH : Inverse hyperbolic tangent

      CEIL : Ceiling

      FLOOR : Floor

      ERF : Gauss error function

      NOT : Not

      SIGN : Sign. If input > 0, output 1; if input < 0, output -1; if input == 0, output 0.

      ROUND : Round to nearest even for floating-point data type.

      ISINF : Return true if the input value equals +/- infinity for floating-point data type.

      ISNAN : Return true if the input value equals NaN for floating-point data type.
    """
    ABS: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.ABS: 4>
    ACOS: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.ACOS: 12>
    ACOSH: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.ACOSH: 15>
    ASIN: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.ASIN: 11>
    ASINH: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.ASINH: 14>
    ATAN: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.ATAN: 13>
    ATANH: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.ATANH: 16>
    CEIL: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.CEIL: 17>
    COS: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.COS: 7>
    COSH: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.COSH: 10>
    ERF: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.ERF: 19>
    EXP: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.EXP: 0>
    FLOOR: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.FLOOR: 18>
    ISINF: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.ISINF: 23>
    ISNAN: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.ISNAN: 24>
    LOG: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.LOG: 1>
    NEG: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.NEG: 5>
    NOT: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.NOT: 20>
    RECIP: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.RECIP: 3>
    ROUND: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.ROUND: 22>
    SIGN: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.SIGN: 21>
    SIN: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.SIN: 6>
    SINH: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.SINH: 9>
    SQRT: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.SQRT: 2>
    TAN: typing.ClassVar[UnaryOperation]  # value = <UnaryOperation.TAN: 8>
    __members__: typing.ClassVar[dict[str, UnaryOperation]]  # value = {'EXP': <UnaryOperation.EXP: 0>, 'LOG': <UnaryOperation.LOG: 1>, 'SQRT': <UnaryOperation.SQRT: 2>, 'RECIP': <UnaryOperation.RECIP: 3>, 'ABS': <UnaryOperation.ABS: 4>, 'NEG': <UnaryOperation.NEG: 5>, 'SIN': <UnaryOperation.SIN: 6>, 'COS': <UnaryOperation.COS: 7>, 'TAN': <UnaryOperation.TAN: 8>, 'SINH': <UnaryOperation.SINH: 9>, 'COSH': <UnaryOperation.COSH: 10>, 'ASIN': <UnaryOperation.ASIN: 11>, 'ACOS': <UnaryOperation.ACOS: 12>, 'ATAN': <UnaryOperation.ATAN: 13>, 'ASINH': <UnaryOperation.ASINH: 14>, 'ACOSH': <UnaryOperation.ACOSH: 15>, 'ATANH': <UnaryOperation.ATANH: 16>, 'CEIL': <UnaryOperation.CEIL: 17>, 'FLOOR': <UnaryOperation.FLOOR: 18>, 'ERF': <UnaryOperation.ERF: 19>, 'NOT': <UnaryOperation.NOT: 20>, 'SIGN': <UnaryOperation.SIGN: 21>, 'ROUND': <UnaryOperation.ROUND: 22>, 'ISINF': <UnaryOperation.ISINF: 23>, 'ISNAN': <UnaryOperation.ISNAN: 24>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Weights:
    """

        An array of weights used as a layer parameter.
        The weights are held by reference until the engine has been built - deep copies are not made automatically.

        :ivar dtype: :class:`DataType` The type of the weights.
        :ivar size: :class:`int` The number of weights in the array.
        :ivar nbytes: :class:`int` Total bytes consumed by the elements of the weights buffer.
    """
    @typing.overload
    def __init__(self, type: DataType = ...) -> None:
        """
            Initializes an empty (0-length) Weights object with the specified type.

            :type: A type to initialize the weights with. Default: :class:`tensorrt.float32`
        """
    @typing.overload
    def __init__(self, type: DataType, ptr: typing.SupportsInt, count: typing.SupportsInt) -> None:
        """
            Initializes a Weights object with the specified data.

            :type: A type to initialize the weights with.
            :ptr: A pointer to the data.
            :count: The number of weights.
        """
    @typing.overload
    def __init__(self, a: numpy.ndarray) -> None:
        """
            :a: A numpy array whose values to use. No deep copies are made.
        """
    def __len__(self) -> int:
        ...
    def numpy(self) -> typing.Any:
        """
            Create a numpy array using the underlying buffer of this weights object.
            The resulting array is just a view over the existing data, i.e. no deep copy is made.

            If the weights cannot be converted to NumPy (e.g. due to unsupported data type), the original weights are returned.

            :returns: The NumPy array or the original weights.
        """
    @property
    def dtype(self) -> DataType:
        ...
    @property
    def nbytes(self) -> int:
        ...
    @property
    def size(self) -> int:
        ...
class WeightsRole:
    """
    How a layer uses particular Weights. The power weights of an IScaleLayer are omitted.  Refitting those is not supported.

    Members:

      KERNEL : Kernel for :class:`IConvolutionLayer` or :class:`IDeconvolutionLayer` .

      BIAS : Bias for :class:`IConvolutionLayer` or :class:`IDeconvolutionLayer` .

      SHIFT : Shift part of :class:`IScaleLayer` .

      SCALE : Scale part of :class:`IScaleLayer` .

      CONSTANT : Weights for :class:`IConstantLayer` .

      ANY : Any other weights role.
    """
    ANY: typing.ClassVar[WeightsRole]  # value = <WeightsRole.ANY: 5>
    BIAS: typing.ClassVar[WeightsRole]  # value = <WeightsRole.BIAS: 1>
    CONSTANT: typing.ClassVar[WeightsRole]  # value = <WeightsRole.CONSTANT: 4>
    KERNEL: typing.ClassVar[WeightsRole]  # value = <WeightsRole.KERNEL: 0>
    SCALE: typing.ClassVar[WeightsRole]  # value = <WeightsRole.SCALE: 3>
    SHIFT: typing.ClassVar[WeightsRole]  # value = <WeightsRole.SHIFT: 2>
    __members__: typing.ClassVar[dict[str, WeightsRole]]  # value = {'KERNEL': <WeightsRole.KERNEL: 0>, 'BIAS': <WeightsRole.BIAS: 1>, 'SHIFT': <WeightsRole.SHIFT: 2>, 'SCALE': <WeightsRole.SCALE: 3>, 'CONSTANT': <WeightsRole.CONSTANT: 4>, 'ANY': <WeightsRole.ANY: 5>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
def get_builder_plugin_registry(arg0: ...) -> IPluginRegistry:
    """
        Return the plugin registry used for building engines for the specified runtime
    """
def get_nv_onnx_parser_version() -> int:
    """
    :returns: The Onnx Parser version
    """
def get_plugin_registry() -> ...:
    """
        Return the plugin registry for standard runtime
    """
def init_libnvinfer_plugins(logger: typing_extensions.CapsuleType, namespace: str) -> bool:
    """
        Initialize and register all the existing TensorRT plugins to the :class:`IPluginRegistry` with an optional namespace.
        The plugin library author should ensure that this function name is unique to the library.
        This function should be called once before accessing the Plugin Registry.

        :arg logger: Logger to print plugin registration information.
        :arg namespace: Namespace used to register all the plugins in this library.
    """
_plugin_registry = None
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
