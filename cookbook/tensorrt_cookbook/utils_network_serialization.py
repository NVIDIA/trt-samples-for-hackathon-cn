# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import ctypes
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import List, Set, Union

import numpy as np
import tensorrt as trt

from .utils_function import (datatype_np_to_trt, layer_dynamic_cast, layer_type_to_add_layer_method_name, layer_type_to_layer_type_name, text_to_logger_level)
from .utils_network import print_network

def get_trt_builtin_method_parameter_count(func):
    return len(re.findall(r"\(self:.+(, .+?)", func.__doc__))

class APIExcludeSet:
    common_class_set = {
        "algorithm_selector",
        "builder",
        "error_recorder",
        "gpu_allocator",
        "int8_calibrator",
        "logger",
        "progress_monitor",
    }

    # The members or methods which are not dumped directly in `dump_member()` (maybe dump in special cases).
    # Possible cases:
    # (1) is a setter method (Setter)
    # (2) is a getter with other arguments (Gatter)
    # (3) is dumped in special cases (SP)
    # (4) is included in other field
    # (5) is dumped in other part
    set1 = {
        "build_engine_with_config",  # Setter
        "build_serialized_network",  # Setter
        "build_serialized_network_to_stream",  # Setter
        "create_builder_config",  # Setter
        "create_network",  # Setter
        "create_optimization_profile",  # Setter
        "get_plugin_registry",  # TODO: Add support for plugin and remove this
        "is_network_supported",  # SP
        "reset",  # Setter
    }
    # The members or methods which are not built directly in `build_member()` (maybe built in special cases).
    # Possible cases:
    # (1) is a read-only method (Read-only)
    # (2) is set in special cases (SP)
    # (3) is a extra mark used in this tool (Extra-Mark)
    set2 = {
        "max_DLA_batch_size",  # Read-only
        "num_DLA_cores",  # Read-only
        "platform_has_fast_fp16",  # Read-only
        "platform_has_fast_int8",  # Read-only
        "platform_has_tf32",  # Read-only
    }
    builder_dump_exclude_set = common_class_set | set1
    builder_build_exclude_set = common_class_set | set1 | set2

    set1 = {
        "add_optimization_profile",  # Setter
        "can_run_on_DLA",  # Layer part
        "clear_flag",  # Setter
        "clear_quantization_flag",  # Setter
        "create_timing_cache",  # Setter
        "get_calibration_profile",  # SP
        "get_device_type",  # Layer part
        "get_flag",  # `flags` field
        "get_memory_pool_limit",  # SP
        "get_preview_feature",  # SP
        "get_quantization_flag",  # SP
        "get_timing_cache",  # TODO: add support for timing cache and remove this
        "is_device_type_set",  # Layer part
        "quantization_flags",  # `Quantization Flag` field
        "reset_device_type",  # Setter for per layer
        "reset",  # Setter
        "set_calibration_profile",  # Setter
        "set_device_type",  # Setter for per layer
        "set_flag",  # Setter
        "set_memory_pool_limit",  # Setter
        "set_preview_feature",  # Setter
        "set_quantization_flag",  # Setter
        "set_tactic_sources",  # Setter
        "set_timing_cache",  # Setter
    }
    set2 = {
        "calibration_profile",  # SP
        "default_device_type",  # SP
        "engine_capability",  # SP
        "get_tactic_sources",  # SP
        "hardware_compatibility_level",  # SP
        "memory_pool_limit",  # SP
        "num_optimization_profiles",  # Read-only
        "optimization_profile_list",  # SP
        "preview_feature",  # SP
        "profiling_verbosity",  # SP
        "quantization_flag",  # SP
        "runtime_platform",  # SP
        "tiling_optimization_level",  # SP
    }
    builder_config_dump_exclude_set = common_class_set | set1
    builder_config_build_exclude_set = common_class_set | set1 | set2
    builder_config_memory_exclude_set = {
        "DLA_GLOBAL_DRAM",
        "DLA_LOCAL_DRAM",
        "DLA_MANAGED_SRAM",
        "TACTIC_DRAM",
    }

    set1 = {
        "are_weights_marked_refittable",  # Tensor part
        "get_flag",  # SP
        "get_input",  # Gatter
        "get_layer",  # Gatter
        "get_output",  # Gatter
        "is_debug_tensor",  # Tensor part
        "mark_debug",  # Setter
        "mark_unfused_tensors_as_debug_tensors",  # Setter
        "mark_output_for_shapes",  # Setter
        "mark_output",  # Setter
        "mark_weights_refittable",  # Setter
        "remove_tensor",  # Setter
        "set_weights_name",  # Setter
        "unmark_debug",  # Setter
        "unmark_unfused_tensors_as_debug_tensors",  # Setter
        "unmark_output_for_shapes",  # Setter
        "unmark_output",  # Setter
        "unmark_weights_refittable",  # Setter
    }
    set2 = {
        "flag",  # Extra-mark
        "flags",  # SP
        "has_implicit_batch_dimension",  # Read-only
        "input_tensor_list",  # Extra-mark
        "num_inputs",  # Read-only
        "num_layers",  # Read-only
        "num_outputs",  # Read-only
        "output_tensor_list",  # Extra-mark
    }
    network_dump_exclude_set = common_class_set | set1
    network_build_exclude_set = common_class_set | set1 | set2
    network_exclude_condition = lambda self, key: (key.startswith("add_"))

    set1 = {
        "attention",  # For Attention structure
        "conditional",  # For If-Condition structure
        "loop",  # For Loop structure
        "get_input",  # Gatter
        "get_output_type",  # SP
        "get_output",  # Gatter
        "output_type_is_set",  # Tensor part
        "reset_output_type",  # Setter
        "reset_precision",  # Setter
        "set_input",  # Setter
        "set_output_type",  # Setter
    }
    set2 = {
        "algo_type",  # Extra-mark
        "bias_shape",  # Extra-mark
        "can_run_on_DLA",  # Read-only
        "get_device_type",  # Read-only
        "input_tensor_name_list",  # Extra-mark
        "is_device_type_set",  # Read-only
        "kernel_shape",  # Extra-mark
        "layer_index",  # Extra-mark
        "num_inputs",  # Read-only
        "num_outputs",  # Read-only
        "output_tensor_datatype_is_set_list",  # Extra-mark
        "output_tensor_datatype_list",  # Extra-mark
        "output_tensor_name_list",  # Extra-mark
        "power_shape",  # Extra-mark
        "precision_is_set",  # Read-only
        "precision",  # Precision part
        "scale_shape",  # Extra-mark
        "shift_shape",  # Extra-mark
        "type",  # Read-only
        "weight_name_list",  # Extra-mark
        "weights_refittable",  # Extra-mark
    }
    layer_dump_exclude_set = common_class_set | set1
    layer_build_exclude_set = common_class_set | set1 | set2

    set1 = {
        "get_dimension_name",  # SP
        "reset_dynamic_range",  # Setter
        "set_dimension_name",  # Setter
        "set_dynamic_range",  # Setter
    }
    set2 = {
        "dimension_name",  # SP
        "dtype",  # SP
        "dynamic_range",  # SP
        "is_execution_tensor",  # Read-only
        "is_execution_tensor",  # Read-only
        "is_network_input",  # Read-only
        "is_network_output",  # Read-only
        "is_shape_tensor",  # Read-only
        "location",  # SP
        "shape",  # SP
        "allowed_formats",  # SP
        "is_debug_tensor",  # Read-only
    }
    tensor_dump_exclude_set = common_class_set | set1
    tensor_build_exclude_set = common_class_set | set1 | set2

    set1 = {}
    set2 = {}

    @staticmethod
    def split_members(obj: object, exclude_set: Set[str] = set()) -> List[List[str]]:
        members = dir(obj)
        public_member = set(filter(lambda x: not x.startswith("__"), members))
        callback_member = public_member & APIExcludeSet.common_class_set
        callable_member = set(filter(lambda x: callable(getattr(obj, x)), public_member - APIExcludeSet.common_class_set - exclude_set))
        attribution_member = public_member - callback_member - callable_member - exclude_set
        return sorted(list(callback_member)), sorted(list(callable_member)), sorted(list(attribution_member))

class NetworkSerialization:

    # Public functions =================================================================================================
    def __init__(self, json_file: Path = None, para_file: Path = None):
        self.json_file = json_file if json_file is not None else Path("network.json")
        self.para_file = para_file if para_file is not None else Path("network.npz")

        self.api_exclude_set = APIExcludeSet()
        self.big_json = OrderedDict()
        self.weights = OrderedDict()
        self.use_patch_80 = True  # An ugly patch to deal with unexpected value in some layers, hope to remove this in the future

    def serialize(
        self,
        *,
        logger: trt.ILogger = None,
        builder: trt.Builder = None,
        builder_config: trt.IBuilderConfig = None,
        network: trt.INetworkDefinition = None,
        optimization_profile_list: list[trt.IOptimizationProfile] = [],  # TODO: remove this parameter if we can get it from BuilderConfig
        print_network_before_return: bool = False,
    ) -> bool:
        assert logger is not None
        assert builder is not None
        assert builder_config is not None
        assert network is not None

        self.logger = logger
        self.builder = builder
        self.builder_config = builder_config
        self.network = network
        self.optimization_profile_list = optimization_profile_list

        self.dump_builder()
        self.dump_builder_config()
        self.dump_network()
        self.dump_layers()

        with open(self.json_file, "w") as f:
            f.write(json.dumps(self.big_json))

        np.savez(self.para_file, **self.weights)

        if print_network_before_return:
            print_network(self.network)

        return True

    def deserialize(
        self,
        *,
        logger: trt.Logger = None,  # Pass a `trt.Logger` from outside, or we will create one inside.
        logger_level: trt.Logger = None,  # Create a `trt.Logger` inside, but using a customized log level.
        plugin_file_list: list = [],  # If we already have some plugins, just load them.
        callback_object_dict: OrderedDict = OrderedDict(),
        print_network_before_return: bool = False,
    ) -> bool:
        # Copy from `class TRTWrapperV1`
        if logger is None:
            self.logger = trt.Logger(trt.Logger.Severity.VERBOSE if logger_level is None else logger_level)
        else:
            self.logger = logger
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        for plugin_file in plugin_file_list:
            if plugin_file.exists():
                ctypes.cdll.LoadLibrary(plugin_file)

        assert self.json_file.exists()
        with open(self.json_file, "r") as f:
            self.big_json = json.loads(f.read())

        if self.para_file.exists():
            self.weights = np.load(self.para_file)
        else:
            self.log("INFO", f"Failed finding weight file {str(self.json_file)}, use random weight")
            np.random.seed(31193)
            self.weights = None

        self.callback_object_dict = callback_object_dict

        self.build_builder()
        self.build_builder_config()
        self.build_network()
        self.build_layers()
        self.build_profile()

        if print_network_before_return:
            print_network(self.network)

        return True

    # Common tool functions ============================================================================================
    def log(self, level, text) -> None:
        logger_level = text_to_logger_level(level)
        self.logger.log(logger_level, f"[NS] " + text)
        return

    # Serialization tool functions =====================================================================================
    def dump_member(self, obj: object = None, exclude_set: list = [], exclude_condition=(lambda x: False)) -> Union[OrderedDict, [OrderedDict, OrderedDict]]:
        if obj is None:
            self.log("ERROR", f"{str(obj)} is None")
            return OrderedDict()

        obj_dict = OrderedDict()
        if isinstance(obj, trt.ILayer):
            obj_dict["weight_name_list"] = []

        for key in dir(obj):
            if key.startswith("__") or key in exclude_set or exclude_condition(key):
                continue
            value = getattr(obj, key)
            if callable(value):
                if get_trt_builtin_method_parameter_count(value) == 0:
                    obj_dict[key] = value()
                else:
                    self.log("ERROR", f"Skip {str(obj).split(' ')[0][19:]}.{key}")
            elif isinstance(value, (int, float, str, list, tuple)):
                obj_dict[key] = value
            elif isinstance(value, (tuple)):
                obj_dict[key] = list(value)
            elif isinstance(value, (trt.Dims, trt.Permutation)):
                if np.array(value).shape == ():  # Special case, shape of 0 dimension
                    obj_dict[key] = [int(str(value)[1:-1])]  # Convert to string and remove brackets
                else:
                    obj_dict[key] = list(value)
            elif isinstance(value, (np.ndarray)):  # Weights
                obj_dict["weight_name_list"].append(key)
                obj_dict[key] = value
            elif isinstance(value, (trt.Weights)):  # TODO: fix this, we can not get value from `trt.Weights`
                obj_dict["weight_name_list"].append(key)
                obj_dict[key] = np.array(value.numpy())
            elif isinstance(value, (trt.TripLimit)):  # Loop structure
                obj_dict[key] = int(value)
            elif type(value.__class__).__name__ == "pybind11_type":
                obj_dict[key] = int(value)
            elif value is None:
                obj_dict[key] = None
            else:
                self.log("ERROR", f"Error parsing {str(obj).split(' ')[0][19:]}.{key}")

        return obj_dict

    def dump_builder(self) -> None:
        self.big_json["builder"] = self.dump_member(self.builder, self.api_exclude_set.builder_dump_exclude_set)
        self.big_json["builder"]["is_network_supported"] = self.builder.is_network_supported(self.network, self.builder_config)
        return

    def dump_builder_config(self) -> None:
        builder_config_dict = self.dump_member(self.builder_config, self.api_exclude_set.builder_config_dump_exclude_set)

        # Memory / Preview Feature / Quantization flag
        # TODO: use try-except for the inner for-loop, in case that the members are removed in the future
        feature_name_list = ["MemoryPoolType", "PreviewFeature", "QuantizationFlag"]
        method_name_list = ["memory_pool_limit", "preview_feature", "quantization_flag"]
        for feature_name, method_name in zip(feature_name_list, method_name_list):
            obj_dict = OrderedDict()
            for key, value in getattr(trt, feature_name).__members__.items():  # Save enumerate names as string rather than integer
                obj_dict[key] = getattr(self.builder_config, "get_" + method_name)(value)
            builder_config_dict[method_name] = obj_dict
        """ # e.g., Memory part in code unrolled:
        obj_dict = OrderedDict()
        for key, value in trt.MemoryPoolType.__members__.items():
            obj_dict[key] = self.builder_config.get_memory_pool_limit(value)
        builder_config_dict["memory_pool_limit"] = obj_dict
        """

        # Optimization Profile
        all_op_dump = []  # List of all Optimization Profile
        if self.builder_config.num_optimization_profiles > 0:
            assert len(self.optimization_profile_list) == self.builder_config.num_optimization_profiles
            for op in self.optimization_profile_list:
                op_dict = {}  # Map of one Optimization Profile
                for j in range(self.network.num_inputs):
                    tensor = self.network.get_input(j)
                    tensor_name = tensor.name
                    shape_list = op.get_shape_input(tensor_name) if tensor.is_shape_tensor else op.get_shape(tensor_name)
                    op_dict[tensor_name] = OrderedDict()
                    if len(shape_list) == 0:
                        self.log("WARNING", f"No Optimization Profile for input tensor: {tensor_name}")
                    else:
                        op_dict[tensor_name]["is_shape_tensor"] = tensor.is_shape_tensor
                        op_dict[tensor_name]["min"], op_dict[tensor_name]["opt"], op_dict[tensor_name]["max"] = [tuple(shape) for shape in shape_list]
                all_op_dump.append(op_dict)
        builder_config_dict["optimization_profile_list"] = all_op_dump

        # Int8 Calibrator Profile - deprecated
        op_dict = {}  # Map of one calibration Profile
        calibration_op = self.builder_config.get_calibration_profile()
        if calibration_op is not None:
            for j in range(self.network.num_inputs):
                tensor_name = self.network.get_input(j).name
                shape_list = calibration_op.get_shape(tensor_name)
                op_dict[tensor_name] = OrderedDict()
                if len(shape_list) == 0:
                    self.log("ERROR", f"No calibration Profile for input tensor {tensor_name}")
                else:
                    op_dict[tensor_name]["is_shape_tensor"] = tensor.is_shape_tensor
                    op_dict[tensor_name]["min"], op_dict[tensor_name]["opt"], op_dict[tensor_name]["max"] = [tuple(shape) for shape in shape_list]
        builder_config_dict["calibration_profile"] = op_dict

        self.big_json["builder_config"] = builder_config_dict
        return

    def dump_tensor(self, tensor: trt.ITensor) -> OrderedDict:
        tensor_dict = self.dump_member(tensor, self.api_exclude_set.tensor_dump_exclude_set)
        tensor_dict["dimension_name"] = [tensor.get_dimension_name(i) for i in range(len(tensor.shape))]
        tensor_dict["is_debug_tensor"] = self.network.is_debug_tensor(tensor)
        return tensor_dict

    def dump_network(self) -> None:
        network_dict = self.dump_member(self.network, self.api_exclude_set.network_dump_exclude_set, self.api_exclude_set.network_exclude_condition)

        # Flag
        obj_dict = OrderedDict()
        for key, value in trt.NetworkDefinitionCreationFlag.__members__.items():
            obj_dict[key] = self.network.get_flag(value)
        network_dict["flag"] = obj_dict

        # I/O tensors
        obj_dict = []
        for i in range(self.network.num_inputs):
            tensor = self.network.get_input(i)
            obj_dict.append(self.dump_tensor(tensor))
        network_dict["input_tensor_list"] = obj_dict

        obj_dict = []
        for i in range(self.network.num_outputs):
            tensor = self.network.get_output(i)
            obj_dict.append(self.dump_tensor(tensor))
        network_dict["output_tensor_list"] = obj_dict

        self.big_json["network"] = network_dict
        return

    def dump_layers(self):
        self.big_json["layer"] = []
        self.big_json["tensor"] = {}  # Map: tensor name -> tensor dump
        self.big_json["loop"] = {}  # Map: loop name -> map of loop members
        self.big_json["if"] = {}  # Map: if name -> map of if members
        self.big_json["attention"] = {}  # Map: attention name -> map of attention members

        for i in range(self.network.num_layers):
            layer = self.network.get_layer(i)

            layer_type_from_base_class = layer.type  # `type` is overridden in Activation / Pooling Layer, so we need to save it before dynamic cast
            layer_dynamic_cast(layer)  # Dynamic cast to real layer type
            layer_dict = self.dump_member(layer, self.api_exclude_set.layer_dump_exclude_set)
            layer_dict["layer_index"] = i  # Extra-mark

            # Methods from BuilderConfig
            layer_dict["can_run_on_DLA"] = self.builder_config.can_run_on_DLA(layer)
            layer_dict["get_device_type"] = int(self.builder_config.get_device_type(layer))
            layer_dict["is_device_type_set"] = self.builder_config.is_device_type_set(layer)

            # Weights
            if len(layer_dict["weight_name_list"]) > 0:
                for weight_name in layer_dict["weight_name_list"]:
                    self.weights[layer.name + "-" + weight_name] = layer_dict.pop(weight_name)

            # Input / output tensors
            input_tensor_name_list = []
            for j in range(layer.num_inputs):
                tensor = layer.get_input(j)
                if (layer.type == trt.LayerType.FILL and j == 0 and tensor is None) or \
                    (layer.type == trt.LayerType.SLICE and tensor is None):
                    # Input tensor can be None if:
                    # 1. Fill layer, linspace fill mode, input tensor 0 could be None
                    # 2. Slice layer, start / shape / stride / fill_value / axes tensors can be None
                    input_tensor_name_list.append(None)
                else:
                    input_tensor_name_list.append(tensor.name)
                    self.big_json["tensor"].setdefault(tensor.name, self.dump_tensor(tensor))  # Add to "tenosr" field if it does not exist
            layer_dict["input_tensor_name_list"] = input_tensor_name_list

            output_tensor_name_list = []
            output_tensor_datatype_list = []
            output_tensor_datatype_is_set_list = []
            for j in range(layer.num_outputs):
                tensor = layer.get_output(j)
                output_tensor_name_list.append(tensor.name)
                output_tensor_datatype_list.append(int(layer.get_output_type(j)))
                output_tensor_datatype_is_set_list.append(int(layer.output_type_is_set(j)))
                self.big_json["tensor"].setdefault(tensor.name, self.dump_tensor(tensor))  # Add to "tenosr" field if it does not exist
            layer_dict["output_tensor_name_list"] = output_tensor_name_list
            layer_dict["output_tensor_datatype_list"] = output_tensor_datatype_list
            layer_dict["output_tensor_datatype_is_set_list"] = output_tensor_datatype_is_set_list

            if isinstance(layer, (trt.IActivationLayer, trt.IPoolingLayer)):
                layer_dict["algo_type"] = layer_dict.pop("type")
                layer_dict["type"] = int(layer_type_from_base_class)

            elif isinstance(layer, (trt.IConvolutionLayer, trt.IDeconvolutionLayer)):
                layer_dict["kernel_shape"] = [layer.num_output_maps, layer.get_input(0).shape[1], *list(layer.kernel_size_nd)]
                layer_dict["bias_shape"] = list(layer.bias.shape)
                layer_dict["kernel_refittable"] = self.network.are_weights_marked_refittable(layer.name)
                layer_dict["bias_refittable"] = self.network.are_weights_marked_refittable(layer.name)

            elif isinstance(layer, trt.IScaleLayer):
                layer_dict["shift_shape"] = layer.shift.shape
                layer_dict["scale_shape"] = layer.scale.shape
                layer_dict["power_shape"] = layer.power.shape

            elif isinstance(layer, trt.IShuffleLayer):
                if self.use_patch_80:
                    try:
                        _ = len(layer.reshape_dims)
                    except ValueError:
                        layer_dict["reshape_dims"] = ()

            elif isinstance(layer, trt.IConstantLayer):
                layer_dict["weights_refittable"] = self.network.are_weights_marked_refittable(layer.name)

            elif isinstance(layer, trt.ISliceLayer):
                layer_dict["is_fill"] = (layer.mode == trt.SampleMode.FILL and layer.get_input(4) is not None)
                if self.use_patch_80:
                    axes_dump = ast.literal_eval(str(layer.axes))
                    if isinstance(axes_dump, int) and axes_dump > 8:
                        layer_dict["axes"] = None
                    try:
                        _ = len(layer.start)
                    except ValueError:
                        layer_dict["start"] = ()
                    try:
                        _ = len(layer.shape)
                    except ValueError:
                        layer_dict["shape"] = ()
                    try:
                        _ = len(layer.stride)
                    except ValueError:
                        layer_dict["stride"] = ()

            elif isinstance(layer, trt.IResizeLayer):
                is_dynamic_resize = (layer.num_inputs == 2)
                layer_dict["is_dynamic_resize"] = is_dynamic_resize
                layer_dict["is_static_scale_mode"] = (not is_dynamic_resize and len(layer.scales) > 0)

            elif isinstance(layer, trt.IFillLayer):
                is_dynamic_fill = layer.get_input(0) is not None
                layer_dict["is_dynamic_fill"] = is_dynamic_fill
                if is_dynamic_fill:  # The shape of output tensor is determined by input tenor 0 in dynamic fill mode
                    layer_dict["shape"] = []  # Just a place-holder

            elif isinstance(layer, (trt.ITripLimitLayer, trt.IRecurrenceLayer, trt.IIteratorLayer, trt.ILoopOutputLayer)):
                # Search `loop_name` every time since the appearance order of layers in loop is uncertain
                loop_name = layer.loop.name
                if loop_name not in self.big_json["loop"]:
                    d = OrderedDict()
                    d["iterator_layer_name_list"] = []  # could be more than one
                    d["loop_output_layer_name_list"] = []  # could be more than one
                    d["recurrence_layer_name_list"] = []  # could be more than one
                    d["trip_limit_layer_name"] = None  # could only be one
                    self.big_json["loop"][loop_name] = d
                key = {trt.LayerType.TRIP_LIMIT: "trip_limit_layer_name", trt.LayerType.RECURRENCE: "recurrence_layer_name_list", trt.LayerType.ITERATOR: "iterator_layer_name_list", trt.LayerType.LOOP_OUTPUT: "loop_output_layer_name_list"}.get(layer.type)
                if layer.type == trt.LayerType.TRIP_LIMIT:
                    self.big_json["loop"][loop_name][key] = layer.name
                else:
                    self.big_json["loop"][loop_name][key].append(layer.name)

            elif isinstance(layer, (trt.IConditionLayer, trt.IIfConditionalInputLayer, trt.IIfConditionalOutputLayer)):
                # Search `if_name` every time since the appearance order of layers in if condition is uncertain
                if_name = layer.conditional.name
                if if_name not in self.big_json["if"]:
                    d = OrderedDict()
                    d["condition_layer"] = None
                    d["condition_input_layer"] = None
                    d["condition_output_layer"] = None
                    self.big_json["if"][if_name] = d
                key = {trt.LayerType.CONDITION: "condition_layer", trt.LayerType.CONDITIONAL_INPUT: "condition_input_layer", trt.LayerType.CONDITIONAL_OUTPUT: "condition_output_layer"}.get(layer.type)
                self.big_json["if"][if_name][key] = layer.name

            elif isinstance(layer, (trt.IAttentionInputLayer, trt.IAttentionOutputLayer)):
                attention_name = layer.attention.name
                if attention_name not in self.big_json["attention"]:
                    d = OrderedDict()
                    d["attention_input_layer"] = None
                    d["attention_output_layer"] = None
                    self.big_json["attention"][attention_name] = d
                key = {trt.LayerType.ATTENTION_INPUT: "attention_input_layer", trt.LayerType.ATTENTION_OUTPUT: "attention_output_layer"}.get(layer.type)
                self.big_json["attention"][attention_name][key] = layer.name

            self.big_json["layer"].append(layer_dict)

        return

    # Deserialization tool functions ===================================================================================
    def build_member(self, obj: object = None, obj_dict: OrderedDict = OrderedDict(), exclude_set: list = [], exclude_condition=(lambda x: False)) -> None:
        if obj is None:
            self.log("ERROR", f"{str(obj)} is None")
            return

        for key, value in obj_dict.items():
            if key.startswith("__") or key in exclude_set or exclude_condition(key):  # TODO: should we add "or value is None" here?
                continue
            try:
                setattr(obj, key, value)
                self.log("VERBOSE", f"Build {str(obj).split(' ')[0][19:]}.{key} = {value}")
            except TypeError:
                self.log("WARNING", f"Skip {str(obj).split(' ')[0][19:]}.{key}")
            except AttributeError:
                self.log("ERROR", f"Error parsing {str(obj).split(' ')[0][19:]}.{key}")

        return

    def build_builder(self) -> None:
        self.builder = trt.Builder(self.logger)
        self.build_member(self.builder, self.big_json["builder"], self.api_exclude_set.builder_build_exclude_set)

        if "error_recorder" in self.callback_object_dict:
            self.builder.error_recorder = self.callback_object_dict["error_recorder"]

        return

    def build_builder_config(self) -> None:
        build_config_dump = self.big_json["builder_config"]
        self.builder_config = self.builder.create_builder_config()
        self.build_member(self.builder_config, build_config_dump, self.api_exclude_set.builder_config_build_exclude_set)

        # Default Device Type
        self.builder_config.default_device_type = trt.DeviceType(build_config_dump["default_device_type"])

        # Tactic Sources
        self.builder_config.set_tactic_sources(build_config_dump["get_tactic_sources"])

        # Engine Capability / Hardware Compatilility Level / Profiling Verbosity / Runtime Platform / Tiling Optimization Level
        # Setter name: `<method_name>`
        feature_name_list = ["EngineCapability", "HardwareCompatibilityLevel", "ProfilingVerbosity", "RuntimePlatform", "TilingOptimizationLevel"]
        method_name_list = ["engine_capability", "hardware_compatibility_level", "profiling_verbosity", "runtime_platform", "tiling_optimization_level"]
        for feature_name, method_name in zip(feature_name_list, method_name_list):
            setattr(self.builder_config, method_name, getattr(trt, feature_name)(build_config_dump[method_name]))
        """ # # For example, Engine Capability in code unrolled:
        self.builder_config.engine_capability = trt.EngineCapability(build_config_dump["engine_capability"])
        """
        # Quantization Flag (override member `quantization_flags`)
        # wili: why this API does not align with Memory / Preview Feature?
        for key, value in trt.QuantizationFlag.__members__.items():
            if build_config_dump["quantization_flag"][key]:
                self.builder_config.set_quantization_flag(trt.QuantizationFlag(key))

        # Memory / Preview Feature
        # Setter name: `set_<method_name>`
        feature_name_list = ["MemoryPoolType", "PreviewFeature"]
        method_name_list = ["memory_pool_limit", "preview_feature"]
        for feature_name, method_name in zip(feature_name_list, method_name_list):
            for key, value in getattr(trt, feature_name).__members__.items():
                if feature_name == "MemoryPoolType" and key in self.api_exclude_set.builder_config_memory_exclude_set:  # Remove DLA related memory content
                    continue
                getattr(self.builder_config, "set_" + method_name)(getattr(getattr(trt, feature_name), key), build_config_dump[method_name][key])
        """ # For example, Memory part in code unrolled:
        for key, value in trt.MemoryPoolType.__members__.items():
            if key in self.api_exclude_set.builder_config_memory_exclude_set:  # Remove DLA related content
                continue
            self.builder_config.set_memory_pool_limit(getattr(trt.MemoryPoolType, key), build_config_dump["memory_pool_limit"][key])
        """

        if "algorithm_selector" in self.callback_object_dict:
            self.builder_config.algorithm_selector = self.callback_object_dict["algorithm_selector"]
        if "int8_calibrator" in self.callback_object_dict:
            self.builder_config.int8_calibrator = self.callback_object_dict["int8_calibrator"]
        if "progress_monitor" in self.callback_object_dict:
            self.builder_config.progress_monitor = self.callback_object_dict["progress_monitor"]

        return

    def build_network(self) -> None:
        network_dict = self.big_json["network"]
        # Using `flag` rather than `flags`
        #flag = network_dict["flags"]
        flag = 0
        for key, value in trt.NetworkDefinitionCreationFlag.__members__.items():
            if network_dict["flag"][key]:
                flag |= 1 << int(getattr(trt.NetworkDefinitionCreationFlag, key))
        self.network = self.builder.create_network(flag)
        self.build_member(self.network, network_dict, self.api_exclude_set.network_build_exclude_set)

        return

    def build_tensor(self, tensor, tensor_dict) -> OrderedDict:
        self.build_member(tensor, tensor_dict, self.api_exclude_set.tensor_build_exclude_set)
        self.tensor_map[tensor.name] = tensor

        # Data Type
        if tensor.is_network_input or tensor_dict["is_network_output"]:  # Do not use `tensor.is_network_input` since no tensor is marked as output till now
            if trt.DataType(tensor_dict["dtype"]) != trt.DataType.FP4:  # Skip output FP4 since numpy can not deal with this. TODO: remove this constrain
                tensor.dtype = trt.DataType(tensor_dict["dtype"])  # No effect for intermediate tensors

        # Dimension Name
        if tensor.is_network_input:
            for i in range(len(tensor_dict["shape"])):
                tensor.set_dimension_name(i, tensor_dict["dimension_name"][i])

        # Dynamic Range
        if tensor_dict["dynamic_range"] is not None:
            tensor.dynamic_range = tensor_dict["dynamic_range"]

        # Location
        tensor.location = trt.TensorLocation(tensor_dict["location"])

        # Allowed Format
        # Do not set it if there is no constrain on tensor, or error like below will be raised:
        # (x: has dataType Float unsupported by tensor's allowed TensorFormats.)
        if tensor_dict["allowed_formats"] != 2 ** len(trt.TensorFormat.__members__) - 1:  # All 1 bit mask
            bit_mask = tensor_dict["allowed_formats"]
            if tensor.dtype == trt.DataType.FLOAT:
                bit_mask &= 1 << int(trt.TensorFormat.LINEAR) | 1 << int(trt.TensorFormat.CHW32) | 1 << int(trt.TensorFormat.HWC)
            elif tensor.dtype in [trt.DataType.HALF, trt.DataType.BF16]:  # TODO: check BF16
                bit_mask &= \
                    1 << int(trt.TensorFormat.LINEAR) | 1 << int(trt.TensorFormat.CHW2) | 1 << int(trt.TensorFormat.HWC8) | \
                    1 << int(trt.TensorFormat.CHW4) | 1 << int(trt.TensorFormat.CHW16) | 1 << int(trt.TensorFormat.CHW32) | \
                    1 << int(trt.TensorFormat.DHWC8) | 1 << int(trt.TensorFormat.CDHW32) | 1 << int(trt.TensorFormat.HWC16)
            elif tensor.dtype == trt.DataType.INT32:
                bit_mask &= 1 << int(trt.TensorFormat.LINEAR) | 1 << int(trt.TensorFormat.CHW32)
            elif tensor.dtype in [trt.DataType.INT8, trt.DataType.UINT8]:  # TODO: check UINT8
                bit_mask &= 1 << int(trt.TensorFormat.LINEAR) | 1 << int(trt.TensorFormat.CHW4) | 1 << int(trt.TensorFormat.CHW32) | 1 << int(trt.TensorFormat.CDHW32)
            elif tensor.dtype == trt.DataType.BOOL:
                bit_mask &= 1 << int(trt.TensorFormat.LINEAR)
            else:  # TODO: check kFP8, kINT64, kINT4, kFP4
                message = f"Allowed format {tensor_dict['allowed_formats']} of tensor {tensor.name} with data type {trt.DataType(tensor.dtype)} is not supported"
                self.log("ERROR", message)

            tensor.allowed_formats = bit_mask

        return

    def add_set_input_tensor(self, layer_index, input_tensor_name_list, index):
        assert len(input_tensor_name_list) >= index + 1
        tensor_name = input_tensor_name_list[index]
        if tensor_name in self.tensor_map:
            self.set_input_tensor_map[index] = self.tensor_map[tensor_name]
        else:
            self.later_layer_map[layer_index] = [tensor_name, index]
        return

    def update_loop(self, layer_dict, argument_list, attribution_map):
        layer_name = layer_dict["name"]
        kind_name = {trt.LayerType.TRIP_LIMIT: "trip_limit_layer_name", trt.LayerType.RECURRENCE: "recurrence_layer_name_list", trt.LayerType.ITERATOR: "iterator_layer_name_list", trt.LayerType.LOOP_OUTPUT: "loop_output_layer_name_list"}.get(trt.LayerType(layer_dict["type"]))
        loop_name = None
        for key, value in self.loop_map.items():
            if kind_name == "trip_limit_layer_name" and layer_name == value[kind_name] or layer_name in value[kind_name]:
                loop_name = key
        if loop_name is None:
            for key, value in self.big_json["loop"].items():
                if kind_name == "trip_limit_layer_name" and layer_name == value[kind_name] or layer_name in value[kind_name]:
                    self.loop_map[key] = {}
                    self.loop_map[key]["trip_limit_layer_name"] = layer_name if kind_name == "trip_limit_layer_name" else value["trip_limit_layer_name"]
                    self.loop_map[key]["recurrence_layer_name_list"] = [layer_name] if kind_name == "recurrence_layer_name_list" else value["recurrence_layer_name_list"]
                    self.loop_map[key]["iterator_layer_name_list"] = [layer_name] if kind_name == "iterator_layer_name_list" else value["iterator_layer_name_list"]
                    self.loop_map[key]["loop_output_layer_name_list"] = [layer_name] if kind_name == "loop_output_layer_name_list" else value["loop_output_layer_name_list"]
                    self.loop_map[key]["recurrence_layer_list"] = []  # Save layer object
                    self.loop_map[key]["loop_output_layer_list"] = []  # Save layer object
                    self.loop_map[key]["loop"] = self.network.add_loop()  # Save loop object
                    loop_name = key
                    break
        loop = self.loop_map[loop_name]["loop"]
        if kind_name == "trip_limit_layer_name":
            layer = loop.add_trip_limit(self.tensor_map[layer_dict["input_tensor_name_list"][0]], trt.TripLimit(layer_dict["kind"]))
            attribution_map["kind"] = None
        elif kind_name == "recurrence_layer_name_list":
            layer = loop.add_recurrence(self.tensor_map[layer_dict["input_tensor_name_list"][0]])
            if layer_dict["input_tensor_name_list"][1] in self.tensor_map:
                layer.set_input(1, self.tensor_map[layer_dict["input_tensor_name_list"][1]])
            self.loop_map[loop_name]["recurrence_layer_list"].append(layer)
        elif kind_name == "iterator_layer_name_list":
            layer = loop.add_iterator(self.tensor_map[layer_dict["input_tensor_name_list"][0]], layer_dict["axis"], layer_dict["reverse"])
            attribution_map["reverse"] = None
        elif kind_name == "loop_output_layer_name_list":
            layer = loop.add_loop_output(self.tensor_map[layer_dict["input_tensor_name_list"][0]], trt.LoopOutput(layer_dict["kind"]), layer_dict["axis"])
            self.loop_map[loop_name]["loop_output_layer_list"].append(layer)
            attribution_map["kind"] = None
        else:
            self.log("ERROR", f"Error loop layer name: {kind_name}")

        return layer

    def update_if(self, layer_dict, argument_list):
        layer_name = layer_dict["name"]
        kind_name = {trt.LayerType.CONDITION: "condition_layer", trt.LayerType.CONDITIONAL_INPUT: "condition_input_layer", trt.LayerType.CONDITIONAL_OUTPUT: "condition_output_layer"}.get(trt.LayerType(layer_dict["type"]))
        if_name = None
        for key, value in self.if_map.items():
            if layer_name == value[kind_name]:
                if_name = key
                break
        if if_name is None:
            for key, value in self.big_json["if"].items():
                if layer_name == value[kind_name]:
                    self.if_map[key] = {}
                    self.if_map[key]["condition_layer"] = layer_name if kind_name == "condition_layer" else value["condition_layer"]
                    self.if_map[key]["condition_input_layer"] = layer_name if kind_name == "condition_input_layer" else value["condition_input_layer"]
                    self.if_map[key]["condition_output_layer"] = layer_name if kind_name == "condition_output_layer" else value["condition_output_layer"]
                    self.if_map[key]["IfCondition"] = self.network.add_if_conditional()  # Save if object
                    if_name = key
                    break
        loop = self.if_map[if_name]["IfCondition"]
        if kind_name == "condition_layer":
            layer = loop.set_condition(self.tensor_map[layer_dict["input_tensor_name_list"][0]])
        elif kind_name == "condition_input_layer":
            layer = loop.add_input(self.tensor_map[layer_dict["input_tensor_name_list"][0]])
        elif kind_name == "condition_output_layer":
            layer = loop.add_output(self.tensor_map[layer_dict["input_tensor_name_list"][0]], self.tensor_map[layer_dict["input_tensor_name_list"][1]])
        else:
            self.log("ERROR", f"Error loop layer name: {kind_name}")

        return layer

    def update_attention(self):
        pass

    def build_layer(self, layer_dict) -> OrderedDict:
        layer_index = layer_dict["layer_index"]
        layer_type = trt.LayerType(layer_dict["type"])
        add_layer_method_name = layer_type_to_add_layer_method_name(layer_type)  # Get exact name of `add_*` API for this layer
        argument_list = []  # List of tensors / arguments used in `add_*` API
        self.set_input_tensor_map = {}  # Map: index -> tensor,  tensors used in `set_input` API
        attribution_map = {}  # Map: attribution name -> attribution value

        if len(layer_dict["input_tensor_name_list"]) > 0 and layer_dict["input_tensor_name_list"][0] is not None:  # Some layers has no input tensor
            assert layer_dict["input_tensor_name_list"][0] in self.tensor_map
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][0]])

        need_call_add_layer = True  # If-Condition or Loop structure do not need to add layer in network

        # Use `match-case` when yapf supports
        # Sorted by `int(layer_type)`, might change with TRT release in the future
        if layer_type in [trt.LayerType.CONVOLUTION, trt.LayerType.DECONVOLUTION]:  # 0, 7
            assert len(layer_dict["input_tensor_name_list"]) in [1, 2]
            argument_list.extend([layer_dict["num_output_maps"], layer_dict["kernel_size_nd"]])
            if self.weights is not None:
                kernel = self.weights[layer_dict["name"] + "-kernel"]
                bias = self.weights[layer_dict["name"] + "-bias"]
            else:
                kernel_shape = layer_dict["lKernelShape"]
                bias_shape = layer_dict["lBiasShape"]
                kernel = np.random.rand(np.prod(kernel_shape)).astype(np.float32).reshape(kernel_shape) * 2 - 1
                bias = np.random.rand(np.prod(bias_shape)).astype(np.float32).reshape(bias_shape) * 2 - 1
            argument_list.append(trt.Weights(np.ascontiguousarray(kernel)))
            argument_list.append(trt.Weights(np.ascontiguousarray(bias)))
            if np.prod(kernel.shape) == 0:  # Int8-QDQ
                assert len(layer_dict["input_tensor_name_list"]) == 2
                argument_list[-2] = trt.Weights()
                self.add_set_input_tensor(layer_index, layer_dict["input_tensor_name_list"], 1)
            attribution_map["padding_mode"] = trt.PaddingMode(layer_dict["padding_mode"])

        elif layer_type == trt.LayerType.CAST:  # 1
            assert len(layer_dict["input_tensor_name_list"]) == 1
            argument_list.append(trt.DataType(layer_dict["to_type"]))
            attribution_map["to_type"] = trt.DataType(layer_dict["to_type"])

        elif layer_type == trt.LayerType.ACTIVATION:  # 2
            assert len(layer_dict["input_tensor_name_list"]) == 1
            argument_list.append(trt.ActivationType(layer_dict["algo_type"]))
            attribution_map["type"] = trt.ActivationType(layer_dict["algo_type"])

        elif layer_type == trt.LayerType.POOLING:  # 3
            assert len(layer_dict["input_tensor_name_list"]) == 1
            argument_list.append(trt.PoolingType(layer_dict["algo_type"]))
            argument_list.append(layer_dict["window_size_nd"])
            attribution_map["padding_mode"] = trt.PaddingMode(layer_dict["padding_mode"])
            attribution_map["type"] = trt.PoolingType(layer_dict["algo_type"])

        elif layer_type == trt.LayerType.LRN:  # 4
            assert len(layer_dict["input_tensor_name_list"]) == 1
            argument_list.extend([layer_dict["window_size"], layer_dict["alpha"], layer_dict["beta"], layer_dict["k"]])

        elif layer_type == trt.LayerType.SCALE:  # 5
            assert len(layer_dict["input_tensor_name_list"]) == 1
            argument_list.append(trt.ScaleMode(layer_dict["mode"]))
            if self.weights is not None:
                shift = self.weights[layer_dict["name"] + "-shift"]
                scale = self.weights[layer_dict["name"] + "-scale"]
                power = self.weights[layer_dict["name"] + "-power"]
            else:
                shift_shape = layer_dict["shift_shape"]
                scale_shape = layer_dict["scale_shape"]
                power_shape = layer_dict["power_shape"]
                shift = np.random.rand(np.prod(shift_shape)).astype(np.float32).reshape(shift_shape) * 2 - 1
                scale = np.random.rand(np.prod(scale_shape)).astype(np.float32).reshape(scale_shape) * 2 - 1
                power = np.ones(np.prod(power_shape)).astype(np.float32).reshape(power_shape)
            argument_list.append(trt.Weights(np.ascontiguousarray(shift)))
            argument_list.append(trt.Weights(np.ascontiguousarray(scale)))
            argument_list.append(trt.Weights(np.ascontiguousarray(power)))
            argument_list.append(layer_dict["channel_axis"])

        elif layer_type == trt.LayerType.SOFTMAX:  # 6
            assert len(layer_dict["input_tensor_name_list"]) == 1

        elif layer_type == trt.LayerType.CONCATENATION:  # 8
            assert len(layer_dict["input_tensor_name_list"]) > 0
            input_tensor_list = []
            for tensor_name in layer_dict["input_tensor_name_list"]:
                input_tensor_list.append(self.tensor_map[tensor_name])
            argument_list = [input_tensor_list]  # Rather than `append`

        elif layer_type == trt.LayerType.ELEMENTWISE:  # 9
            assert len(layer_dict["input_tensor_name_list"]) == 2
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][1]])
            argument_list.append(trt.ElementWiseOperation(layer_dict["op"]))
            attribution_map["op"] = trt.ElementWiseOperation(layer_dict["op"])

        elif layer_type in [trt.LayerType.PLUGIN, trt.LayerType.PLUGIN_V2, trt.LayerType.PLUGIN_V3]:  # 10, 21, 46
            self.log("ERROR", "Plugin Layer not supported")  # TODO: add support for plugin to remove this

        elif layer_type == trt.LayerType.UNARY:  # 11
            assert len(layer_dict["input_tensor_name_list"]) == 1
            argument_list.append(trt.UnaryOperation(layer_dict["op"]))
            attribution_map["op"] = trt.UnaryOperation(layer_dict["op"])

        elif layer_type == trt.LayerType.PADDING:  # 12
            assert len(layer_dict["input_tensor_name_list"]) == 1
            argument_list.extend([layer_dict["pre_padding_nd"], layer_dict["post_padding_nd"]])

        elif layer_type == trt.LayerType.SHUFFLE:  # 13
            assert len(layer_dict["input_tensor_name_list"]) in [1, 2]
            if len(layer_dict["input_tensor_name_list"]) == 2:  # Dynamic shuffle
                self.add_set_input_tensor(layer_index, layer_dict["input_tensor_name_list"], 1)
                attribution_map["reshape_dims"] = None  # Skip setting it in attribution
            if self.use_patch_80:
                if isinstance(layer_dict["reshape_dims"], list) and len(layer_dict["reshape_dims"]) == 0:
                    attribution_map["reshape_dims"] = None  # overwrite useless value

        elif layer_type == trt.LayerType.REDUCE:  # 14
            assert len(layer_dict["input_tensor_name_list"]) == 1
            argument_list.append(trt.ReduceOperation(layer_dict["op"]))
            argument_list.append(layer_dict["axes"])
            argument_list.append(layer_dict["keep_dims"])
            attribution_map["op"] = trt.ReduceOperation(layer_dict["op"])

        elif layer_type == trt.LayerType.TOPK:  # 15
            assert len(layer_dict["input_tensor_name_list"]) in [1, 2]
            argument_list.append(trt.TopKOperation(layer_dict["op"]))
            if len(layer_dict["input_tensor_name_list"]) == 2:
                self.add_set_input_tensor(layer_index, layer_dict["input_tensor_name_list"], 1)
                argument_list.append(0)  # Place-holder
            else:
                argument_list.append(layer_dict["k"])
            argument_list.append(layer_dict["axes"])
            attribution_map["op"] = None  # Do not set it in attribution

        elif layer_type == trt.LayerType.GATHER:  # 16
            assert len(layer_dict["input_tensor_name_list"]) == 2
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][1]])
            argument_list.append(trt.GatherMode(layer_dict["mode"]))
            attribution_map["axis"] = layer_dict["axis"]

        elif layer_type == trt.LayerType.MATRIX_MULTIPLY:  # 17
            assert len(layer_dict["input_tensor_name_list"]) == 2
            argument_list.append(trt.MatrixOperation(layer_dict["op0"]))
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][1]])
            argument_list.append(trt.MatrixOperation(layer_dict["op1"]))
            attribution_map["op0"] = trt.MatrixOperation(layer_dict["op0"])
            attribution_map["op1"] = trt.MatrixOperation(layer_dict["op1"])

        elif layer_type == trt.LayerType.RAGGED_SOFTMAX:  # 18
            assert len(layer_dict["input_tensor_name_list"]) == 2
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][1]])

        elif layer_type == trt.LayerType.CONSTANT:  # 19
            assert len(layer_dict["input_tensor_name_list"]) == 0
            if self.weights is not None:
                weight = self.weights[layer_dict["name"] + "-weights"]
            else:
                weight_shape = layer_dict["shape"]
                data_type = trt.nptype(trt.DataType(layer_dict["output_data_type_list"][0]))
                weight = np.random.rand(np.prod(weight_shape)).astype(data_type).reshape(weight_shape)
            if weight.shape == (0, ):  # Special process for weight of shape (0,)
                argument_list.extend([[0], trt.Weights(datatype_np_to_trt(weight.dtype))])
                #argument_list.extend([weight.shape, trt.Weights(np.ascontiguousarray(weight))])
            else:
                argument_list.extend([weight.shape, trt.Weights(np.ascontiguousarray(weight))])

        elif layer_type == trt.LayerType.IDENTITY:  # 20
            assert len(layer_dict["input_tensor_name_list"]) == 1

        elif layer_type == trt.LayerType.SLICE:  # 22
            n_input_tensor = len(layer_dict["input_tensor_name_list"])
            assert n_input_tensor in [1, 2, 3, 4, 5, 6]
            argument_list.extend([layer_dict["start"], layer_dict["shape"], layer_dict["stride"]])  # Place-holders
            if n_input_tensor >= 2 and layer_dict["start"] == []:
                argument_list[1] = [0] * argument_list[0].shape[0]  # set `start` forcely in dynamic slice mode
            if n_input_tensor >= 2 and layer_dict["input_tensor_name_list"][1] is not None:
                self.add_set_input_tensor(layer_index, layer_dict["input_tensor_name_list"], 1)
                attribution_map["start"] = None  # Do not set it if using input tensor
            if n_input_tensor >= 3 and layer_dict["input_tensor_name_list"][2] is not None:
                self.add_set_input_tensor(layer_index, layer_dict["input_tensor_name_list"], 2)
                attribution_map["shape"] = None  # Do not set it if using input tensor
            if n_input_tensor >= 4 and layer_dict["input_tensor_name_list"][3] is not None:
                self.add_set_input_tensor(layer_index, layer_dict["input_tensor_name_list"], 3)
                attribution_map["stride"] = None  # Do not set it if using input tensor
            if n_input_tensor >= 5 and trt.SampleMode(layer_dict["mode"]) == trt.SampleMode.FILL and layer_dict["is_fill"] == True:
                self.add_set_input_tensor(layer_index, layer_dict["input_tensor_name_list"], 4)
            attribution_map["mode"] = trt.SampleMode(layer_dict["mode"])
            attribution_map["is_fill"] = None  # Extra-mark

        elif layer_type == trt.LayerType.SHAPE:  # 23
            assert len(layer_dict["input_tensor_name_list"]) == 1

        elif layer_type == trt.LayerType.PARAMETRIC_RELU:  # 24
            assert len(layer_dict["input_tensor_name_list"]) == 2
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][1]])

        elif layer_type == trt.LayerType.RESIZE:  # 25
            assert len(layer_dict["input_tensor_name_list"]) in [1, 2]
            if layer_dict["is_dynamic_resize"]:
                assert len(layer_dict["input_tensor_name_list"]) == 2
                self.add_set_input_tensor(layer_index, layer_dict["input_tensor_name_list"], 1)
            if layer_dict["is_static_scale_mode"]:
                attribution_map["shape"] = None  # Skip setting `shape` in static scale mode
            attribution_map["resize_mode"] = trt.InterpolationMode(layer_dict["resize_mode"])
            attribution_map["coordinate_transformation"] = trt.ResizeCoordinateTransformation(layer_dict["coordinate_transformation"])
            attribution_map["selector_for_single_pixel"] = trt.ResizeSelector(layer_dict["selector_for_single_pixel"])
            attribution_map["nearest_rounding"] = trt.ResizeRoundMode(layer_dict["nearest_rounding"])
            attribution_map["is_dynamic_resize"] = None
            attribution_map["is_static_shape_mode"] = None

        elif layer_type in [trt.LayerType.TRIP_LIMIT, trt.LayerType.RECURRENCE, trt.LayerType.ITERATOR, trt.LayerType.LOOP_OUTPUT]:  # 26, 27, 28, 29
            layer = self.update_loop(layer_dict, argument_list, attribution_map)
            need_call_add_layer = False

        elif layer_type == trt.LayerType.SELECT:  # 30
            assert len(layer_dict["input_tensor_name_list"]) == 3
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][1]])
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][2]])

        elif layer_type == trt.LayerType.FILL:  # 31
            assert len(layer_dict["input_tensor_name_list"]) in [0, 2, 3]
            if layer_dict["is_dynamic_fill"]:
                argument_list = []  # Remove the first input tensor and use `set_input` instead
                self.add_set_input_tensor(layer_index, layer_dict["input_tensor_name_list"], 0)
            argument_list.extend([trt.Dims(layer_dict["shape"]), trt.FillOperation(layer_dict["operation"]), trt.DataType(layer_dict["to_type"])])
            if len(layer_dict["input_tensor_name_list"]) == 3:
                self.add_set_input_tensor(layer_index, layer_dict["input_tensor_name_list"], 1)
                self.add_set_input_tensor(layer_index, layer_dict["input_tensor_name_list"], 2)
                attribution_map["alpha"] = None  # Do not set `alpha` and `beta` in this case, or error like below will be raised:
                attribution_map["beta"] = None
                # Skipping tactic 0x0000000000000000 due to exception Assertion dims.nbDims == 1 failed. Alpha and beta tensor should be set when output an ND tensor.
            #if "Range" in layer_dict["name"]:  # The special case: Range node from ONNX, TODO: remove this branch
            #    self.set_input_tensor_map[1] = self.constant_layers_for_Range_node[0].get_output(0)
            #    self.set_input_tensor_map[2] = self.constant_layers_for_Range_node[1].get_output(0)
            # Set some attributions as None to avoid setting them in `build_member`
            attribution_map["is_alpha_beta_int64"] = None  # Read-only atttribution
            attribution_map["is_dynamic_fill"] = None  # Extra-mark
            attribution_map["shape"] = None  # Useless after `add_fill` is called
            attribution_map["to_type"] = None  # Do not set it after `add_fill` is called

        elif layer_type in [trt.LayerType.QUANTIZE, trt.LayerType.DEQUANTIZE]:  # 32, 33
            assert len(layer_dict["input_tensor_name_list"]) in [2, 3]
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][1]])
            if len(layer_dict["input_tensor_name_list"]) == 3:
                self.add_set_input_tensor(layer_index, layer_dict["input_tensor_name_list"], 2)

        elif layer_type in [trt.LayerType.CONDITION, trt.LayerType.CONDITIONAL_INPUT, trt.LayerType.CONDITIONAL_OUTPUT]:  # 34, 35, 36
            layer = self.update_if(layer_dict, argument_list)
            need_call_add_layer = False

        elif layer_type == trt.LayerType.SCATTER:  # 37
            assert len(layer_dict["input_tensor_name_list"]) == 3
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][1]])
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][2]])
            argument_list.append(trt.ScatterMode(layer_dict["mode"]))

        elif layer_type == trt.LayerType.EINSUM:  # 38
            assert len(layer_dict["input_tensor_name_list"]) > 0
            input_tensor_list = []
            for tensor_name in layer_dict["input_tensor_name_list"]:
                input_tensor_list.append(self.tensor_map[tensor_name])
            argument_list = [input_tensor_list]  # Rather than `append`
            argument_list.append(layer_dict["equation"])

        elif layer_type == trt.LayerType.ASSERTION:  # 39
            assert len(layer_dict["input_tensor_name_list"]) == 1
            argument_list.append(layer_dict["message"])

        elif layer_type == trt.LayerType.ONE_HOT:  # 40
            assert len(layer_dict["input_tensor_name_list"]) == 3
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][1]])
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][2]])
            argument_list.append(layer_dict["axis"])

        elif layer_type == trt.LayerType.NON_ZERO:  # 41
            assert len(layer_dict["input_tensor_name_list"]) == 1

        elif layer_type == trt.LayerType.GRID_SAMPLE:  # 42
            assert len(layer_dict["input_tensor_name_list"]) == 2
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][1]])
            attribution_map["interpolation_mode"] = trt.InterpolationMode(layer_dict["interpolation_mode"])
            attribution_map["sample_mode"] = trt.SampleMode(layer_dict["sample_mode"])

        elif layer_type == trt.LayerType.NMS:  # 43
            assert len(layer_dict["input_tensor_name_list"]) == 3
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][1]])
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][2]])
            attribution_map["bounding_box_format"] = trt.BoundingBoxFormat(layer_dict["bounding_box_format"])

        elif layer_type == trt.LayerType.REVERSE_SEQUENCE:  # 44
            assert len(layer_dict["input_tensor_name_list"]) == 2
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][1]])

        elif layer_type == trt.LayerType.NORMALIZATION:  # 45
            assert len(layer_dict["input_tensor_name_list"]) == 3
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][1]])
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][2]])
            argument_list.append(layer_dict["axes"])
            attribution_map["compute_precision"] = trt.DataType(layer_dict["compute_precision"])

        elif layer_type in [trt.LayerType.SQUEEZE, trt.LayerType.UNSQUEEZE]:  # 47, 48
            assert len(layer_dict["input_tensor_name_list"]) == 2
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][1]])

        elif layer_type == trt.LayerType.CUMULATIVE:  # 49
            assert len(layer_dict["input_tensor_name_list"]) == 2
            argument_list.append(self.tensor_map[layer_dict["input_tensor_name_list"][1]])
            argument_list.append(trt.CumulativeOperation(layer_dict["op"]))
            argument_list.append(layer_dict["exclusive"])
            argument_list.append(layer_dict["reverse"])
            attribution_map["op"] = trt.CumulativeOperation(layer_dict["op"])

        elif layer_type == trt.LayerType.DYNAMIC_QUANTIZE:  # 50
            assert len(layer_dict["input_tensor_name_list"]) == 2
            argument_list.append(layer_dict["axis"])
            argument_list.append(layer_dict["block_size"])
            argument_list.append(trt.DataType(layer_dict["to_type"]))
            argument_list.append(trt.DataType(layer_dict["scale_type"]))
            self.add_set_input_tensor(layer_index, layer_dict["input_tensor_name_list"], 1)
            attribution_map["to_type"] = trt.DataType(layer_dict["to_type"])
            attribution_map["scale_type"] = trt.DataType(layer_dict["scale_type"])

        else:
            self.log("ERROR", f"Error parsing layer {layer_dict['name']}, type: {layer_type}")

        # Add the layer and set attributions
        if need_call_add_layer:
            layer = getattr(self.network, add_layer_method_name)(*argument_list)

        for index, tensor in self.set_input_tensor_map.items():
            layer.set_input(index, tensor)

        # Set attributions
        self.build_member(layer, layer_dict, self.api_exclude_set.layer_build_exclude_set | set(attribution_map.keys()))
        for key, value in attribution_map.items():
            if value is not None:
                setattr(layer, key, value)

        # Method from BuilderConfig
        self.builder_config.set_device_type(layer, trt.DeviceType(layer_dict["get_device_type"]))

        # More operations after adding the layer
        if layer_type in [trt.LayerType.QUANTIZE, trt.LayerType.DEQUANTIZE]:  # 32, 33
            if layer.axis == -1:  # TODO: check whether this is necessary
                layer.axis = 0  # Change it into 0 (per-tensor Quantization / Dequantization)

        if layer_index in self.later_layer_map:
            self.later_layer_map[layer_index].append(layer)  # Add layer object to the map

        # Build output tensors, mark debug tensors
        assert layer.num_outputs == len(layer_dict["output_tensor_name_list"])
        for i, tensor_name in enumerate(layer_dict["output_tensor_name_list"]):
            tensor_dict = self.big_json["tensor"][tensor_name]
            layer.set_output_type(i, trt.DataType(tensor_dict["dtype"]))
            self.build_tensor(layer.get_output(i), tensor_dict)
            if tensor_dict["is_debug_tensor"]:
                self.network.mark_debug(layer.get_output(i))

        return

    def build_layers(self):
        network_dict = self.big_json["network"]
        layer_dict = self.big_json["layer"]
        tensor_dict = self.big_json["tensor"]
        # Map from "tensor name" to "corresponding tensors which has been in the built network"
        self.tensor_map = OrderedDict()
        # Map from "if structure name" to "names of corresponding layers in this structure"
        self.if_map = {}
        # Map: from "loop structure name" to "names of corresponding layers in this structure"
        self.loop_map = {}
        # Map from "layer index" to "corresponding name of missing tensor in this layer"
        # In some cases, the shape tensor consumed in an early layer is produced by a later layer, so we need to mark them
        self.later_layer_map = {}

        for tensor_dict in network_dict["input_tensor_list"]:
            tensor = self.network.add_input(tensor_dict["name"], trt.DataType(tensor_dict["dtype"]), tensor_dict["shape"])
            self.tensor_map[tensor.name] = tensor
            self.build_tensor(tensor, tensor_dict)

        # Constant layer for Range Node from ONNX file
        constant_layer0_for_Range = self.network.add_constant([], trt.Weights(np.ascontiguousarray(np.array([0], dtype=np.int32))))
        constant_layer0_for_Range.name = "ConstantLayer0ForRangeNoe"
        constant_layer0_for_Range.get_output(0).name = "ConstantTensor0ForRangeNoe"
        constant_layer1_for_Range = self.network.add_constant([1], trt.Weights(np.ascontiguousarray(np.array([1], dtype=np.int32))))
        constant_layer1_for_Range.name = "ConstantLayer1ForRangeNoe"
        constant_layer1_for_Range.get_output(0).name = "ConstantTensor1ForRangeNoe"
        self.constant_layers_for_Range_node = [constant_layer0_for_Range, constant_layer1_for_Range]

        # Rebuild network layer by layer
        for i, singe_layer_dump in enumerate(layer_dict):
            message = f"{i:5d}->{layer_type_to_layer_type_name(trt.LayerType(singe_layer_dump['type'])):<15s}: {singe_layer_dump['name']}"
            self.log("VERBOSE", message)
            #for key, value in singe_layer_dump.items():  # Too much output
            #    self.log("VERBOSE", f"      {key}:{value}")
            self.build_layer(singe_layer_dump)

        # Addition scan, reassign the second input tensor of Shuffle layer
        for old_layer_index, [tensor_name, index, layer] in self.later_layer_map.items():
            if layer.type == trt.LayerType.SHUFFLE:
                assert index == 1
            elif layer.type == trt.LayerType.FILL:
                assert index in [1, 2, 3, 4]
            elif layer.type == trt.LayerType.SLICE:
                pass
            else:
                self.log("ERROR", f"Error checking missing tensor {tensor_name} in layer {old_layer_index}")
            tensor = self.tensor_map[tensor_name]  # Should be able to be found now
            layer.set_input(index, tensor)

        # Addition scan, reassign the second input tensor of recurrence layer or output layer in Loop structure
        for loop_name in self.loop_map.keys():
            for recurrence_layer_name, recurrence_layer in zip(self.loop_map[loop_name]["recurrence_layer_name_list"], self.loop_map[loop_name]["recurrence_layer_list"]):
                for layer in self.big_json["layer"]:
                    if layer["name"] == recurrence_layer_name:
                        input_tensor_name = layer["input_tensor_name_list"][1]
                        break
                input_tensor = self.tensor_map[input_tensor_name]
                recurrence_layer.set_input(1, input_tensor)

            for output_layer_name, output_layer in zip(self.loop_map[loop_name]["loop_output_layer_name_list"], self.loop_map[loop_name]["loop_output_layer_list"]):
                if output_layer.kind == trt.LoopOutput.LAST_VALUE:  # Only CONCATENTE and REVERSE mode need the second input tensor
                    continue
                for layer in self.big_json["layer"]:
                    if layer["name"] == output_layer_name:
                        input_tensor_name = layer["input_tensor_name_list"][1]
                        break
                input_tensor = self.tensor_map[input_tensor_name]
                output_layer.set_input(1, input_tensor)

        # mark output tensor
        for tensor in network_dict["output_tensor_list"]:
            output_tensor = self.tensor_map[tensor["name"]]
            assert (output_tensor.name in self.tensor_map)
            self.network.mark_output(output_tensor)

        return

    def build_profile(self) -> None:
        for op_dict in self.big_json["builder_config"]["optimization_profile_list"]:
            op = self.builder.create_optimization_profile()
            for j in range(self.network.num_inputs):
                tensor_name = self.network.get_input(j).name
                assert tensor_name in op_dict
                if op_dict[tensor_name] == {}:
                    continue
                argument_list = [tensor_name, op_dict[tensor_name]["min"], op_dict[tensor_name]["opt"], op_dict[tensor_name]["max"]]
                if op_dict[tensor_name]["is_shape_tensor"]:
                    op.set_shape_input(*argument_list)
                else:
                    op.set_shape(*argument_list)
            self.builder_config.add_optimization_profile(op)

        # Int8 Calibration Profile
        op_dict = self.big_json["builder_config"]["calibration_profile"]
        if len(op_dict) > 0:
            op = self.builder.create_optimization_profile()
            for j in range(self.network.num_inputs):
                tensor_name = self.network.get_input(j).name
                assert tensor_name in op_dict
                if op_dict[tensor_name] == {}:
                    continue
                argument_list = [tensor_name, op_dict[tensor_name]["min"], op_dict[tensor_name]["opt"], op_dict[tensor_name]["max"]]
                if op_dict[tensor_name]["is_shape_tensor"]:
                    op.set_shape_input(*argument_list)
                else:
                    op.set_shape(*argument_list)
            self.config.set_calibration_profile(op)

        return
