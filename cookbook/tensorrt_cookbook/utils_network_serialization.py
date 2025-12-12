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
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import List, Set, Union
import ctypes

import numpy as np
import tensorrt as trt

from .utils_function import (datatype_np_to_trt, layer_dynamic_cast, layer_type_to_add_layer_method_name, layer_type_to_layer_type_name, text_to_logger_level)
from .utils_network import print_network

def get_trt_builtin_method_parameter_count(func):
    return len(re.findall(r"\(self:.+(, .+?)", func.__doc__))

class APIExcludeSet:
    common_set = {
        "algorithm_selector",
        "builder",
        "error_recorder",
        "gpu_allocator",
        "int8_calibrator",
        "logger",
        "progress_monitor",
    }

    # The members or methods which are not dumped directly in `dump_member()`
    # Possible cases:
    # (1) is a setter method (Setter)
    # (2) is a getter with other arguments (Gatter)
    # (3) is dumped in special cases (SP)
    # (4) is included in other field
    # (5) is dumped in other part
    set1 = {
        "build_engine_with_config",  # Setter
        "build_serialized_network",  # Setter
        "create_builder_config",  # Setter
        "create_network",  # Setter
        "create_optimization_profile",  # Setter
        "get_plugin_registry",  # TODO: Add support for plugin and remove this
        "is_network_supported",  # SP
        "reset",  # Setter
    }
    # The members or methods which are not set directly in `build_member()`, but set in special cases.
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
    builder_dump_exclude_set = common_set | set1
    builder_build_exclude_set = common_set | set1 | set2

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
    builder_config_dump_exclude_set = common_set | set1
    builder_config_build_exclude_set = common_set | set1 | set2
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
        "mark_output_for_shapes",  # Setter
        "mark_output",  # Setter
        "mark_weights_refittable",  # Setter
        "remove_tensor",  # Setter
        "set_weights_name",  # Setter
        "unmark_debug",  # Setter
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
    network_dump_exclude_set = common_set | set1
    network_build_exclude_set = common_set | set1 | set2
    network_exclude_condition = lambda self, key: (key.startswith("add_"))

    set1 = {
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
        "algo-type",  # Extra-mark
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
    layer_dump_exclude_set = common_set | set1
    layer_build_exclude_set = common_set | set1 | set2

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
    tensor_dump_exclude_set = common_set | set1
    tensor_build_exclude_set = common_set | set1 | set2

    set1 = {}
    set2 = {}

    @staticmethod
    def split_members(obj: object, exclude_set: Set[str] = set()) -> List[List[str]]:
        members = dir(obj)
        public_member = set(filter(lambda x: not x.startswith("__"), members))
        callback_member = public_member & APIExcludeSet.common_set
        callable_member = set(filter(lambda x: callable(getattr(obj, x)), public_member - APIExcludeSet.common_set - exclude_set))
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
    ) -> None:
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

        return

    def deserialize(
        self,
        *,
        logger: trt.Logger = None,  # Pass a `trt.Logger` from outside, or we will create one inside.
        logger_level: trt.Logger = None,  # Create a `trt.Logger` inside, but using a customized log level.
        plugin_file_list: list = [],  # If we already have some plugins, just load them.
        callback_object_dict: OrderedDict = OrderedDict(),
        print_network_before_return: bool = False,
    ) -> trt.IHostMemory:
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

        return

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

        obj_dump = OrderedDict()
        if isinstance(obj, trt.ILayer):
            obj_dump["weight_name_list"] = []

        for key in dir(obj):
            if key.startswith("__") or key in exclude_set or exclude_condition(key):
                continue
            value = getattr(obj, key)
            if callable(value):
                if get_trt_builtin_method_parameter_count(value) == 0:
                    obj_dump[key] = value()
                else:
                    self.log("ERROR", f"Skip {str(obj).split(' ')[0][19:]}.{key}")
            elif isinstance(value, (int, float, str, list, tuple)):
                obj_dump[key] = value
            elif isinstance(value, (tuple)):
                obj_dump[key] = list(value)
            elif isinstance(value, (trt.Dims, trt.Permutation)):
                if np.array(value).shape == ():  # Special case, shape of 0 dimension
                    obj_dump[key] = [int(str(value)[1:-1])]  # Convert to string and remove brackets
                else:
                    obj_dump[key] = list(value)
            elif isinstance(value, (np.ndarray)):  # Weights
                obj_dump["weight_name_list"].append(key)
                obj_dump[key] = value
            elif isinstance(value, (trt.Weights)):  # TODO: fix this, we can not get value from `trt.Weights`
                obj_dump["weight_name_list"].append(key)
                obj_dump[key] = np.array(value.numpy())
            elif isinstance(value, (trt.TripLimit)):  # Loop structure
                obj_dump[key] = int(value)
            elif type(value.__class__).__name__ == "pybind11_type":
                obj_dump[key] = int(value)
            elif value is None:
                obj_dump[key] = None
            else:
                self.log("ERROR", f"Error parsing {str(obj).split(' ')[0][19:]}.{key}")

        return obj_dump

    def dump_builder(self) -> None:
        self.big_json["builder"] = self.dump_member(self.builder, self.api_exclude_set.builder_dump_exclude_set)
        self.big_json["builder"]["is_network_supported"] = self.builder.is_network_supported(self.network, self.builder_config)
        return

    def dump_builder_config(self) -> None:
        builder_config_dump = self.dump_member(self.builder_config, self.api_exclude_set.builder_config_dump_exclude_set)

        # Memory / Preview Feature / Quantization flag
        feature_name_list = ["MemoryPoolType", "PreviewFeature", "QuantizationFlag"]
        method_name_list = ["memory_pool_limit", "preview_feature", "quantization_flag"]
        for feature_name, method_name in zip(feature_name_list, method_name_list):
            dump = OrderedDict()
            for key, value in getattr(trt, feature_name).__members__.items():  # Save enumerate names as string rather than integer
                dump[key] = getattr(self.builder_config, "get_" + method_name)(value)
            builder_config_dump[method_name] = dump
        """ # e.g., Memory part in code unrolled:
        dump = OrderedDict()
        for key, value in trt.MemoryPoolType.__members__.items():
            dump[key] = self.builder_config.get_memory_pool_limit(value)
        builder_config_dump["memory_pool_limit"] = dump
        """

        # Optimization Profile
        all_op_dump = []  # List of all Optimization Profile
        if self.builder_config.num_optimization_profiles > 0:
            assert len(self.optimization_profile_list) == self.builder_config.num_optimization_profiles
            for op in self.optimization_profile_list:
                op_dump = {}  # Map of one Optimization Profile
                for j in range(self.network.num_inputs):
                    tensor = self.network.get_input(j)
                    tensor_name = tensor.name
                    shape_list = op.get_shape_input(tensor_name) if tensor.is_shape_tensor else op.get_shape(tensor_name)
                    op_dump[tensor_name] = OrderedDict()
                    if len(shape_list) == 0:
                        self.log("WARNING", f"No Optimization Profile for input tensor: {tensor_name}")
                    else:
                        op_dump[tensor_name]["is_shape_tensor"] = tensor.is_shape_tensor
                        op_dump[tensor_name]["min"], op_dump[tensor_name]["opt"], op_dump[tensor_name]["max"] = [tuple(shape) for shape in shape_list]
                all_op_dump.append(op_dump)
        builder_config_dump["optimization_profile_list"] = all_op_dump

        # Int8 Calibrator Profile [deprecated]
        op_dump = {}  # Map of one calibration Profile
        calibration_op = self.builder_config.get_calibration_profile()
        if calibration_op is not None:
            for j in range(self.network.num_inputs):
                tensor_name = self.network.get_input(j).name
                shape_list = calibration_op.get_shape(tensor_name)
                op_dump[tensor_name] = OrderedDict()
                if len(shape_list) == 0:
                    self.log("ERROR", f"No calibration Profile for input tensor {tensor_name}")
                else:
                    op_dump[tensor_name]["is_shape_tensor"] = tensor.is_shape_tensor
                    op_dump[tensor_name]["min"], op_dump[tensor_name]["opt"], op_dump[tensor_name]["max"] = [tuple(shape) for shape in shape_list]
        builder_config_dump["calibration_profile"] = op_dump

        self.big_json["builder_config"] = builder_config_dump
        return

    def dump_tensor(self, tensor: trt.ITensor) -> OrderedDict:
        tensor_dump = self.dump_member(tensor, self.api_exclude_set.tensor_dump_exclude_set)
        tensor_dump["dimension_name"] = [tensor.get_dimension_name(i) for i in range(len(tensor.shape))]
        tensor_dump["is_debug_tensor"] = self.network.is_debug_tensor(tensor)
        return tensor_dump

    def dump_network(self) -> None:
        network_dump = self.dump_member(self.network, self.api_exclude_set.network_dump_exclude_set, self.api_exclude_set.network_exclude_condition)

        # Flag
        dump = OrderedDict()
        for key, value in trt.NetworkDefinitionCreationFlag.__members__.items():
            dump[key] = self.network.get_flag(value)
        network_dump["flag"] = dump

        # I/O tensors
        dump = []
        for i in range(self.network.num_inputs):
            tensor = self.network.get_input(i)
            dump.append(self.dump_tensor(tensor))
        network_dump["input_tensor_list"] = dump

        dump = []
        for i in range(self.network.num_outputs):
            tensor = self.network.get_output(i)
            dump.append(self.dump_tensor(tensor))
        network_dump["output_tensor_list"] = dump

        self.big_json["network"] = network_dump
        return

    def dump_layers(self):
        self.big_json["layer"] = []
        self.big_json["tensor"] = {}  # Map: tensor name -> tensor dump
        self.big_json["loop"] = {}  # Map: loop name -> map of loop members
        self.big_json["if"] = {}  # Map: tensor name -> map of if members

        for i in range(self.network.num_layers):
            layer = self.network.get_layer(i)

            layer_type_from_base_class = layer.type  # `type` is overridden in Activation / Pooling Layer, so we need to save it before dynamic cast
            layer_dynamic_cast(layer)  # Dynamic cast to real layer type
            layer_dump = self.dump_member(layer, self.api_exclude_set.layer_dump_exclude_set)
            layer_dump["layer_index"] = i  # Extra-mark

            # Methods from BuilderConfig
            layer_dump["can_run_on_DLA"] = self.builder_config.can_run_on_DLA(layer)
            layer_dump["get_device_type"] = int(self.builder_config.get_device_type(layer))
            layer_dump["is_device_type_set"] = self.builder_config.is_device_type_set(layer)

            # Weights
            if len(layer_dump["weight_name_list"]) > 0:
                for weight_name in layer_dump["weight_name_list"]:
                    self.weights[layer.name + "-" + weight_name] = layer_dump.pop(weight_name)

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
            layer_dump["input_tensor_name_list"] = input_tensor_name_list

            output_tensor_name_list = []
            output_tensor_datatype_list = []
            output_tensor_datatype_is_set_list = []
            for j in range(layer.num_outputs):
                tensor = layer.get_output(j)
                output_tensor_name_list.append(tensor.name)
                output_tensor_datatype_list.append(int(layer.get_output_type(j)))
                output_tensor_datatype_is_set_list.append(int(layer.output_type_is_set(j)))
                self.big_json["tensor"].setdefault(tensor.name, self.dump_tensor(tensor))  # Add to "tenosr" field if it does not exist
            layer_dump["output_tensor_name_list"] = output_tensor_name_list
            layer_dump["output_tensor_datatype_list"] = output_tensor_datatype_list
            layer_dump["output_tensor_datatype_is_set_list"] = output_tensor_datatype_is_set_list

            # Special cases for some layers
            # TODO: test cases in unit test:
            # 1. Plugin_V2
            # 2. static Shuffle without setting reshape_dims
            # 3. Slice with different count of input tensors, especially Fill mode with 5 input tensors
            # 4. Whether "shape" or "scales" is parsed correctly in Resize layer
            # 5. How to check layer.shape in static resize mode + use scale mode in Resize layer
            # 6. Resize layer:
            #if layer.resize_mode == trt.ResizeMode.LINEAR and layer.coordinate_transformation == trt.ResizeCoordinateTransformation.ASYMMETRIC:
            #        print("[Warning from NetworkInspector]: ResizeCoordinateTransformation of Resize Layer %s is set as HALF_PIXEL though default behaviour or your explicit set is ASYMMETRIC mode, please refer to the source code of NetworkInspector if you insist to use ASYMMETRIC mode!" % layer.name)
            #        layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
            #        #layer.coordinate_transformation = trt.ResizeCoordinateTransformation.ASYMMETRIC # uncomment this line if you want to use ASYMMETRIC mode
            # 7. Fill layer, "shape" member in dynamic / static fill mode
            # 8. Convolution layer, no bias
            # 9. Constant layer with 0 / 1 dimension

            if isinstance(layer, (trt.IActivationLayer, trt.IPoolingLayer)):
                layer_dump["algo-type"] = layer_dump.pop("type")
                layer_dump["type"] = int(layer_type_from_base_class)

            elif isinstance(layer, (trt.IConvolutionLayer, trt.IDeconvolutionLayer)):
                layer_dump["kernel_shape"] = [layer.num_output_maps, layer.get_input(0).shape[1], *list(layer.kernel_size_nd)]
                layer_dump["bias_shape"] = list(layer.bias.shape)
                layer_dump["kernel_refittable"] = self.network.are_weights_marked_refittable(layer.name)
                layer_dump["bias_refittable"] = self.network.are_weights_marked_refittable(layer.name)

            elif isinstance(layer, trt.IScaleLayer):
                layer_dump["shift_shape"] = layer.shift.shape
                layer_dump["scale_shape"] = layer.scale.shape
                layer_dump["power_shape"] = layer.power.shape

            elif isinstance(layer, trt.IShuffleLayer):
                if self.use_patch_80:
                    try:
                        _ = len(layer.reshape_dims)
                    except ValueError:
                        layer_dump["reshape_dims"] = ()

            elif isinstance(layer, trt.IConstantLayer):
                layer_dump["weights_refittable"] = self.network.are_weights_marked_refittable(layer.name)

            elif isinstance(layer, trt.ISliceLayer):
                layer_dump["is_fill"] = (layer.mode == trt.SampleMode.FILL and layer.get_input(4) is not None)
                if self.use_patch_80:
                    axes_dump = ast.literal_eval(str(layer.axes))
                    if isinstance(axes_dump, int) and axes_dump > 8:
                        layer_dump["axes"] = None
                    try:
                        _ = len(layer.start)
                    except ValueError:
                        layer_dump["start"] = ()
                    try:
                        _ = len(layer.shape)
                    except ValueError:
                        layer_dump["shape"] = ()
                    try:
                        _ = len(layer.stride)
                    except ValueError:
                        layer_dump["stride"] = ()

            elif isinstance(layer, trt.IResizeLayer):
                is_dynamic_resize = (layer.num_inputs == 2)
                layer_dump["is_dynamic_resize"] = is_dynamic_resize
                layer_dump["is_static_scale_mode"] = (not is_dynamic_resize and len(layer.scales) > 0)

            elif isinstance(layer, trt.IFillLayer):
                is_dynamic_fill = layer.get_input(0) is not None
                layer_dump["is_dynamic_fill"] = is_dynamic_fill
                if is_dynamic_fill:  # The shape of output tensor is determined by input tenor 0 in dynamic fill mode
                    layer_dump["shape"] = []  # Just a place-holder

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
                    d["ConditionLayer"] = None
                    d["ConditionalInputLayer"] = None
                    d["ConditionalOutputLayer"] = None
                    self.big_json["if"][if_name] = d
                key = {trt.LayerType.CONDITION: "ConditionLayer", trt.LayerType.CONDITIONAL_INPUT: "ConditionalInputLayer", trt.LayerType.CONDITIONAL_OUTPUT: "ConditionalOutputLayer"}.get(layer.type)
                self.big_json["if"][if_name][key] = layer.name

            self.big_json["layer"].append(layer_dump)

        return

    # Deserialization tool functions ===================================================================================
    def build_member(self, obj: object = None, dump: OrderedDict = OrderedDict(), exclude_set: list = [], exclude_condition=(lambda x: False)) -> None:
        if obj is None:
            self.log("ERROR", f"{str(obj)} is None")
            return

        for key, value in dump.items():
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
        network_dump = self.big_json["network"]
        # Using `flag` rather than `flags`
        #flag = network_dump["flags"]
        flag = 0
        for key, value in trt.NetworkDefinitionCreationFlag.__members__.items():
            if network_dump["flag"][key]:
                flag |= 1 << int(getattr(trt.NetworkDefinitionCreationFlag, key))
        self.network = self.builder.create_network(flag)
        self.build_member(self.network, network_dump, self.api_exclude_set.network_build_exclude_set)

        return

    def build_tensor(self, tensor, tensor_dump) -> OrderedDict:
        self.build_member(tensor, tensor_dump, self.api_exclude_set.tensor_build_exclude_set)
        self.tensor_map[tensor.name] = tensor

        # Data Type
        if tensor.is_network_input or tensor_dump["is_network_output"]:  # Do not use `tensor.is_network_input` since no tensor is marked as output till now
            if trt.DataType(tensor_dump["dtype"]) != trt.DataType.FP4:  # Skip output FP4 since numpy can not deal with this. TODO: remove this constrain
                tensor.dtype = trt.DataType(tensor_dump["dtype"])  # No effect for intermediate tensors

        # Dimension Name
        if tensor.is_network_input:
            for i in range(len(tensor_dump["shape"])):
                tensor.set_dimension_name(i, tensor_dump["dimension_name"][i])

        # Dynamic Range
        if tensor_dump["dynamic_range"] is not None:
            tensor.dynamic_range = tensor_dump["dynamic_range"]

        # Location
        tensor.location = trt.TensorLocation(tensor_dump["location"])

        # Allowed Format
        # Do not set it if there is no constrain on tensor, or error like below will be raised:
        # (x: has dataType Float unsupported by tensor's allowed TensorFormats.)
        if tensor_dump["allowed_formats"] != 2 ** len(trt.TensorFormat.__members__) - 1:  # All 1 bit mask
            bit_mask = tensor_dump["allowed_formats"]
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
                message = f"Allowed format {tensor_dump['allowed_formats']} of tensor {tensor.name} with data type {trt.DataType(tensor.dtype)} is not supported"
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

    def update_loop(self, layer_dump, argument_list, attribution_map):
        layer_name = layer_dump["name"]
        kind_name = {trt.LayerType.TRIP_LIMIT: "trip_limit_layer_name", trt.LayerType.RECURRENCE: "recurrence_layer_name_list", trt.LayerType.ITERATOR: "iterator_layer_name_list", trt.LayerType.LOOP_OUTPUT: "loop_output_layer_name_list"}.get(trt.LayerType(layer_dump["type"]))
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
            layer = loop.add_trip_limit(self.tensor_map[layer_dump["input_tensor_name_list"][0]], trt.TripLimit(layer_dump["kind"]))
            attribution_map["kind"] = None
        elif kind_name == "recurrence_layer_name_list":
            layer = loop.add_recurrence(self.tensor_map[layer_dump["input_tensor_name_list"][0]])
            if layer_dump["input_tensor_name_list"][1] in self.tensor_map:
                layer.set_input(1, self.tensor_map[layer_dump["input_tensor_name_list"][1]])
            self.loop_map[loop_name]["recurrence_layer_list"].append(layer)
        elif kind_name == "iterator_layer_name_list":
            layer = loop.add_iterator(self.tensor_map[layer_dump["input_tensor_name_list"][0]], layer_dump["axis"], layer_dump["reverse"])
            attribution_map["reverse"] = None
        elif kind_name == "loop_output_layer_name_list":
            layer = loop.add_loop_output(self.tensor_map[layer_dump["input_tensor_name_list"][0]], trt.LoopOutput(layer_dump["kind"]), layer_dump["axis"])
            self.loop_map[loop_name]["loop_output_layer_list"].append(layer)
            attribution_map["kind"] = None
        else:
            self.log("ERROR", f"Error loop layer name: {kind_name}")

        return layer

    def update_if(self, layer_dump, argument_list):
        layer_name = layer_dump["name"]
        kind_name = {trt.LayerType.CONDITION: "ConditionLayer", trt.LayerType.CONDITIONAL_INPUT: "ConditionalInputLayer", trt.LayerType.CONDITIONAL_OUTPUT: "ConditionalOutputLayer"}.get(trt.LayerType(layer_dump["type"]))
        if_name = None
        for key, value in self.if_map.items():
            if layer_name == value[kind_name]:
                if_name = key
                break
        if if_name is None:
            for key, value in self.big_json["if"].items():
                if layer_name == value[kind_name]:
                    self.if_map[key] = {}
                    self.if_map[key]["ConditionLayer"] = layer_name if kind_name == "ConditionLayer" else value["ConditionLayer"]
                    self.if_map[key]["ConditionalInputLayer"] = layer_name if kind_name == "ConditionalInputLayer" else value["ConditionalInputLayer"]
                    self.if_map[key]["ConditionalOutputLayer"] = layer_name if kind_name == "ConditionalOutputLayer" else value["ConditionalOutputLayer"]
                    self.if_map[key]["IfCondition"] = self.network.add_if_conditional()  # Save if object
                    if_name = key
                    break
        loop = self.if_map[if_name]["IfCondition"]
        if kind_name == "ConditionLayer":
            layer = loop.set_condition(self.tensor_map[layer_dump["input_tensor_name_list"][0]])
        elif kind_name == "ConditionalInputLayer":
            layer = loop.add_input(self.tensor_map[layer_dump["input_tensor_name_list"][0]])
        elif kind_name == "ConditionalOutputLayer":
            layer = loop.add_output(self.tensor_map[layer_dump["input_tensor_name_list"][0]], self.tensor_map[layer_dump["input_tensor_name_list"][1]])
        else:
            self.log("ERROR", f"Error loop layer name: {kind_name}")

        return layer

    def build_layer(self, layer_dump) -> OrderedDict:
        layer_index = layer_dump["layer_index"]
        layer_type = trt.LayerType(layer_dump["type"])
        add_layer_method_name = layer_type_to_add_layer_method_name(layer_type)  # Get exact name of `add_*` API for this layer
        argument_list = []  # List of tensors / arguments used in `add_*` API
        self.set_input_tensor_map = {}  # Map: index -> tensor,  tensors used in `set_input` API
        attribution_map = {}  # Map: attribution name -> attribution value

        if len(layer_dump["input_tensor_name_list"]) > 0 and layer_dump["input_tensor_name_list"][0] is not None:  # Some layers has no input tensor
            assert layer_dump["input_tensor_name_list"][0] in self.tensor_map
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][0]])

        need_call_add_layer = True  # If-Condition or Loop structure do not need to add layer in network

        # Use `match-case` when yapf supports
        # Sorted by `int(layer_type)`, might change with TRT release in the future
        if layer_type in [trt.LayerType.CONVOLUTION, trt.LayerType.DECONVOLUTION]:  # 0, 7
            assert len(layer_dump["input_tensor_name_list"]) in [1, 2]
            argument_list.extend([layer_dump["num_output_maps"], layer_dump["kernel_size_nd"]])
            if self.weights is not None:
                kernel = self.weights[layer_dump["name"] + "-kernel"]
                bias = self.weights[layer_dump["name"] + "-bias"]
            else:
                kernel_shape = layer_dump["lKernelShape"]
                bias_shape = layer_dump["lBiasShape"]
                kernel = np.random.rand(np.prod(kernel_shape)).astype(np.float32).reshape(kernel_shape) * 2 - 1
                bias = np.random.rand(np.prod(bias_shape)).astype(np.float32).reshape(bias_shape) * 2 - 1
            argument_list.append(trt.Weights(np.ascontiguousarray(kernel)))
            argument_list.append(trt.Weights(np.ascontiguousarray(bias)))
            if np.prod(kernel.shape) == 0:  # Int8-QDQ
                assert len(layer_dump["input_tensor_name_list"]) == 2
                argument_list[-2] = trt.Weights()
                self.add_set_input_tensor(layer_index, layer_dump["input_tensor_name_list"], 1)
            attribution_map["padding_mode"] = trt.PaddingMode(layer_dump["padding_mode"])

        elif layer_type == trt.LayerType.CAST:  # 1
            assert len(layer_dump["input_tensor_name_list"]) == 1
            argument_list.append(trt.DataType(layer_dump["to_type"]))
            attribution_map["to_type"] = trt.DataType(layer_dump["to_type"])

        elif layer_type == trt.LayerType.ACTIVATION:  # 2
            assert len(layer_dump["input_tensor_name_list"]) == 1
            argument_list.append(trt.ActivationType(layer_dump["algo-type"]))
            attribution_map["type"] = trt.ActivationType(layer_dump["algo-type"])

        elif layer_type == trt.LayerType.POOLING:  # 3
            assert len(layer_dump["input_tensor_name_list"]) == 1
            argument_list.append(trt.PoolingType(layer_dump["algo-type"]))
            argument_list.append(layer_dump["window_size_nd"])
            attribution_map["padding_mode"] = trt.PaddingMode(layer_dump["padding_mode"])
            attribution_map["type"] = trt.PoolingType(layer_dump["algo-type"])

        elif layer_type == trt.LayerType.LRN:  # 4
            assert len(layer_dump["input_tensor_name_list"]) == 1
            argument_list.extend([layer_dump["window_size"], layer_dump["alpha"], layer_dump["beta"], layer_dump["k"]])

        elif layer_type == trt.LayerType.SCALE:  # 5
            assert len(layer_dump["input_tensor_name_list"]) == 1
            argument_list.append(trt.ScaleMode(layer_dump["mode"]))
            if self.weights is not None:
                shift = self.weights[layer_dump["name"] + "-shift"]
                scale = self.weights[layer_dump["name"] + "-scale"]
                power = self.weights[layer_dump["name"] + "-power"]
            else:
                shift_shape = layer_dump["shift_shape"]
                scale_shape = layer_dump["scale_shape"]
                power_shape = layer_dump["power_shape"]
                shift = np.random.rand(np.prod(shift_shape)).astype(np.float32).reshape(shift_shape) * 2 - 1
                scale = np.random.rand(np.prod(scale_shape)).astype(np.float32).reshape(scale_shape) * 2 - 1
                power = np.ones(np.prod(power_shape)).astype(np.float32).reshape(power_shape)
            argument_list.append(trt.Weights(np.ascontiguousarray(shift)))
            argument_list.append(trt.Weights(np.ascontiguousarray(scale)))
            argument_list.append(trt.Weights(np.ascontiguousarray(power)))
            argument_list.append(layer_dump["channel_axis"])

        elif layer_type == trt.LayerType.SOFTMAX:  # 6
            assert len(layer_dump["input_tensor_name_list"]) == 1

        elif layer_type == trt.LayerType.CONCATENATION:  # 8
            assert len(layer_dump["input_tensor_name_list"]) > 0
            input_tensor_list = []
            for tensor_name in layer_dump["input_tensor_name_list"]:
                input_tensor_list.append(self.tensor_map[tensor_name])
            argument_list = [input_tensor_list]  # Rather than `append`

        elif layer_type == trt.LayerType.ELEMENTWISE:  # 9
            assert len(layer_dump["input_tensor_name_list"]) == 2
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][1]])
            argument_list.append(trt.ElementWiseOperation(layer_dump["op"]))
            attribution_map["op"] = trt.ElementWiseOperation(layer_dump["op"])

        elif layer_type in [trt.LayerType.PLUGIN, trt.LayerType.PLUGIN_V2, trt.LayerType.PLUGIN_V3]:  # 10, 21, 46
            self.log("ERROR", "Plugin Layer not supported")  # TODO: add support for plugin to remove this

        elif layer_type == trt.LayerType.UNARY:  # 11
            assert len(layer_dump["input_tensor_name_list"]) == 1
            argument_list.append(trt.UnaryOperation(layer_dump["op"]))
            attribution_map["op"] = trt.UnaryOperation(layer_dump["op"])

        elif layer_type == trt.LayerType.PADDING:  # 12
            assert len(layer_dump["input_tensor_name_list"]) == 1
            argument_list.extend([layer_dump["pre_padding_nd"], layer_dump["post_padding_nd"]])

        elif layer_type == trt.LayerType.SHUFFLE:  # 13
            assert len(layer_dump["input_tensor_name_list"]) in [1, 2]
            if len(layer_dump["input_tensor_name_list"]) == 2:  # Dynamic shuffle
                self.add_set_input_tensor(layer_index, layer_dump["input_tensor_name_list"], 1)
                attribution_map["reshape_dims"] = None  # Skip setting it in attribution
            if self.use_patch_80:
                if isinstance(layer_dump["reshape_dims"], list) and len(layer_dump["reshape_dims"]) == 0:
                    attribution_map["reshape_dims"] = None  # overwrite useless value

        elif layer_type == trt.LayerType.REDUCE:  # 14
            assert len(layer_dump["input_tensor_name_list"]) == 1
            argument_list.append(trt.ReduceOperation(layer_dump["op"]))
            argument_list.append(layer_dump["axes"])
            argument_list.append(layer_dump["keep_dims"])
            attribution_map["op"] = trt.ReduceOperation(layer_dump["op"])

        elif layer_type == trt.LayerType.TOPK:  # 15
            assert len(layer_dump["input_tensor_name_list"]) in [1, 2]
            argument_list.append(trt.TopKOperation(layer_dump["op"]))
            if len(layer_dump["input_tensor_name_list"]) == 2:
                self.add_set_input_tensor(layer_index, layer_dump["input_tensor_name_list"], 1)
                argument_list.append(0)  # Place-holder
            else:
                argument_list.append(layer_dump["k"])
            argument_list.append(layer_dump["axes"])
            attribution_map["op"] = None  # Do not set it in attribution

        elif layer_type == trt.LayerType.GATHER:  # 16
            assert len(layer_dump["input_tensor_name_list"]) == 2
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][1]])
            argument_list.append(trt.GatherMode(layer_dump["mode"]))
            attribution_map["axis"] = layer_dump["axis"]

        elif layer_type == trt.LayerType.MATRIX_MULTIPLY:  # 17
            assert len(layer_dump["input_tensor_name_list"]) == 2
            argument_list.append(trt.MatrixOperation(layer_dump["op0"]))
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][1]])
            argument_list.append(trt.MatrixOperation(layer_dump["op1"]))
            attribution_map["op0"] = trt.MatrixOperation(layer_dump["op0"])
            attribution_map["op1"] = trt.MatrixOperation(layer_dump["op1"])

        elif layer_type == trt.LayerType.RAGGED_SOFTMAX:  # 18
            assert len(layer_dump["input_tensor_name_list"]) == 2
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][1]])

        elif layer_type == trt.LayerType.CONSTANT:  # 19
            assert len(layer_dump["input_tensor_name_list"]) == 0
            if self.weights is not None:
                weight = self.weights[layer_dump["name"] + "-weights"]
            else:
                weight_shape = layer_dump["shape"]
                data_type = trt.nptype(trt.DataType(layer_dump["output_data_type_list"][0]))
                weight = np.random.rand(np.prod(weight_shape)).astype(data_type).reshape(weight_shape)
            if weight.shape == (0, ):  # Special process for weight of shape (0,)
                argument_list.extend([[0], trt.Weights(datatype_np_to_trt(weight.dtype))])
                #argument_list.extend([weight.shape, trt.Weights(np.ascontiguousarray(weight))])
            else:
                argument_list.extend([weight.shape, trt.Weights(np.ascontiguousarray(weight))])

        elif layer_type == trt.LayerType.IDENTITY:  # 20
            assert len(layer_dump["input_tensor_name_list"]) == 1

        elif layer_type == trt.LayerType.SLICE:  # 22
            n_input_tensor = len(layer_dump["input_tensor_name_list"])
            assert n_input_tensor in [1, 2, 3, 4, 5, 6]
            argument_list.extend([layer_dump["start"], layer_dump["shape"], layer_dump["stride"]])  # Place-holders
            if n_input_tensor >= 2 and layer_dump["start"] == []:
                argument_list[1] = [0] * argument_list[0].shape[0]  # set `start` forcely in dynamic slice mode
            if n_input_tensor >= 2 and layer_dump["input_tensor_name_list"][1] is not None:
                self.add_set_input_tensor(layer_index, layer_dump["input_tensor_name_list"], 1)
                attribution_map["start"] = None  # Do not set it if using input tensor
            if n_input_tensor >= 3 and layer_dump["input_tensor_name_list"][2] is not None:
                self.add_set_input_tensor(layer_index, layer_dump["input_tensor_name_list"], 2)
                attribution_map["shape"] = None  # Do not set it if using input tensor
            if n_input_tensor >= 4 and layer_dump["input_tensor_name_list"][3] is not None:
                self.add_set_input_tensor(layer_index, layer_dump["input_tensor_name_list"], 3)
                attribution_map["stride"] = None  # Do not set it if using input tensor
            if n_input_tensor >= 5 and trt.SampleMode(layer_dump["mode"]) == trt.SampleMode.FILL and layer_dump["is_fill"] == True:
                self.add_set_input_tensor(layer_index, layer_dump["input_tensor_name_list"], 4)
            attribution_map["mode"] = trt.SampleMode(layer_dump["mode"])
            attribution_map["is_fill"] = None  # Extra-mark

        elif layer_type == trt.LayerType.SHAPE:  # 23
            assert len(layer_dump["input_tensor_name_list"]) == 1

        elif layer_type == trt.LayerType.PARAMETRIC_RELU:  # 24
            assert len(layer_dump["input_tensor_name_list"]) == 2
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][1]])

        elif layer_type == trt.LayerType.RESIZE:  # 25
            assert len(layer_dump["input_tensor_name_list"]) in [1, 2]
            if layer_dump["is_dynamic_resize"]:
                assert len(layer_dump["input_tensor_name_list"]) == 2
                self.add_set_input_tensor(layer_index, layer_dump["input_tensor_name_list"], 1)
            if layer_dump["is_static_scale_mode"]:
                attribution_map["shape"] = None  # Skip setting `shape` in static scale mode
            attribution_map["resize_mode"] = trt.InterpolationMode(layer_dump["resize_mode"])
            attribution_map["coordinate_transformation"] = trt.ResizeCoordinateTransformation(layer_dump["coordinate_transformation"])
            attribution_map["selector_for_single_pixel"] = trt.ResizeSelector(layer_dump["selector_for_single_pixel"])
            attribution_map["nearest_rounding"] = trt.ResizeRoundMode(layer_dump["nearest_rounding"])
            attribution_map["is_dynamic_resize"] = None
            attribution_map["is_static_shape_mode"] = None

        elif layer_type in [trt.LayerType.TRIP_LIMIT, trt.LayerType.RECURRENCE, trt.LayerType.ITERATOR, trt.LayerType.LOOP_OUTPUT]:  # 26, 27, 28, 29
            layer = self.update_loop(layer_dump, argument_list, attribution_map)
            need_call_add_layer = False

        elif layer_type == trt.LayerType.SELECT:  # 30
            assert len(layer_dump["input_tensor_name_list"]) == 3
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][1]])
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][2]])

        elif layer_type == trt.LayerType.FILL:  # 31
            assert len(layer_dump["input_tensor_name_list"]) in [0, 2, 3]
            if layer_dump["is_dynamic_fill"]:
                argument_list = []  # Remove the first input tensor and use `set_input` instead
                self.add_set_input_tensor(layer_index, layer_dump["input_tensor_name_list"], 0)
            argument_list.extend([trt.Dims(layer_dump["shape"]), trt.FillOperation(layer_dump["operation"]), trt.DataType(layer_dump["to_type"])])
            if len(layer_dump["input_tensor_name_list"]) == 3:
                self.add_set_input_tensor(layer_index, layer_dump["input_tensor_name_list"], 1)
                self.add_set_input_tensor(layer_index, layer_dump["input_tensor_name_list"], 2)
                attribution_map["alpha"] = None  # Do not set `alpha` and `beta` in this case, or error like below will be raised:
                attribution_map["beta"] = None
                # Skipping tactic 0x0000000000000000 due to exception Assertion dims.nbDims == 1 failed. Alpha and beta tensor should be set when output an ND tensor.
            #if "Range" in layer_dump["name"]:  # The special case: Range node from ONNX, TODO: remove this branch
            #    self.set_input_tensor_map[1] = self.constant_layers_for_Range_node[0].get_output(0)
            #    self.set_input_tensor_map[2] = self.constant_layers_for_Range_node[1].get_output(0)
            # Set some attributions as None to avoid setting them in `build_member`
            attribution_map["is_alpha_beta_int64"] = None  # Read-only atttribution
            attribution_map["is_dynamic_fill"] = None  # Extra-mark
            attribution_map["shape"] = None  # Useless after `add_fill` is called
            attribution_map["to_type"] = None  # Do not set it after `add_fill` is called

        elif layer_type in [trt.LayerType.QUANTIZE, trt.LayerType.DEQUANTIZE]:  # 32, 33
            assert len(layer_dump["input_tensor_name_list"]) in [2, 3]
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][1]])
            if len(layer_dump["input_tensor_name_list"]) == 3:
                self.add_set_input_tensor(layer_index, layer_dump["input_tensor_name_list"], 2)

        elif layer_type in [trt.LayerType.CONDITION, trt.LayerType.CONDITIONAL_INPUT, trt.LayerType.CONDITIONAL_OUTPUT]:  # 34, 35, 36
            layer = self.update_if(layer_dump, argument_list)
            need_call_add_layer = False

        elif layer_type == trt.LayerType.SCATTER:  # 37
            assert len(layer_dump["input_tensor_name_list"]) == 3
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][1]])
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][2]])
            argument_list.append(trt.ScatterMode(layer_dump["mode"]))

        elif layer_type == trt.LayerType.EINSUM:  # 38
            assert len(layer_dump["input_tensor_name_list"]) > 0
            input_tensor_list = []
            for tensor_name in layer_dump["input_tensor_name_list"]:
                input_tensor_list.append(self.tensor_map[tensor_name])
            argument_list = [input_tensor_list]  # Rather than `append`
            argument_list.append(layer_dump["equation"])

        elif layer_type == trt.LayerType.ASSERTION:  # 39
            assert len(layer_dump["input_tensor_name_list"]) == 1
            argument_list.append(layer_dump["message"])

        elif layer_type == trt.LayerType.ONE_HOT:  # 40
            assert len(layer_dump["input_tensor_name_list"]) == 3
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][1]])
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][2]])
            argument_list.append(layer_dump["axis"])

        elif layer_type == trt.LayerType.NON_ZERO:  # 41
            assert len(layer_dump["input_tensor_name_list"]) == 1

        elif layer_type == trt.LayerType.GRID_SAMPLE:  # 42
            assert len(layer_dump["input_tensor_name_list"]) == 2
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][1]])
            attribution_map["interpolation_mode"] = trt.InterpolationMode(layer_dump["interpolation_mode"])
            attribution_map["sample_mode"] = trt.SampleMode(layer_dump["sample_mode"])

        elif layer_type == trt.LayerType.NMS:  # 43
            assert len(layer_dump["input_tensor_name_list"]) == 3
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][1]])
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][2]])
            attribution_map["bounding_box_format"] = trt.BoundingBoxFormat(layer_dump["bounding_box_format"])

        elif layer_type == trt.LayerType.REVERSE_SEQUENCE:  # 44
            assert len(layer_dump["input_tensor_name_list"]) == 2
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][1]])

        elif layer_type == trt.LayerType.NORMALIZATION:  # 45
            assert len(layer_dump["input_tensor_name_list"]) == 3
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][1]])
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][2]])
            argument_list.append(layer_dump["axes"])
            attribution_map["compute_precision"] = trt.DataType(layer_dump["compute_precision"])

        elif layer_type in [trt.LayerType.SQUEEZE, trt.LayerType.UNSQUEEZE]:  # 47, 48
            assert len(layer_dump["input_tensor_name_list"]) == 2
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][1]])

        elif layer_type == trt.LayerType.CUMULATIVE:  # 49
            assert len(layer_dump["input_tensor_name_list"]) == 2
            argument_list.append(self.tensor_map[layer_dump["input_tensor_name_list"][1]])
            argument_list.append(trt.CumulativeOperation(layer_dump["op"]))
            argument_list.append(layer_dump["exclusive"])
            argument_list.append(layer_dump["reverse"])
            attribution_map["op"] = trt.CumulativeOperation(layer_dump["op"])

        elif layer_type == trt.LayerType.DYNAMIC_QUANTIZE:  # 50
            assert len(layer_dump["input_tensor_name_list"]) == 2
            argument_list.append(layer_dump["axis"])
            argument_list.append(layer_dump["block_size"])
            argument_list.append(trt.DataType(layer_dump["to_type"]))
            argument_list.append(trt.DataType(layer_dump["scale_type"]))
            self.add_set_input_tensor(layer_index, layer_dump["input_tensor_name_list"], 1)
            attribution_map["to_type"] = trt.DataType(layer_dump["to_type"])
            attribution_map["scale_type"] = trt.DataType(layer_dump["scale_type"])

        else:
            self.log("ERROR", f"Error parsing layer {layer_dump['name']}, type: {layer_type}")

        # Add the layer and set attributions
        if need_call_add_layer:
            layer = getattr(self.network, add_layer_method_name)(*argument_list)

        for index, tensor in self.set_input_tensor_map.items():
            layer.set_input(index, tensor)

        # Set attributions
        self.build_member(layer, layer_dump, self.api_exclude_set.layer_build_exclude_set | set(attribution_map.keys()))
        for key, value in attribution_map.items():
            if value is not None:
                setattr(layer, key, value)

        # Method from BuilderConfig
        self.builder_config.set_device_type(layer, trt.DeviceType(layer_dump["get_device_type"]))

        # More operations after adding the layer
        if layer_type in [trt.LayerType.QUANTIZE, trt.LayerType.DEQUANTIZE]:  # 32, 33
            if layer.axis == -1:  # TODO: check whether this is necessary
                layer.axis = 0  # Change it into 0 (per-tensor Quantization / Dequantization)

        if layer_index in self.later_layer_map:
            self.later_layer_map[layer_index].append(layer)  # Add layer object to the map

        # Build output tensors, mark debug tensors
        assert layer.num_outputs == len(layer_dump["output_tensor_name_list"])
        for i, tensor_name in enumerate(layer_dump["output_tensor_name_list"]):
            tensor_dump = self.big_json["tensor"][tensor_name]
            layer.set_output_type(i, trt.DataType(tensor_dump["dtype"]))
            self.build_tensor(layer.get_output(i), tensor_dump)
            if tensor_dump["is_debug_tensor"]:
                self.network.mark_debug(layer.get_output(i))

        return

    def build_layers(self):
        network_dump = self.big_json["network"]
        layer_dump = self.big_json["layer"]
        tensor_dump = self.big_json["tensor"]
        # Map from "tensor name" to "corresponding tensors which has been in the built network"
        self.tensor_map = OrderedDict()
        # Map from "if structure name" to "names of corresponding layers in this structure"
        self.if_map = {}
        # Map: from "loop structure name" to "names of corresponding layers in this structure"
        self.loop_map = {}
        # Map from "layer index" to "corresponding name of missing tensor in this layer"
        # In some cases, the shape tensor consumed in an early layer is produced by a later layer, so we need to mark them
        self.later_layer_map = {}

        for tensor_dump in network_dump["input_tensor_list"]:
            tensor = self.network.add_input(tensor_dump["name"], trt.DataType(tensor_dump["dtype"]), tensor_dump["shape"])
            self.tensor_map[tensor.name] = tensor
            self.build_tensor(tensor, tensor_dump)

        # Constant layer for Range Node from ONNX file
        constant_layer0_for_Range = self.network.add_constant([], trt.Weights(np.ascontiguousarray(np.array([0], dtype=np.int32))))
        constant_layer0_for_Range.name = "ConstantLayer0ForRangeNoe"
        constant_layer0_for_Range.get_output(0).name = "ConstantTensor0ForRangeNoe"
        constant_layer1_for_Range = self.network.add_constant([1], trt.Weights(np.ascontiguousarray(np.array([1], dtype=np.int32))))
        constant_layer1_for_Range.name = "ConstantLayer1ForRangeNoe"
        constant_layer1_for_Range.get_output(0).name = "ConstantTensor1ForRangeNoe"
        self.constant_layers_for_Range_node = [constant_layer0_for_Range, constant_layer1_for_Range]

        # Rebuild network layer by layer
        for i, singe_layer_dump in enumerate(layer_dump):
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
        for tensor in network_dump["output_tensor_list"]:
            output_tensor = self.tensor_map[tensor["name"]]
            assert (output_tensor.name in self.tensor_map)
            self.network.mark_output(output_tensor)

        return

    def build_profile(self) -> None:
        for op_dump in self.big_json["builder_config"]["optimization_profile_list"]:
            op = self.builder.create_optimization_profile()
            for j in range(self.network.num_inputs):
                tensor_name = self.network.get_input(j).name
                assert tensor_name in op_dump
                if op_dump[tensor_name] == {}:
                    continue
                argument_list = [tensor_name, op_dump[tensor_name]["min"], op_dump[tensor_name]["opt"], op_dump[tensor_name]["max"]]
                if op_dump[tensor_name]["is_shape_tensor"]:
                    op.set_shape_input(*argument_list)
                else:
                    op.set_shape(*argument_list)
            self.builder_config.add_optimization_profile(op)

        # Int8 Calibration Profile
        op_dump = self.big_json["builder_config"]["calibration_profile"]
        if len(op_dump) > 0:
            op = self.builder.create_optimization_profile()
            for j in range(self.network.num_inputs):
                tensor_name = self.network.get_input(j).name
                assert tensor_name in op_dump
                if op_dump[tensor_name] == {}:
                    continue
                argument_list = [tensor_name, op_dump[tensor_name]["min"], op_dump[tensor_name]["opt"], op_dump[tensor_name]["max"]]
                if op_dump[tensor_name]["is_shape_tensor"]:
                    op.set_shape_input(*argument_list)
                else:
                    op.set_shape(*argument_list)
            self.config.set_calibration_profile(op)

        return
