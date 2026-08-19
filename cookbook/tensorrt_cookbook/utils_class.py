# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
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

import ctypes
from collections import OrderedDict
from pathlib import Path
from typing import Union

import numpy as np
import nvtx
import tensorrt as trt
import torch
from cuda.bindings import runtime as cudart

from .utils_cookbook import text_to_logger_level
from .utils_function import (datatype_cast, print_array_information)
from .utils_plugin import load_plugin_files

class CookbookLogger(trt.ILogger):
    """Simple custom TensorRT logger with configurable minimum severity."""

    def __init__(self, min_severity=trt.ILogger.Severity.INTERNAL_ERROR) -> None:
        """Initialize logger with a minimum severity threshold."""
        trt.ILogger.__init__(self)
        # int(trt.ILogger.Severity.INTERNAL_ERROR) == 0
        # int(trt.ILogger.Severity.ERROR) == 1
        # int(trt.ILogger.Severity.WARNING) == 2
        # int(trt.ILogger.Severity.INFO) == 3
        # int(trt.ILogger.Severity.VERBOSE) == 4
        self.min_severity = min_severity

    def log(self, severity, msg) -> None:
        """Emit a message when ``severity`` is above the configured threshold."""
        if severity <= self.min_severity:
            print(f"[My Logger] {msg}")  # customerized log content

class CookbookProfiler(trt.IProfiler):
    """Profiler callback that prints per-layer execution time."""

    def __init__(self) -> None:
        """Initialize TensorRT profiler callback."""
        super().__init__()

    def report_layer_time(self, layer_name, time_ms) -> None:
        """Print elapsed execution time for one layer."""
        print(f"Timing: {time_ms * 1000: 8.3f}us -> {layer_name}")

class CookbookDebugListener(trt.IDebugListener):  # `trt.IDebugListener` since TensorRT-10.0
    # implement a call back class to get information of the debug tensors
    """Debug tensor callback that can print and optionally validate tensor values."""

    def __init__(self, expect_result: dict | None = None, epsilon: float = 1e-5, log: bool = False):
        """Initialize debug listener with optional expected tensors for validation."""
        expect_result = expect_result or {}

        if log:
            print("[CookbookDebugListener::__init__]")
        super().__init__()
        self.expect_result = expect_result  # an optional dictionary containing expected result
        self.epsilon = epsilon
        self.log = log

    def process_debug_tensor(
        self,
        addr,
        location: trt.TensorLocation,
        type: trt.tensorrt.DataType,
        shape: trt.tensorrt.Dims,
        name: str,
        stream: int,
    ):
        """Copy, print, and optionally compare a debug tensor captured by TensorRT."""
        host_buffer = np.empty(tuple(shape), dtype=trt.nptype(type))
        if location == trt.TensorLocation.DEVICE:
            cudart.cudaStreamSynchronize(stream)  # might be removed in the future
            cudart.cudaMemcpyAsync(host_buffer.ctypes.data, addr, host_buffer.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
            cudart.cudaStreamSynchronize(stream)
        else:  # location == trt.TensorLocation.HOST
            ctypes.memmove(host_buffer.ctypes.data, addr, host_buffer.nbytes)  # copy from the host address into our buffer

        # we can print information from `host_buffer` here
        print_array_information(host_buffer, name)

        # Compare host_buffer with optional expected result
        if name in self.expect_result.keys():
            diff = np.max(np.abs(host_buffer - self.expect_result[name])) < self.epsilon
            print(f"#### Check debug tensor {name}: {diff} ####")  # print result by print for assert or anything else,

        return True  # return value does not reflect the check

class CookbookErrorRecorder(trt.IErrorRecorder):
    """Error recorder implementation for collecting TensorRT runtime/build errors."""

    def __init__(self, log: bool = False) -> None:
        """Initialize an in-memory TensorRT error recorder."""
        if log:
            print("[CookbookErrorRecorder::__init__]")
        super().__init__()
        self.error_list = []
        self.n_max_error = 256
        self.log = log

    def clear(self) -> None:
        """Clear all recorded errors."""
        if self.log:
            print("[CookbookErrorRecorder::clear]")
        self.error_list = []
        return None

    def get_error_code(self, index) -> int:
        """Return error code at ``index`` or SUCCESS when index is invalid."""
        if self.log:
            print(f"[CookbookErrorRecorder::get_error_code] {index=}")
        # Values of error code
        # trt.ErrorCodeTRT.SUCCESS  # 0
        # trt.ErrorCodeTRT.UNSPECIFIED_ERROR  # 1
        # trt.ErrorCodeTRT.INTERNAL_ERROR  # 2
        # trt.ErrorCodeTRT.INVALID_ARGUMENT  # 3
        # trt.ErrorCodeTRT.INVALID_CONFIG  # 4
        # trt.ErrorCodeTRT.FAILED_ALLOCATION  # 5
        # trt.ErrorCodeTRT.FAILED_INITIALIZATION  # 6
        # trt.ErrorCodeTRT.FAILED_EXECUTION  # 7
        # trt.ErrorCodeTRT.FAILED_COMPUTATION  # 8
        # trt.ErrorCodeTRT.INVALID_STATE  # 9
        # trt.ErrorCodeTRT.UNSUPPORTED_STATE  # 10
        if index < 0 or index >= len(self.error_list):
            print("Error index")
            return trt.ErrorCodeTRT.SUCCESS
        return self.error_list[index][0]

    def get_error_desc(self, index) -> str:
        """Return error description at ``index`` or empty string when invalid."""
        if self.log:
            print(f"[CookbookErrorRecorder::get_error_desc] {index=}")
        if index < 0 or index >= len(self.error_list):
            print("Error index")
            return ""
        return self.error_list[index][1]

    def has_overflowed(self) -> bool:
        """Return whether recorded errors reached capacity."""
        if self.log:
            print("[CookbookErrorRecorder::has_overflowed]")
        return len(self.error_list) >= self.n_max_error

    def num_errors(self) -> int:
        """Return number of currently recorded errors."""
        if self.log:
            print("[CookbookErrorRecorder::num_errors]")
        return len(self.error_list)

    def report_error(self, error_code, error_description) -> None:
        """Append one TensorRT error into the internal list."""
        print(f"[CookbookErrorRecorder::report_error]\n    n={len(self.error_list)},code={error_code},info={error_description}")
        self.error_list.append([error_code, error_description])
        if self.has_overflowed():
            print("Error Overflow")
        return

    def hello_world(self) -> str:  # not necessary API
        """Return object identity text for debugging."""
        return str(id(self))

class CookbookGpuAllocator(trt.IGpuAllocator):
    """GPU allocator implementation that tracks allocation metadata."""

    def __init__(self, log: bool = False):
        """Initialize allocator state and allocation tracking arrays."""
        if log:
            print("[CookbookGpuAllocator::__init__]")
        super().__init__()
        self.address_list = []
        self.flag_list = []
        self.size_list = []
        self.log = log

    def allocate(self, size, alignment, flag):
        """Allocate device memory and track metadata for later reallocation/free."""
        if self.log:
            print(f"[CookbookGpuAllocator::allocate] {size=},{alignment=},{flag=}")
        status, address = cudart.cudaMalloc(size)
        if status != cudart.cudaError_t.cudaSuccess:
            print(f"Fail allocating {size}B")
            return 0
        self.address_list.append(address)
        self.flag_list.append(bool(flag))  # Size is flexible (reallocate can be called) if True, which is contrary with int(trt.AllocatorFlag.RESIZABLE) == 0
        self.size_list.append(size)
        return address

    def deallocate(self, address):
        """Free a tracked device allocation by address."""
        if self.log:
            print(f"[CookbookGpuAllocator::deallocate] {address=}")
        try:
            index = self.address_list.index(address)
        except ValueError as e:
            print(f"Fail finding address {address} in address_list, {e}")
            return False

        status = cudart.cudaFree(address)
        if status[0] != cudart.cudaError_t.cudaSuccess:
            print(f"Fail deallocating address {address}")
            return False

        del self.address_list[index]
        del self.flag_list[index]
        del self.size_list[index]
        return True

    def reallocate(self, old_address, alignment, new_size):
        """Resize a tracked allocation by allocating/copying/freeing device buffers."""
        if self.log:
            print(f"[CookbookGpuAllocator::reallocate] {old_address=},{alignment=},{new_size=}")
        try:
            index = self.address_list.index(old_address)
        except ValueError:
            print(f"Fail finding address {old_address} in address_list")
            return 0

        if self.flag_list[index] == False:
            print("Old buffer is not resizeable")
            return 0

        if new_size <= self.size_list[index]:  # smaller than the older size
            print("New size is not larger than the old one")
            return old_address

        new_address = self.allocate(new_size, alignment, self.flag_list[index])
        if new_address == 0:
            print("Fail reallocating new buffer")
            return 0

        status = cudart.cudaMemcpy(new_address, old_address, self.size_list[index], cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        if status[0] != cudart.cudaError_t.cudaSuccess:
            print(f"Fail copy old_address from buffer from old buffer {old_address} to new one (new_address)")
            return old_address

        status = self.deallocate(old_address)
        if status == False:
            print(f"Fail deallocating old buffer {old_address}")
            return new_address

        return new_address

class CookbookGpuAsyncAllocator(trt.IGpuAsyncAllocator):
    """GPU allocator implementation that tracks allocation metadata."""

    def __init__(self, log: bool = False):
        """Initialize async allocation tracking and optional logging."""
        super().__init__()
        self.address_list = []
        self.log = log

    def allocate_async(self, size, alignment, flags, stream):
        """Allocate device memory asynchronously and record the pointer."""
        if self.log:
            print(f"[CookbookGpuAsyncAllocator::allocate_async] {size=}, {alignment=}, {flags=}, {stream=}")
        status, address = cudart.cudaMallocAsync(size, stream)
        if status != cudart.cudaError_t.cudaSuccess:
            print(f"Fail allocating {size}B on stream {stream}")
            return 0
        self.address_list.append(address)
        return address

    def deallocate_async(self, memory, stream):
        """Free async-allocated device memory on the specified CUDA stream."""
        if self.log:
            print(f"[CookbookGpuAsyncAllocator::deallocate_async] {memory=}, {stream=}")
        try:
            self.address_list.remove(memory)
        except ValueError:
            pass
        status = cudart.cudaFreeAsync(memory, stream)
        return status == cudart.cudaError_t.cudaSuccess

class CookbookOutputAllocator(trt.IOutputAllocator):
    """Output allocator for data-dependent output shapes at runtime."""

    def __init__(self, log: bool = False) -> None:
        """Initialize output allocator state for DDS outputs."""
        if log:
            print("[CookbookOutputAllocator::__init__]")
        super().__init__()
        # members for outside use
        self.shape = None
        self.n_bytes = 0
        self.address = 0
        self.log = log

    def reallocate_output(self, tensor_name, old_address, size, alignment) -> int:
        """Synchronously reallocate output storage."""
        if self.log:
            print(f"[CookbookOutputAllocator::reallocate_output] {tensor_name=}, {old_address=}, {size=}, {alignment=}")
        return self.reallocate_common(tensor_name, old_address, size, alignment)

    def reallocate_output_async(self, tensor_name, old_address, size, alignment, stream) -> int:
        """Asynchronously reallocate output storage on a CUDA stream."""
        if self.log:
            print(f"[CookbookOutputAllocator::reallocate_output_async] {tensor_name=}, {old_address=}, {size=}, {alignment=}, {stream=}")
        return self.reallocate_common(tensor_name, old_address, size, alignment, stream)

    def notify_shape(self, tensor_name, shape):
        """Receive final runtime shape for a DDS output tensor."""
        if self.log:
            print(f"[CookbookOutputAllocator::notify_shape] {tensor_name=}, {shape=}")
        self.shape = shape
        return

    def reallocate_common(self, tensor_name, old_address, size, alignment, stream=-1):  # not necessary API
        """Internal helper that implements sync/async output reallocation."""
        if size <= self.n_bytes:
            return old_address
        if old_address != 0:
            status = cudart.cudaFree(old_address)
            if status != cudart.cudaError_t.cudaSuccess:
                print(f"Fail freeing {old_address}")
                return 0
        if stream == -1:
            status, address = cudart.cudaMalloc(size)
        else:
            status, address = cudart.cudaMallocAsync(size, stream)
        if status != cudart.cudaError_t.cudaSuccess:
            if self.log:
                print("Fail allocating new buffer")
            return 0
        self.n_bytes = size
        self.address = address
        return address

class CookbookProgressMonitor(trt.IProgressMonitor):
    """Progress monitor that prints hierarchical build phases and steps."""

    def __init__(self, log=False) -> None:
        """Initialize progress monitor tree state."""
        if log:
            print("[CookbookProgressMonitor::__init__]")
        trt.IProgressMonitor.__init__(self)
        self.level = 0
        self.n_step = [0 for _ in range(10)]
        self.log = log

    def phase_start(self, phase_name, parent_phase, num_steps) -> None:
        """Handle start event of a build phase."""
        if self.log:
            print(f"[CookbookProgressMonitor::phase_start]{phase_name=},{parent_phase=},{num_steps=}")
        print("|   " * self.level + f"Start[{phase_name}]:{parent_phase=},{num_steps=}")
        self.level += 1
        self.n_step[self.level] = num_steps
        return

    def phase_finish(self, phase_name) -> None:
        """Handle end event of a build phase."""
        if self.log:
            print(f"[CookbookProgressMonitor::phase_finish]{phase_name=}")
        self.level -= 1
        print("|   " * self.level + f"End  [{phase_name}]")

        return

    def step_complete(self, phase_name, step) -> bool:
        """Handle completion event of one step inside a phase."""
        if self.log:
            print(f"[CookbookProgressMonitor::step_complete]{phase_name=},{step=}")

        head = "└" if step == self.n_step[self.level] - 1 else "├"
        print("|   " * (self.level - 1) + f"{head}   Step [{phase_name}]:{step=}")
        return True

class CookbookStreamWriter(trt.IStreamWriter):
    """Stream writer that writes serialized engine bytes to a file."""

    def __init__(self, file_name: str):
        """Initialize writer with destination file path."""
        super().__init__()
        self.file_name = file_name

    def write(self, buffer: bytes) -> int:
        """Write bytes to file and return number of bytes written."""
        with open(self.file_name, "wb") as f:
            f.write(buffer)
        return len(buffer)

class CookbookStreamReaderV2(trt.IStreamReaderV2):
    """In-memory ``IStreamReaderV2`` adapter for TensorRT deserialization."""

    def __init__(self, bytes):
        """Initialize in-memory stream reader from a bytes object."""
        super().__init__()
        self.bytes = bytes
        self.len = len(bytes)
        self.index = 0

    def read(self, size, cudaStreamPtr):
        """Read ``size`` bytes from current index and advance the cursor."""
        assert self.index + size <= self.len
        data = self.bytes[self.index:self.index + size]
        self.index += size
        return data

    def seek(self, offset, where):
        """Seek read cursor according to TensorRT ``SeekPosition``."""
        if where == trt.SeekPosition.SET:
            self.index = offset
        elif where == trt.SeekPosition.CUR:
            self.index += offset
        elif where == trt.SeekPosition.END:
            self.index = self.len - offset
        else:
            raise ValueError(f"Invalid seek position: {where}")

class TRTWrapperV1:
    """Core TensorRT wrapper that simplifies build/setup/infer workflows."""

    # Just a wrapper for the usage of TensorRT APIs, which can be unpacked back as process programming.
    # We use this for decreasing lines of code in most examples, though increasing complexity for reading.
    # I don't like this style of examples, but it might be huge workload to fix something in all examples.
    # So I'm sorry to present the examples like this :p

    def __init__(
        self,
        *,
        logger: Union[trt.Logger, trt.Logger.Severity, str] | None = None,  # Pass a `trt.Logger` from outside, or a logger level to create it inside
        trt_file: Path | None = None,  # If we already have a TensorRT engine file, just load it rather than build it from scratch.
        plugin_file_list: list[Union[Path, str]] | None = None,  # If we already have some plugins, just load them.
        callback_object_dict: dict | None = None,
    ) -> None:
        """Create a TensorRT wrapper with optional preloaded engine and callbacks."""
        plugin_file_list = plugin_file_list or []
        callback_object_dict = callback_object_dict or {}

        # Create a logger
        if isinstance(logger, trt.Logger):
            self.logger = logger
        elif isinstance(logger, trt.Logger.Severity):
            self.logger = trt.Logger(logger)
        elif isinstance(logger, str):
            self.logger = trt.Logger(text_to_logger_level(logger))
        else:
            self.logger = trt.Logger()

        # Load plugins from file if provided
        load_plugin_files(plugin_file_list, self.logger)

        self.callback_object_dict = callback_object_dict

        # Load engine bytes from file
        if trt_file is not None and trt_file.exists():
            with open(trt_file, "rb") as f:
                self.engine_bytes = f.read()
        else:
            # Build it from scratch
            self.builder = trt.Builder(self.logger)
            self.network = self.builder.create_network()
            self.profile = self.builder.create_optimization_profile()
            self.builder_config = self.builder.create_builder_config()
            self.engine_bytes = None

            self.builder.error_recorder = self.callback_object_dict.get("error_recorder", None)
            self.builder_config.progress_monitor = self.callback_object_dict.get("progress_monitor", None)

        self.runtime = None
        self.engine = None
        self.context = None
        self.stream = 0

        return

    # ================================ Buildtime actions
    def build(self, output_tensor_list: list | None = None, *, extra_profile_list: list | None = None) -> bool:
        """Mark outputs, add optimization profiles, and build serialized engine bytes."""
        output_tensor_list = output_tensor_list or []
        extra_profile_list = extra_profile_list or []

        # Mark output tensors of the network and build engine bytes
        for tensor in output_tensor_list:
            self.network.mark_output(tensor)

        # Add optimization profiles into BuilderConfig, forcing `self.profile` to be the first one
        if False in extra_profile_list:
            # extra_profile_list as [False] is a special signal to skip call of`builder_config.add_optimization_profile`
            pass
        else:
            self.builder_config.add_optimization_profile(self.profile)
            for profile in extra_profile_list:
                self.builder_config.add_optimization_profile(profile)

        self.engine_bytes = self.builder.build_serialized_network(self.network, self.builder_config)
        return self.engine_bytes is not None

    def serialize_engine(self, trt_file: Path, b_remove_old_file: bool = True) -> bool:
        """Save serialized engine bytes to a plan file."""
        # Save engine bytes as TensorRT engine file
        if self.engine_bytes is None:
            print("Fail to serialize engine since engine_bytes is None.")
            return False
        if b_remove_old_file and trt_file.exists():
            trt_file.unlink()
        with open(trt_file, "wb") as f:
            f.write(self.engine_bytes)
        return True

    # ================================ Runtime tool functions
    def _setup_utils(self):
        """Initialize runtime, engine, context, and IO tensor metadata."""
        # Get input data and do preprocess before inference
        if self.runtime is None:  # Just in case we already have an runtime from outside
            self.runtime = trt.Runtime(self.logger)
        if self.engine is None:  # Just in case we already have an engine from outside
            self.engine = self.runtime.deserialize_cuda_engine(self.engine_bytes)
        if self.context is None:  # Just in case we already have an context from outside
            self.context = self.engine.create_execution_context()

        self.runtime.gpu_allocator = self.callback_object_dict.get("gpu_allocator", None)

        self.tensor_name_list = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.n_input = sum([self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT for name in self.tensor_name_list])
        self.n_output = self.engine.num_io_tensors - self.n_input

    def _setup_shape(self, input_data):
        """Set dynamic input shapes from provided input data."""
        for name, data in input_data.items():
            if name not in self.tensor_name_list[:self.n_input]:
                print(f"Skip `{name}` in data map")
                continue
            self.context.set_input_shape(name, data.shape)

        invalid_tensor_name_list = self.context.infer_shapes()
        if len(invalid_tensor_name_list) > 0:
            print(f"Invalid input tensor: {invalid_tensor_name_list}")

    def _setup_print_io_tensors(self):
        """Print engine and context IO tensor information."""
        # Print information of input / output tensors
        for name in self.tensor_name_list:
            mode = self.engine.get_tensor_mode(name)
            data_type = self.engine.get_tensor_dtype(name)
            buildtime_shape = self.engine.get_tensor_shape(name)
            runtime_shape = self.context.get_tensor_shape(name)
            print(f"{'Input ' if mode == trt.TensorIOMode.INPUT else 'Output'}->{data_type}, {buildtime_shape}, {runtime_shape}, {name}")

    def _setup_buffer(self, input_data):
        """Allocate host/device buffers and bind tensor addresses."""
        # Prepare work before inference
        self.buffer = OrderedDict()
        for name in self.tensor_name_list:
            data_type = self.engine.get_tensor_dtype(name)
            runtime_shape = self.context.get_tensor_shape(name)
            n_byte = trt.volume(runtime_shape) * data_type.itemsize
            host_buffer = np.empty(runtime_shape, dtype=datatype_cast(data_type, "np"))
            # `cudaMalloc(0)` succeeds but returns a NULL address, and binding NULL to a tensor makes
            # `enqueueV3` refuse to run (it only says so through its return value). A zero-volume
            # tensor is normal -- an empty batch, a detector that found nothing -- so give it a byte.
            device_buffer = cudart.cudaMalloc(max(n_byte, 1))[1]
            self.buffer[name] = [host_buffer, device_buffer, n_byte]

        for name, data in input_data.items():
            self.buffer[name][0] = np.ascontiguousarray(data)

        for name in self.tensor_name_list:
            self.context.set_tensor_address(name, self.buffer[name][1])

    # ================================ Runtime actions
    def setup(self, input_data: dict | None = None, *, b_print_io: bool = True) -> None:
        """Prepare runtime resources, shapes, and buffers before inference."""
        input_data = input_data or {}
        # Get input data and do preprocess before inference
        self._setup_utils()

        self._setup_shape(input_data)

        if b_print_io:
            self._setup_print_io_tensors()

        self._setup_buffer(input_data)

        return

    def infer(self, *, b_print_io: bool = True, stream: int = 0, b_get_timeline: bool = False) -> None:
        """Run one inference pass and optionally print outputs and timeline markers."""
        # Update customized CUDA stream if provided
        if stream != 0:
            self.stream = stream

        # Memory copy from host to device
        for name in self.tensor_name_list:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                cudart.cudaMemcpyAsync(self.buffer[name][1], self.buffer[name][0].ctypes.data, self.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)

        # Do inference
        self.context.execute_async_v3(self.stream)

        # Do more inference if we want to get a timeline
        if b_get_timeline:
            for _ in range(10):  # warm up
                self.context.execute_async_v3(self.stream)
            cudart.cudaStreamSynchronize(self.stream)
            for _ in range(30):
                with nvtx.annotate("Inference", color="green"):
                    self.context.execute_async_v3(self.stream)
            cudart.cudaStreamSynchronize(self.stream)

        # Memory copy from device to host
        for name in self.tensor_name_list:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                cudart.cudaMemcpyAsync(self.buffer[name][0].ctypes.data, self.buffer[name][1], self.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)

        cudart.cudaStreamSynchronize(self.stream)

        # Print output
        if b_print_io:
            for name in self.tensor_name_list:
                print(name)
                print(self.buffer[name][0])

        return

    def __del__(self):
        """Try best to release device buffers allocated by this wrapper."""
        # free_plugin_files()
        # Free device memory
        if hasattr(self, "buffer") and self.buffer is not None and len(self.buffer) > 0:
            for _, device_buffer, _ in self.buffer.values():
                if device_buffer not in (None, 0):
                    try:
                        cudart.cudaFree(device_buffer)
                    except Exception:
                        pass
        return

class TRTWrapperDDS(TRTWrapperV1):
    """Wrapper variant for data-dependent-shape outputs using output allocators."""

    # Override for Data-dependent-Shape (DDS) mode
    # TRTWrapperDDS = TRTWrapperV1 + CookbookOutputAllocator

    def __init__(
        self,
        *,
        logger: Union[trt.Logger, trt.Logger.Severity, str] = None,
        trt_file: Path = None,
        plugin_file_list: list | None = None,
        callback_object_dict: dict | None = None,
    ) -> None:
        """Initialize DDS wrapper using ``TRTWrapperV1`` base configuration."""
        TRTWrapperV1.__init__(
            self,
            logger=logger,
            trt_file=trt_file,
            plugin_file_list=plugin_file_list,
            callback_object_dict=callback_object_dict,
        )

    # ================================ Runtime tool functions
    def _setup_buffer_dds(self, input_data):
        """Allocate buffers and output allocators for DDS-capable execution."""
        # Prepare work before inference
        self.buffer = OrderedDict()
        self.output_allocator_map = OrderedDict()
        for name in self.tensor_name_list:
            data_type = self.engine.get_tensor_dtype(name)
            runtime_shape = self.context.get_tensor_shape(name)
            if -1 in runtime_shape:  # for Data-Dependent-Shape (DDS) output, "else" branch for normal output
                n_byte = 0  # self.context.get_max_output_size(name)
                self.output_allocator_map[name] = CookbookOutputAllocator()
                self.context.set_output_allocator(name, self.output_allocator_map[name])
                host_buffer = np.empty(0, dtype=trt.nptype(data_type))
                device_buffer = 0
            else:
                n_byte = trt.volume(runtime_shape) * data_type.itemsize
                host_buffer = np.empty(runtime_shape, dtype=datatype_cast(data_type, "np"))
                # `cudaMalloc(0)` succeeds but returns a NULL address, and binding NULL to a tensor makes
                # `enqueueV3` refuse to run (it only says so through its return value). A zero-volume
                # tensor is normal -- an empty batch, a detector that found nothing -- so give it a byte.
                device_buffer = cudart.cudaMalloc(max(n_byte, 1))[1]
            self.buffer[name] = [host_buffer, device_buffer, n_byte]

        for name, data in input_data.items():
            self.buffer[name][0] = np.ascontiguousarray(data)

        for name in self.tensor_name_list:
            self.context.set_tensor_address(name, self.buffer[name][1])

    # ================================ Runtime actions
    def setup(self, input_data: dict | None = None, *, b_print_io: bool = True) -> None:
        """Prepare DDS runtime resources, shapes, and buffers."""
        input_data = input_data or {}
        # Get input data and do preprocess before inference
        self._setup_utils()

        self._setup_shape(input_data)

        if b_print_io:
            self._setup_print_io_tensors()

        self._setup_buffer_dds(input_data)

        return

    def infer(self, *, b_print_io: bool = True, stream: int = 0, b_get_timeline: bool = False) -> None:
        """Run DDS inference and materialize dynamic outputs."""
        # Update customized CUDA stream if provided
        if stream != 0:
            self.stream = stream

        # Memory copy from host to device
        for name in self.tensor_name_list:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                cudart.cudaMemcpyAsync(self.buffer[name][1], self.buffer[name][0].ctypes.data, self.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)

        # Do inference
        self.context.execute_async_v3(self.stream)

        # Do more inference if we want to get a timeline
        if b_get_timeline:
            for _ in range(10):  # warm up
                self.context.execute_async_v3(self.stream)
            cudart.cudaStreamSynchronize(self.stream)
            with nvtx.annotate("Inference", color="green"):
                self.context.execute_async_v3(self.stream)
            cudart.cudaStreamSynchronize(self.stream)

        # Get output shape from OutputAllocator
        for name in self.tensor_name_list:
            if -1 in self.context.get_tensor_shape(name):
                myOutputAllocator = self.context.get_output_allocator(name)
                runtime_shape = myOutputAllocator.shape
                data_type = self.engine.get_tensor_dtype(name)
                host_buffer = np.empty(runtime_shape, dtype=datatype_cast(data_type, "np"))
                device_buffer = myOutputAllocator.address
                n_bytes = trt.volume(runtime_shape) * data_type.itemsize
                self.buffer[name] = [host_buffer, device_buffer, n_bytes]

        # Memory copy from device to host
        for name in self.tensor_name_list:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                cudart.cudaMemcpyAsync(self.buffer[name][0].ctypes.data, self.buffer[name][1], self.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)

        cudart.cudaStreamSynchronize(self.stream)

        # Print output
        if b_print_io:
            for name in self.tensor_name_list:
                print(name)
                print(self.buffer[name][0])

        return

class TRTWrapperShapeInput(TRTWrapperV1):
    """Wrapper variant for networks that use shape input tensors."""

    # Override for model with Shape-Input-Tensor
    # There 5 differences during `setup()` and `infer()`, see the code below

    def __init__(
        self,
        *,
        logger: Union[trt.Logger, trt.Logger.Severity, str] = None,
        trt_file: Path = None,
        plugin_file_list: list | None = None,
        callback_object_dict: dict | None = None,
    ) -> None:
        """Initialize shape-input wrapper using ``TRTWrapperV1`` base configuration."""
        TRTWrapperV1.__init__(
            self,
            logger=logger,
            trt_file=trt_file,
            plugin_file_list=plugin_file_list,
            callback_object_dict=callback_object_dict,
        )

    # ================================ Runtime tool functions
    def _setup_shape_si(self, input_data):
        """Bind shape-input tensors and dynamic input shapes."""
        for name, data in input_data.items():
            if name not in self.tensor_name_list[:self.n_input]:
                print(f"Skip `{name}` in input data")
                continue
            # Key difference, use `set_tensor_address()` instead of `set_input_shape()` for shape input tensor
            if self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                self.context.set_input_shape(name, data.shape)
            else:
                self.context.set_tensor_address(name, data.ctypes.data)

        invalid_tensor_name_list = self.context.infer_shapes()
        if len(invalid_tensor_name_list) > 0:
            print(f"Invalid input tensor: {invalid_tensor_name_list}")

    def _setup_buffer_si(self, input_data):
        """Allocate/bind buffers while handling host-resident shape tensors."""
        # Prepare work before inference
        self.buffer = OrderedDict()
        for name in self.tensor_name_list:
            data_type = self.engine.get_tensor_dtype(name)
            runtime_shape = self.context.get_tensor_shape(name)
            n_byte = trt.volume(runtime_shape) * data_type.itemsize
            host_buffer = np.empty(runtime_shape, dtype=datatype_cast(data_type, "np"))
            # Key difference, no need to allocate device buffer for shape tensor
            if self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                # `cudaMalloc(0)` succeeds but returns a NULL address, and binding NULL to a tensor makes
                # `enqueueV3` refuse to run (it only says so through its return value). A zero-volume
                # tensor is normal -- an empty batch, a detector that found nothing -- so give it a byte.
                device_buffer = cudart.cudaMalloc(max(n_byte, 1))[1]
            else:
                device_buffer = None
            self.buffer[name] = [host_buffer, device_buffer, n_byte]

        for name, data in input_data.items():
            self.buffer[name][0] = np.ascontiguousarray(data)

        for name in self.tensor_name_list:
            # Key difference, we have called `set_tensor_address()` for shape input tensors before
            if self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                self.context.set_tensor_address(name, self.buffer[name][1])
            elif self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.context.set_tensor_address(name, self.buffer[name][0].ctypes.data)

    # ================================ Runtime actions
    def setup(self, input_data: dict | None = None, *, b_print_io: bool = True) -> None:
        """Prepare resources for networks containing shape-input tensors."""
        input_data = input_data or {}
        # Get input data and do preprocess before inference
        self._setup_utils()

        self._setup_shape_si(input_data)

        if b_print_io:
            self._setup_print_io_tensors()

        self._setup_buffer_si(input_data)

        return

    def infer(self, *, b_print_io: bool = True, stream: int = 0, b_get_timeline: bool = False) -> None:
        """Run inference for shape-input networks and fetch outputs."""
        # Update customized CUDA stream if provided
        if stream != 0:
            self.stream = stream

        # Memory copy from host to device
        for name in self.tensor_name_list:
            # Key difference, need not to copy shape tensor buffer between CPU and GPU
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT and self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                cudart.cudaMemcpyAsync(self.buffer[name][1], self.buffer[name][0].ctypes.data, self.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)

        # Do inference
        self.context.execute_async_v3(self.stream)

        # Do more inference if we want to get a timeline
        if b_get_timeline:
            for _ in range(10):  # warm up
                self.context.execute_async_v3(self.stream)
            cudart.cudaStreamSynchronize(self.stream)
            with nvtx.annotate("Inference", color="green"):
                self.context.execute_async_v3(self.stream)
            cudart.cudaStreamSynchronize(self.stream)

        # Memory copy from device to host
        for name in self.tensor_name_list:
            # Key difference, need not to copy shape tensor buffer between CPU and GPU
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT and self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                cudart.cudaMemcpyAsync(self.buffer[name][0].ctypes.data, self.buffer[name][1], self.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)

        cudart.cudaStreamSynchronize(self.stream)

        # Print output
        if b_print_io:
            for name in self.tensor_name_list:
                print(name)
                print(self.buffer[name][0])

        # Free device memory
        for _, device_buffer, _ in self.buffer.values():
            cudart.cudaFree(device_buffer)

        return

class TRTWrapperV2(TRTWrapperDDS, TRTWrapperShapeInput):
    """Combined wrapper supporting both DDS outputs and shape-input workflows."""

    # TRTWrapperV2 = TRTWrapperV1 + TRTWrapperDDS + TRTWrapperShapeInput, pretty complex

    def __init__(
        self,
        *,
        logger: Union[trt.Logger, trt.Logger.Severity, str] = None,
        trt_file: Path = None,
        plugin_file_list: list | None = None,
        callback_object_dict: dict | None = None,
    ) -> None:
        """Initialize combined DDS + shape-input wrapper."""
        TRTWrapperV1.__init__(
            self,
            logger=logger,
            trt_file=trt_file,
            plugin_file_list=plugin_file_list,
            callback_object_dict=callback_object_dict,
        )

    def setup(self, input_data: dict | None = None, *, b_print_io: bool = True) -> None:
        """Prepare buffers for combined DDS and shape-input execution."""
        input_data = input_data or {}
        # Get input data and do preprocess before inference
        self._setup_utils()

        self._setup_shape_si(input_data)

        if b_print_io:
            self._setup_print_io_tensors()

        # Prepare work before inference - combine DDS and ShapeInput
        self.buffer = OrderedDict()
        self.output_allocator_map = OrderedDict()
        for name in self.tensor_name_list:
            data_type = self.engine.get_tensor_dtype(name)
            runtime_shape = self.context.get_tensor_shape(name)
            if -1 in runtime_shape:  # for Data-Dependent-Shape (DDS) output, "else" branch for normal output
                n_byte = 0  # self.context.get_max_output_size(name)
                self.output_allocator_map[name] = CookbookOutputAllocator()
                self.context.set_output_allocator(name, self.output_allocator_map[name])
                host_buffer = np.empty(0, dtype=trt.nptype(data_type))
                device_buffer = 0
            else:
                n_byte = trt.volume(runtime_shape) * data_type.itemsize
                host_buffer = np.empty(runtime_shape, dtype=datatype_cast(data_type, "np"))
                if self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                    # `cudaMalloc(0)` succeeds but returns a NULL address, and binding NULL to a tensor makes
                    # `enqueueV3` refuse to run (it only says so through its return value). A zero-volume
                    # tensor is normal -- an empty batch, a detector that found nothing -- so give it a byte.
                    device_buffer = cudart.cudaMalloc(max(n_byte, 1))[1]
                else:
                    device_buffer = None
            self.buffer[name] = [host_buffer, device_buffer, n_byte]

        for name, data in input_data.items():
            self.buffer[name][0] = np.ascontiguousarray(data)

        for name in self.tensor_name_list:
            if self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                self.context.set_tensor_address(name, self.buffer[name][1])
            elif self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.context.set_tensor_address(name, self.buffer[name][0].ctypes.data)

        return

    def infer(self, *, b_print_io: bool = True, stream: int = 0, b_get_timeline: bool = False) -> None:
        """Run combined DDS/shape-input inference and collect outputs."""
        # Update customized CUDA stream if provided
        if stream != 0:
            self.stream = stream
        for name in self.tensor_name_list:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT and self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                cudart.cudaMemcpyAsync(self.buffer[name][1], self.buffer[name][0].ctypes.data, self.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)

        # Do inference
        self.context.execute_async_v3(self.stream)

        # Do more inference if we want to get a timeline
        if b_get_timeline:
            for _ in range(10):  # warm up
                self.context.execute_async_v3(self.stream)
            cudart.cudaStreamSynchronize(self.stream)
            with nvtx.annotate("Inference", color="green"):
                self.context.execute_async_v3(self.stream)
            cudart.cudaStreamSynchronize(self.stream)

        # Memory copy from device to host
        for name in self.tensor_name_list:
            if -1 in self.context.get_tensor_shape(name):
                myOutputAllocator = self.context.get_output_allocator(name)
                runtime_shape = myOutputAllocator.shape
                data_type = self.engine.get_tensor_dtype(name)
                host_buffer = np.empty(runtime_shape, dtype=datatype_cast(data_type, "np"))
                device_buffer = myOutputAllocator.address
                n_bytes = trt.volume(runtime_shape) * data_type.itemsize
                self.buffer[name] = [host_buffer, device_buffer, n_bytes]

        for name in self.tensor_name_list:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT and self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                cudart.cudaMemcpyAsync(self.buffer[name][0].ctypes.data, self.buffer[name][1], self.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)

        cudart.cudaStreamSynchronize(self.stream)

        # Print output
        if b_print_io:
            for name in self.tensor_name_list:
                print(name)
                print(self.buffer[name][0])

        return

class TRTWrapperV2Torch(TRTWrapperDDS, TRTWrapperShapeInput):
    """Torch-buffer variant of ``TRTWrapperV2`` for GPU tensor interoperability."""

    # TRTWrapperV2Torch = TRTWrapperV2 using pyTorch API

    def __init__(
        self,
        *,
        logger: Union[trt.Logger, trt.Logger.Severity, str] = None,
        trt_file: Path = None,
        plugin_file_list: list | None = None,
        callback_object_dict: dict | None = None,
    ) -> None:
        """Initialize Torch-based combined wrapper."""
        TRTWrapperV1.__init__(
            self,
            logger=logger,
            trt_file=trt_file,
            plugin_file_list=plugin_file_list,
            callback_object_dict=callback_object_dict,
        )

    def setup(self, input_data: dict | None = None, *, b_print_io: bool = True) -> None:
        """Prepare Torch tensors and bindings for inference."""
        input_data = input_data or {}
        # Get input data and do preprocess before inference
        self._setup_utils()

        self._setup_shape_si(input_data)

        if b_print_io:
            self._setup_print_io_tensors()

        # Prepare work before inference - use torch rather than numpy
        self.buffer = OrderedDict()
        self.output_allocator_map = OrderedDict()
        for name in self.tensor_name_list:
            data_type = self.engine.get_tensor_dtype(name)
            runtime_shape = self.context.get_tensor_shape(name)
            if -1 in runtime_shape:  # for Data-Dependent-Shape (DDS) output, "else" branch for normal output
                n_byte = 0  # self.context.get_max_output_size(name)
                self.output_allocator_map[name] = CookbookOutputAllocator()
                self.context.set_output_allocator(name, self.output_allocator_map[name])
                buffer = torch.empty(0, dtype=datatype_cast(data_type, "torch")).cuda()
            else:
                buffer = torch.empty(tuple(runtime_shape), dtype=datatype_cast(data_type, "torch")).cuda()
            self.buffer[name] = buffer

        for name, data in input_data.items():
            if self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                self.buffer[name] = torch.Tensor(np.array(data)).contiguous().cuda()
            else:
                self.buffer[name] = torch.Tensor(np.array(data)).contiguous()

        for name in self.tensor_name_list:
            if self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                self.context.set_tensor_address(name, self.buffer[name].data_ptr())
            elif self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.context.set_tensor_address(name, self.buffer[name].data_ptr())

        return

    def infer(self, *, b_print_io: bool = True, stream: int = 0, b_get_timeline: bool = False) -> None:
        """Run inference and keep outputs in Torch-friendly buffers."""
        # Update customized CUDA stream if provided
        if stream != 0:
            self.stream = stream

        # Do inference
        self.context.execute_async_v3(self.stream)

        # Do more inference if we want to get a timeline
        if b_get_timeline:
            for _ in range(10):  # warm up
                self.context.execute_async_v3(self.stream)
            cudart.cudaStreamSynchronize(self.stream)
            with nvtx.annotate("Inference", color="green"):
                self.context.execute_async_v3(self.stream)
            cudart.cudaStreamSynchronize(self.stream)

        # Memory copy from device to host
        for name in self.tensor_name_list:
            if -1 in self.context.get_tensor_shape(name):
                myOutputAllocator = self.context.get_output_allocator(name)
                runtime_shape = myOutputAllocator.shape
                data_type = self.engine.get_tensor_dtype(name)
                device_buffer = myOutputAllocator.address
                n_bytes = trt.volume(runtime_shape) * data_type.itemsize
                # TODO: construct a tensor in-place
                tensor = torch.empty(tuple(runtime_shape), dtype=datatype_cast(data_type, "torch"), device='cuda')
                cudart.cudaMemcpyAsync(tensor.data_ptr(), device_buffer, n_bytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, self.stream)
                self.buffer[name] = tensor.cpu()

        cudart.cudaStreamSynchronize(self.stream)

        # Print output
        if b_print_io:
            for name in self.tensor_name_list:
                print(name)
                print(self.buffer[name])

    def __del__(self):
        """Destructor placeholder; Torch manages buffer lifetime."""
        pass  # cudaFree is not needed in pyTorch

        return
