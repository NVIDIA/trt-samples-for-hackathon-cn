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
#

import ctypes
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import nvtx
import tensorrt as trt
import torch
from cuda.bindings import runtime as cudart

from .utils_function import (byte_to_string, datatype_trt_to_str, datatype_trt_to_torch, print_array_information, text_to_logger_level)

class CookbookLogger(trt.ILogger):

    def __init__(self, min_severity=trt.ILogger.Severity.INTERNAL_ERROR) -> None:
        trt.ILogger.__init__(self)
        # int(trt.ILogger.Severity.INTERNAL_ERROR) == 0
        # int(trt.ILogger.Severity.ERROR) == 1
        # int(trt.ILogger.Severity.WARNING) == 2
        # int(trt.ILogger.Severity.INFO) == 3
        # int(trt.ILogger.Severity.VERBOSE) == 4
        self.min_severity = min_severity

    def log(self, severity, msg) -> None:
        if severity <= self.min_severity:
            print(f"[My Logger] {msg}")  # customerized log content

class CookbookProfiler(trt.IProfiler):

    def __init__(self) -> None:
        super().__init__()

    def report_layer_time(self, layer_name, time_ms) -> None:
        print(f"Timing: {time_ms * 1000: 8.3f}us -> {layer_name}")

class CookbookDebugListener(trt.IDebugListener):  # `trt.IDebugListener` since TensorRT-10.0
    # implement a call back class to get information of the debug tensors

    def __init__(self, expect_result: dict = {}, epsilon: float = 1e-5, log: bool = False):
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
        host_buffer = np.empty(tuple(shape), dtype=trt.nptype(type))
        if location == trt.TensorLocation.DEVICE:
            cudart.cudaStreamSynchronize(stream)  # might be removed in the future
            cudart.cudaMemcpyAsync(host_buffer.ctypes.data, addr, host_buffer.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
            cudart.cudaStreamSynchronize(stream)
        else:
            host_buffer.ctypes.data = addr

        # we can print information from `host_buffer` here
        print_array_information(host_buffer, info=name)

        # Compare host_buffer with optional expected result
        if name in self.expect_result.keys():
            diff = np.max(np.abs(host_buffer - self.expect_result[name])) < self.epsilon
            print(f"#### Check debug tensor {name}: {diff} ####")  # print result by print for assert or anything else,

        return True  # return value does not reflect the check

class CookbookErrorRecorder(trt.IErrorRecorder):

    def __init__(self, log: bool = False) -> None:
        if log:
            print("[CookbookErrorRecorder::__init__]")
        super().__init__()
        self.error_list = []
        self.n_max_error = 256
        self.log = log

    def clear(self) -> None:
        if self.log:
            print("[CookbookErrorRecorder::clear]")
        self.error_list = []
        return None

    def get_error_code(self, index) -> int:
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
        if self.log:
            print(f"[CookbookErrorRecorder::get_error_desc] {index=}")
        if index < 0 or index >= len(self.error_list):
            print("Error index")
            return ""
        return self.error_list[index][1]

    def has_overflowed(self) -> bool:
        if self.log:
            print("[CookbookErrorRecorder::has_overflowed]")
        return len(self.error_list) >= self.n_max_error

    def num_errors(self) -> int:
        if self.log:
            print("[CookbookErrorRecorder::num_errors]")
        return len(self.error_list)

    def report_error(self, error_code, error_description) -> None:
        print(f"[CookbookErrorRecorder::report_error]\n    n={len(self.error_list)},code={error_code},info={error_description}")
        self.error_list.append([error_code, error_description])
        if self.has_overflowed():
            print("Error Overflow")
        return

    def hello_world(self) -> str:  # not necessary API
        return str(id(self))

class CookbookGpuAllocator(trt.IGpuAllocator):

    def __init__(self, log: bool = False):
        if log:
            print("[CookbookGpuAllocator::__init__]")
        super().__init__()
        self.address_list = []
        self.flag_list = []
        self.size_list = []
        self.log = log

    def allocate(self, size, alignment, flag):
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
        if self.log:
            print(f"[CookbookGpuAllocator::deallocate] {address=}")
        try:
            index = self.address_list.index(address)
        except:
            print(f"Fail finding address {address} in address_list")
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
        if self.log:
            print(f"[CookbookGpuAllocator::reallocate] {old_address=},{alignment=},{new_size=}")
        try:
            index = self.address_list.index(old_address)
        except:
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

class CookbookOutputAllocator(trt.IOutputAllocator):

    def __init__(self, log: bool = False) -> None:
        if log:
            print("[CookbookOutputAllocator::__init__]")
        super().__init__()
        # members for outside use
        self.shape = None
        self.n_bytes = 0
        self.address = 0
        self.log = log

    def reallocate_output(self, tensor_name, old_address, size, alignment) -> int:
        if self.log:
            print(f"[CookbookOutputAllocator::reallocate_output] {tensor_name=}, {old_address=}, {size=}, {alignment=}")
        return self.reallocate_common(tensor_name, old_address, size, alignment)

    def reallocate_output_async(self, tensor_name, old_address, size, alignment, stream) -> int:
        if self.log:
            print(f"[CookbookOutputAllocator::reallocate_output_async] {tensor_name=}, {old_address=}, {size=}, {alignment=}, {stream=}")
        return self.reallocate_common(tensor_name, old_address, size, alignment, stream)

    def notify_shape(self, tensor_name, shape):
        if self.log:
            print(f"[CookbookOutputAllocator::notify_shape] {tensor_name=}, {shape=}")
        self.shape = shape
        return

    def reallocate_common(self, tensor_name, old_address, size, alignment, stream=-1):  # not necessary API
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

class CookbookAlgorithmSelector(trt.IAlgorithmSelector):

    def __init__(self, i_strategy=0, log=False) -> None:  # Pass a number on behalf of our customerized strategy to select algorithm
        if log:
            print("[CookbookAlgorithmSelector::__init__]")
        super().__init__()
        self.i_strategy = i_strategy
        self.log = log

    def select_algorithms(self, layerAlgorithmContext, layerAlgorithmList) -> List[int]:
        if self.log:
            print("[CookbookAlgorithmSelector::select_algorithms]")
        # we print the alternative algorithms of each layer here
        nInput = layerAlgorithmContext.num_inputs
        nOutput = layerAlgorithmContext.num_outputs
        print(f"Layer {layerAlgorithmContext.name}, {nInput=}, {nOutput=}")
        for i in range(nInput + nOutput):
            info = f"    {'Input ' if i < nInput else 'Output'}     {i if i < nInput else i - nInput: 2d}:"
            info += f"shape={layerAlgorithmContext.get_shape(i)}"
            print(info)

        for i, algorithm in enumerate(layerAlgorithmList):
            info = f"    algorithm{i:4d}:"
            info += f"implementation[{algorithm.algorithm_variant.implementation: 10d}],"
            info += f"tactic[{algorithm.algorithm_variant.tactic: 20d}],"
            info += f"timing[{algorithm.timing_msec * 1000: 7.3f}us],"
            info += f"workspace[{byte_to_string(algorithm.workspace_size)}]"
            for j in range(nInput + nOutput):
                io_info = algorithm.get_algorithm_io_info(j)
                info += f"\n                  {'Input ' if j < nInput else 'Output'}{j if j < nInput else j - nInput: 2d}:"
                info += f"datatype={datatype_trt_to_str(io_info.dtype)},"
                info += f"stride={io_info.strides},"
                info += f"vectorized_dim={io_info.vectorized_dim},"
                info += f"components_per_element={io_info.components_per_element}"

            print(info)

        if self.i_strategy == 0:  # choose the algorithm with shortest time, TensorRT default strategy
            timeList = [algorithm.timing_msec for algorithm in layerAlgorithmList]
            result = [np.argmin(timeList)]

        elif self.i_strategy == 1:  # choose the algorithm with longest time, to get a TensorRT engine with worst performance, just for fun :)
            timeList = [algorithm.timing_msec for algorithm in layerAlgorithmList]
            result = [np.argmax(timeList)]

        elif self.i_strategy == 2:  # choose the algorithm using smallest workspace
            workspaceSizeList = [algorithm.workspace_size for algorithm in layerAlgorithmList]
            result = [np.argmin(workspaceSizeList)]

        elif self.i_strategy == 3:  # choose one certain algorithm we have known
            # This strategy can be a workaround for building the exactly same engine, though Timing-Cache is more recommended to do so.
            # The reason is that function select_algorithms is called after the performance test of all algorithms of a layer (you can notice algorithm.timing_msec > 0), so it will not save the time of the test.
            # On the contrary, performance test of the algorithms will be skipped using Timing-Cache, which surely saves a lot of time comparing with Algorithm Selector.
            if layerAlgorithmContext.name == "Convolution1 + Activation1":
                # the number 2147483648 is from VERBOSE log, marking the certain algorithm
                result = [index for index, algorithm in enumerate(layerAlgorithmList) \
                    if algorithm.algorithm_variant.implementation == 2147483657 and algorithm.algorithm_variant.tactic == 6767548733843469815]
            else:  # keep all algorithms for other layers
                result = list(range(len(layerAlgorithmList)))

        else:  # default behavior: keep all algorithms
            result = list(range(len(layerAlgorithmList)))

        return result

    def report_algorithms(self, modelAlgorithmContext, modelAlgorithmList) -> None:  # report the tactic of the whole network
        # some bug in report_algorithms to make the algorithm.timing_msec and algorithm.workspace_size are always 0?
        if self.log:
            print("[CookbookAlgorithmSelector::report_algorithms]")
        for i in range(len(modelAlgorithmContext)):
            context = modelAlgorithmContext[i]
            algorithm = modelAlgorithmList[i]
            nInput = context.num_inputs
            nOutput = context.num_outputs
            print(f"Layer {context.name}, {nInput=}, {nOutput=}")

            info = f"    algorithm    :"
            info += f"implementation[{algorithm.algorithm_variant.implementation: 10d}],"
            info += f"tactic[{algorithm.algorithm_variant.tactic: 20d}],"
            info += f"timing[{algorithm.timing_msec * 1000: 7.3f}us],"
            info += f"workspace[{byte_to_string(algorithm.workspace_size)}]"
            for j in range(nInput + nOutput):
                io_info = algorithm.get_algorithm_io_info(j)
                info += f"\n                  {'Input ' if j < nInput else 'Output'}{j if j < nInput else j - nInput: 2d}:"
                info += f"datatype={datatype_trt_to_str(io_info.dtype)},"
                info += f"stride={io_info.strides},"
                info += f"vectorized_dim={io_info.vectorized_dim},"
                info += f"components_per_element={io_info.components_per_element}"
            print(info)
        return

class CookbookProgressMonitor(trt.IProgressMonitor):

    def __init__(self, log=False) -> None:
        if log:
            print("[CookbookProgressMonitor::__init__]")
        trt.IProgressMonitor.__init__(self)
        self.level = 0
        self.n_step = [0 for _ in range(10)]
        self.log = log

    def phase_start(self, phase_name, parent_phase, num_steps) -> None:
        if self.log:
            print(f"[CookbookProgressMonitor::phase_start]{phase_name=},{parent_phase=},{num_steps=}")
        print("|   " * self.level + f"Start[{phase_name}]:{parent_phase=},{num_steps=}")
        self.level += 1
        self.n_step[self.level] = num_steps
        return

    def phase_finish(self, phase_name) -> None:
        if self.log:
            print(f"[CookbookProgressMonitor::phase_finish]{phase_name=}")
        self.level -= 1
        print("|   " * self.level + f"End  [{phase_name}]")

        return

    def step_complete(self, phase_name, step) -> bool:
        if self.log:
            print(f"[CookbookProgressMonitor::step_complete]{phase_name=},{step=}")

        head = "└" if step == self.n_step[self.level] - 1 else "├"
        print("|   " * (self.level - 1) + f"{head}   Step [{phase_name}]:{step=}")
        return True

class CookbookStreamWriter(trt.IStreamWriter):

    def __init__(self, file_name: str):
        super().__init__()
        self.file_name = file_name

    def write(self, buffer: bytes) -> int:
        with open(self.file_name, "wb") as f:
            f.write(buffer)
        return len(buffer)

class CookbookStreamReader(trt.IStreamReader):

    def __init__(self, file_name: str):
        super().__init__()
        self.file_name = file_name

    def read(self, buffer: bytes) -> int:
        with open(self.file_name, "rb") as f:
            buffer = f.read(buffer)
        return buffer

class CookbookStreamReaderV2(trt.IStreamReaderV2):

    def __init__(self, bytes):
        super().__init__()
        self.bytes = bytes
        self.len = len(bytes)
        self.index = 0

    def read(self, size, cudaStreamPtr):
        assert self.index + size <= self.len
        data = self.bytes[self.index:self.index + size]
        self.index += size
        return data

    def seek(self, offset, where):
        if where == trt.SeekPosition.SET:
            self.index = offset
        elif where == trt.SeekPosition.CUR:
            self.index += offset
        elif where == trt.SeekPosition.END:
            self.index = self.len - offset
        else:
            raise ValueError(f"Invalid seek position: {where}")

class CookbookCalibratorV1(trt.IInt8EntropyCalibrator2):  # only for one-input-network, need refactor

    def __init__(self, n_epoch: int = 1, input_shape: list = [], cache_file: Path = None) -> None:
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.n_epoch = n_epoch
        self.shape = input_shape
        self.cache_file = cache_file
        self.buffer_size = trt.volume(input_shape) * trt.float32.itemsize
        _, self.dIn = cudart.cudaMalloc(self.buffer_size)
        self.count = 0

    def __del__(self) -> None:
        cudart.cudaFree(self.dIn)

    def get_batch_size(self) -> int:  # necessary API
        return self.shape[0]

    def get_batch(self, nameList=None, inputNodeName=None) -> List[int]:  # necessary API
        if self.count < self.n_epoch:
            self.count += 1
            n_element = np.prod(self.shape)
            data = np.random.rand(n_element).astype(np.float32).reshape(*self.shape)
            data = data * n_element * 2 - n_element  # fake normalization
            data = np.ascontiguousarray(data)
            cudart.cudaMemcpy(self.dIn, data.ctypes.data, self.buffer_size, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            return [int(self.dIn)]
        else:
            return None

    def read_calibration_cache(self) -> bytes:  # necessary API
        if self.cache_file.exists():
            print(f"Succeed finding int8 cache file {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                cache = f.read()
                return cache
        else:
            print(f"Fail finding int8 cache file {self.cache_file}")
            return

    def write_calibration_cache(self, cache) -> None:  # necessary API
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"Succeed saving int8 cache file {self.cache_file}")
        return

class CookbookCalibratorMNIST(trt.IInt8EntropyCalibrator2):

    def __init__(
        self,
        input_info: Dict[str, list] = {},
        dataset_path: Path = None,
        int8_cache_file: Path = None,
        is_random_choose: bool = False,
        batch_size: int = 1,
        log: bool = False,
    ) -> None:
        if log:
            print("[CookbookCalibratorMNIST::__init__]")
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.input_info = input_info
        self.dataset = np.load(dataset_path)
        self.int8_cache_file = int8_cache_file
        self.is_random_choose = is_random_choose
        self.batch_size = batch_size
        self.log = log

        self.buffer = {}
        self.max_batch = self.dataset.shape[0]
        self.max_count = (self.max_batch + self.batch_size - 1) // self.batch_size
        self.count = 0
        for name, [dtype, shape] in self.input_info.items():
            buffer_size = dtype.itemsize * np.prod(shape)
            buffer = cudart.cudaMalloc(buffer_size)[1]
            self.buffer[name] = buffer

    def __del__(self) -> None:
        if self.log:
            print("[CookbookCalibratorMNIST::__del__]")
        for name, buffer in self.buffer.items():
            cudart.cudaFree(buffer)

    def get_batch_size(self) -> int:  # necessary API
        if self.log:
            print("[CookbookCalibratorMNIST::get_batch_size]")
        return self.batch_size

    def get_batch(self, names: List[str]) -> List[int]:  # necessary API
        if self.log:
            print(f"[CookbookCalibratorMNIST::get_batch]{self.count:3d}/{self.max_count:3d}")
        output_list = []
        if self.count < self.max_count:
            for name in names:
                if self.is_random_choose:
                    index = np.random.randint(0, self.max_batch, self.batch_size)
                else:
                    low_bound = self.count * self.batch_size
                    high_bound = low_bound + self.batch_size
                    if high_bound >= self.max_batch:
                        low_bound = self.max_batch - self.batch_size
                        high_bound = self.max_batch
                    index = np.arange(low_bound, high_bound)
                data = np.ascontiguousarray(self.dataset[index])
                cudart.cudaMemcpy(self.buffer[name], data.ctypes.data, data.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
                output_list.append(self.buffer[name])
            self.count += 1
        return output_list

    def read_calibration_cache(self) -> bytes:  # necessary API
        if self.log:
            print("[CookbookCalibratorMNIST::read_calibration_cache]")
        if self.int8_cache_file.exists():
            if self.log:
                print(f"Succeed finding int8 cache file {self.int8_cache_file}")
            with open(self.int8_cache_file, "rb") as f:
                cache = f.read()
                return cache
        else:
            if self.log:
                print(f"Fail finding int8 cache file {self.int8_cache_file}")
            return

    def write_calibration_cache(self, cache) -> None:  # necessary API
        if self.log:
            print("[CookbookCalibratorMNIST::write_calibration_cache]")
        with open(self.int8_cache_file, "wb") as f:
            f.write(cache)
        if self.log:
            print(f"Succeed saving int8 cache file {self.int8_cache_file}")
        return

def unit_test_myCalibrator():
    m = CookbookCalibratorV1(5, (1, 1, 28, 28), "./test.int8cache")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")

class TRTWrapperV1:
    # Just a wrapper for the usage of TensorRT APIs, which can be unpacked back as process programming.
    # We use this for decreasing lines of code in most examples, though increasing complexity for reading.
    # I don't like this style of examples, but it might be huge workload to fix something in all examples.
    # So I'm sorry to present the examples like this :p

    def __init__(
        self,
        *,
        logger: Union[trt.Logger, trt.Logger.Severity, str] = None,  # Pass a `trt.Logger` from outside, or a logger level to create it inside
        trt_file: Path = None,  # If we already have a TensorRT engine file, just load it rather than build it from scratch.
        plugin_file_list: list = [],  # If we already have some plugins, just load them.
        callback_object_dict: dict = {},
    ) -> None:
        # Create a logger
        if isinstance(logger, trt.Logger):
            self.logger = logger
        elif isinstance(logger, trt.Logger.Severity):
            self.logger = trt.Logger(logger)
        elif isinstance(logger, str):
            self.logger = trt.Logger(text_to_logger_level(logger))
        else:
            self.logger = trt.Logger()

        # Register standard plugins, not required if we do not use plugin
        trt.init_libnvinfer_plugins(self.logger, namespace="")

        # Load custom plugins
        for plugin_file in plugin_file_list:
            if plugin_file.exists():
                ctypes.cdll.LoadLibrary(plugin_file)

        # Load engine bytes from file, or build it from scratch
        if trt_file is not None and trt_file.exists():
            with open(trt_file, "rb") as f:
                self.engine_bytes = f.read()
        else:
            self.builder = trt.Builder(self.logger)
            self.network = self.builder.create_network()
            self.profile = self.builder.create_optimization_profile()
            self.config = self.builder.create_builder_config()
            self.engine_bytes = None

        self.runtime = None
        self.engine = None
        self.context = None
        self.stream = 0

        self.callback_object_dict = callback_object_dict

        if "error_recorder" in self.callback_object_dict:
            self.builder.error_recorder = self.callback_object_dict["error_recorder"]
        if "algorithm_selector" in self.callback_object_dict:
            self.config.algorithm_selector = self.callback_object_dict["algorithm_selector"]
        if "int8_calibrator" in self.callback_object_dict:
            self.config.int8_calibrator = self.callback_object_dict["int8_calibrator"]
        if "progress_monitor" in self.callback_object_dict:
            self.config.progress_monitor = self.callback_object_dict["progress_monitor"]

        return

    # ================================ Buildtime actions
    def build(self, output_tensor_list: list = []) -> None:
        # Mark output tensors of the network and build engine bytes
        for tensor in output_tensor_list:
            self.network.mark_output(tensor)
        self.engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        return

    def serialize_engine(self, trt_file: Path):
        # Save engine bytes as TensorRT engine file
        if self.engine_bytes is None:
            print("Fail to serialize engine since engine_bytes is None.")
            return
        with open(trt_file, "wb") as f:
            f.write(self.engine_bytes)
        return

    # ================================ Runtime tool functions
    def _setup_utils(self):
        # Get input data and do preprocess before inference
        if self.runtime is None:  # Just in case we already have an runtime from outside
            self.runtime = trt.Runtime(self.logger)
        if self.engine is None:  # Just in case we already have an engine from outside
            self.engine = self.runtime.deserialize_cuda_engine(self.engine_bytes)
        if self.context is None:  # Just in case we already have an context from outside
            self.context = self.engine.create_execution_context()

        if "gpu_allocator" in self.callback_object_dict:
            self.runtime.gpu_allocator = self.callback_object_dict["gpu_allocator"]

        self.tensor_name_list = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.n_input = sum([self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT for name in self.tensor_name_list])
        self.n_output = self.engine.num_io_tensors - self.n_input

    def _setup_shape(self, input_data):
        for name, data in input_data.items():
            if name not in self.tensor_name_list[:self.n_input]:
                print(f"Skip `{name}` in data map")
                continue
            self.context.set_input_shape(name, data.shape)

        invalid_tensor_name_list = self.context.infer_shapes()
        if len(invalid_tensor_name_list) > 0:
            print(f"Invalid input tensor: {invalid_tensor_name_list}")

    def _setup_print_io_tensors(self):
        # Print information of input / output tensors
        for name in self.tensor_name_list:
            mode = self.engine.get_tensor_mode(name)
            data_type = self.engine.get_tensor_dtype(name)
            buildtime_shape = self.engine.get_tensor_shape(name)
            runtime_shape = self.context.get_tensor_shape(name)
            print(f"{'Input ' if mode == trt.TensorIOMode.INPUT else 'Output'}->{data_type}, {buildtime_shape}, {runtime_shape}, {name}")

    def _setup_buffer(self, input_data):
        # Prepare work before inference
        self.buffer = OrderedDict()
        for name in self.tensor_name_list:
            data_type = self.engine.get_tensor_dtype(name)
            runtime_shape = self.context.get_tensor_shape(name)
            n_byte = trt.volume(runtime_shape) * data_type.itemsize
            host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
            device_buffer = cudart.cudaMalloc(n_byte)[1]
            self.buffer[name] = [host_buffer, device_buffer, n_byte]

        for name, data in input_data.items():
            self.buffer[name][0] = np.ascontiguousarray(data)

        for name in self.tensor_name_list:
            self.context.set_tensor_address(name, self.buffer[name][1])

    # ================================ Runtime actions
    def setup(self, input_data: dict = {}, *, b_print_io: bool = True) -> None:
        # Get input data and do preprocess before inference
        self._setup_utils()

        self._setup_shape(input_data)

        if b_print_io:
            self._setup_print_io_tensors()

        self._setup_buffer(input_data)

        return

    def infer(self, *, b_print_io: bool = True, stream: int = 0, b_get_timeline: bool = False) -> None:
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
        return  # TODO: remove this since we need code below
        # Free device memory
        if hasattr(self, "buffer") and self.buffer != None and len(self.buffer) > 0:
            for _, device_buffer, _ in self.buffer.values():
                cudart.cudaFree(device_buffer)
        return

class TRTWrapperDDS(TRTWrapperV1):
    # Override for Data-dependent-Shape (DDS) mode
    # TRTWrapperDDS = TRTWrapperV1 + CookbookOutputAllocator

    def __init__(
        self,
        *,
        logger: Union[trt.Logger, trt.Logger.Severity, str] = None,
        trt_file: Path = None,
        plugin_file_list: list = [],
        callback_object_dict: dict = {},
    ) -> None:
        TRTWrapperV1.__init__(
            self,
            logger=logger,
            trt_file=trt_file,
            plugin_file_list=plugin_file_list,
            callback_object_dict=callback_object_dict,
        )

    # ================================ Runtime tool functions
    def _setup_buffer_dds(self, input_data):
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
                host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
                device_buffer = cudart.cudaMalloc(n_byte)[1]
            self.buffer[name] = [host_buffer, device_buffer, n_byte]

        for name, data in input_data.items():
            self.buffer[name][0] = np.ascontiguousarray(data)

        for name in self.tensor_name_list:
            self.context.set_tensor_address(name, self.buffer[name][1])

    # ================================ Runtime actions
    def setup(self, input_data: dict = {}, *, b_print_io: bool = True) -> None:
        # Get input data and do preprocess before inference
        self._setup_utils()

        self._setup_shape(input_data)

        if b_print_io:
            self._setup_print_io_tensors()

        self._setup_buffer_dds(input_data)

        return

    def infer(self, *, b_print_io: bool = True, stream: int = 0, b_get_timeline: bool = False) -> None:
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
                host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
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
    # Override for model with Shape-Input-Tensor
    # There 5 differences during `setup()` and `infer()`, see the code below

    def __init__(
        self,
        *,
        logger: Union[trt.Logger, trt.Logger.Severity, str] = None,
        trt_file: Path = None,
        plugin_file_list: list = [],
        callback_object_dict: dict = {},
    ) -> None:
        TRTWrapperV1.__init__(
            self,
            logger=logger,
            trt_file=trt_file,
            plugin_file_list=plugin_file_list,
            callback_object_dict=callback_object_dict,
        )

    # ================================ Runtime tool functions
    def _setup_shape_si(self, input_data):
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
        # Prepare work before inference
        self.buffer = OrderedDict()
        for name in self.tensor_name_list:
            data_type = self.engine.get_tensor_dtype(name)
            runtime_shape = self.context.get_tensor_shape(name)
            n_byte = trt.volume(runtime_shape) * data_type.itemsize
            host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
            # Key difference, no need to allocate device buffer for shape tensor
            if self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                device_buffer = cudart.cudaMalloc(n_byte)[1]
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
    def setup(self, input_data: dict = {}, *, b_print_io: bool = True) -> None:
        # Get input data and do preprocess before inference
        self._setup_utils()

        self._setup_shape_si(input_data)

        if b_print_io:
            self._setup_print_io_tensors()

        self._setup_buffer_si(input_data)

        return

    def infer(self, *, b_print_io: bool = True, stream: int = 0, b_get_timeline: bool = False) -> None:
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
    # TRTWrapperV2 = TRTWrapperV1 + TRTWrapperDDS + TRTWrapperShapeInput, pretty complex

    def __init__(
        self,
        *,
        logger: Union[trt.Logger, trt.Logger.Severity, str] = None,
        trt_file: Path = None,
        plugin_file_list: list = [],
        callback_object_dict: dict = {},
    ) -> None:
        TRTWrapperV1.__init__(
            self,
            logger=logger,
            trt_file=trt_file,
            plugin_file_list=plugin_file_list,
            callback_object_dict=callback_object_dict,
        )

    def setup(self, input_data: dict = {}, *, b_print_io: bool = True) -> None:
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
                host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
                if self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                    device_buffer = cudart.cudaMalloc(n_byte)[1]
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
                host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
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
    # TRTWrapperV2Torch = TRTWrapperV2 using pyTorch API

    def __init__(
        self,
        *,
        logger: Union[trt.Logger, trt.Logger.Severity, str] = None,
        trt_file: Path = None,
        plugin_file_list: list = [],
        callback_object_dict: dict = {},
    ) -> None:
        TRTWrapperV1.__init__(
            self,
            logger=logger,
            trt_file=trt_file,
            plugin_file_list=plugin_file_list,
            callback_object_dict=callback_object_dict,
        )

    def setup(self, input_data: dict = {}, *, b_print_io: bool = True) -> None:
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
                buffer = torch.empty(0, dtype=datatype_trt_to_torch(data_type)).cuda()
            else:
                buffer = torch.empty(tuple(runtime_shape), dtype=datatype_trt_to_torch(data_type)).cuda()
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
                tensor = torch.empty(tuple(runtime_shape), dtype=datatype_trt_to_torch(data_type), device='cuda')
                cudart.cudaMemcpyAsync(tensor.data_ptr(), device_buffer, n_bytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, self.stream)
                self.buffer[name] = tensor.cpu()

        cudart.cudaStreamSynchronize(self.stream)

        # Print output
        if b_print_io:
            for name in self.tensor_name_list:
                print(name)
                print(self.buffer[name])

    def __del__(self):
        pass  # cudaFree is not needed in pyTorch

        return
