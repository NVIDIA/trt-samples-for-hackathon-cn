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

import ctypes
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import nvtx
import tensorrt as trt
import torch
from cuda import cudart

np.random.seed(31193)
np.set_printoptions(precision=3, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()

#=======================================================================================================================
def ceil_divide(a, b):
    return (a + b - 1) // b

def round_up(a, b):
    return ceil_divide(a, b) * b

def byte_to_string(xByte):
    if xByte < (1 << 10):
        return f"{xByte: 5.1f}  B"
    if xByte < (1 << 20):
        return f"{xByte / (1 << 10): 5.1f}KiB"
    if xByte < (1 << 30):
        return f"{xByte / (1 << 20): 5.1f}MiB"
    return f"{xByte / (1 << 30): 5.1f}GiB"

def datatype_to_string(datatype_trt):
    # Cast TensorRT data type into string
    return datatype_trt.__str__()[9:]

def datatype_trt_to_torch(datatype_trt):
    # Cast TensorRT data type into Torch
    if datatype_trt == trt.float32:
        return torch.float32
    if datatype_trt == trt.float16:
        return torch.float16
    if datatype_trt == trt.int8:
        return torch.int8
    if datatype_trt == trt.int32:
        return torch.int32
    if datatype_trt == trt.bool:
        return torch.bool
    if datatype_trt == trt.uint8:
        return torch.uint8
    if datatype_trt == trt.DataType.FP8:
        return torch.float8_e4m3fn
    if datatype_trt == trt.bf16:
        return torch.bfloat16
    if datatype_trt == trt.int64:
        return torch.int64
    if datatype_trt == trt.int4:
        return None  # only torch.uint4 is supported
    return None

def datatype_np_to_trt(datatype_np):
    # Cast TensorRT data type into Torch
    if datatype_np == np.float32:
        return trt.float32
    if datatype_np == np.float16:
        return trt.float16
    if datatype_np == np.int8:
        return trt.int8
    if datatype_np == np.int32:
        return trt.int32
    if datatype_np == bool:
        return trt.bool
    if datatype_np == np.uint8:
        return trt.uint8
    if datatype_np == np.int64:
        return trt.int64
    return None

def printArrayInformation(x, info="", n=5):
    # Print statistic information of the tensor `x`
    if 0 in x.shape:
        print('%s:%s' % (info, str(x.shape)))
        return
    x = x.astype(np.float32)
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t', x.reshape(-1)[:n], x.reshape(-1)[-n:])
    return

def check_array(a, b, weak=False, info="", error_epsilon=1e-5):
    # Compare tensor `a` and `b`
    if a.shape != b.shape:
        print(f"[check]Shape different: A{a.shape} : B{b.shape}")
        return
    if weak:
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        res = np.all(np.abs(a - b) < error_epsilon)
    else:
        res = np.all(a == b)
    maxAbsDiff = np.max(np.abs(a - b))
    meanAbsDiff = np.mean(np.abs(a - b))
    maxRelDiff = np.max(np.abs(a - b) / (np.abs(b) + error_epsilon))
    meanRelDiff = np.mean(np.abs(a - b) / (np.abs(b) + error_epsilon))
    result = f"[check]{info}:{res},{maxAbsDiff=:.2e},{meanAbsDiff=:.2e},{maxRelDiff=:.2e},{meanRelDiff=:.2e}"
    if maxAbsDiff > error_epsilon:
        index = np.argmax(np.abs(a - b))
        valueA, valueB = a.flatten()[index], b.flatten()[index]
        shape = a.shape
        indexD = []
        for i in range(len(shape) - 1, -1, -1):
            x = index % shape[i]
            indexD = [x] + indexD
            index = index // shape[i]
        result += f"\n    worstPair=({valueA}:{valueB})@{indexD}"
    print(result)
    return res

def case_mark(f):

    def f_with_mark(*args, **kargs):
        print("=" * 64 + f" Start [{f.__name__}]")
        f(*args, **kargs)
        print("=" * 64 + f" End   [{f.__name__}]")

    return f_with_mark

def print_network(network):
    # print the network for debug
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        print(i, "%s,in=%d,out=%d,%s" % (str(layer.type)[10:], layer.num_inputs, layer.num_outputs, layer.name))
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

#=======================================================================================================================
class MyOutputAllocator(trt.IOutputAllocator):

    def __init__(self, log=False) -> None:
        if log:
            print("[MyOutputAllocator::__init__]")
        super(MyOutputAllocator, self).__init__()
        # members for outside use
        self.shape = None
        self.n_bytes = 0
        self.address = 0
        self.log = log

    def reallocate_output(self, tensor_name, memory, size, alignment) -> int:
        if self.log:
            print(f"[MyOutputAllocator::reallocate_output] {tensor_name=}, old_address={memory}, {size=}, {alignment=}")
        if size <= self.n_bytes:
            return memory
        if memory != 0:
            status = cudart.cudaFree(memory)
            if status != cudart.cudaError_t.cudaSuccess:
                if self.log:
                    print("Fail freeing old memory")
                return 0
        status, address = cudart.cudaMalloc(size)
        if status != cudart.cudaError_t.cudaSuccess:
            if self.log:
                print("Fail allocating new memory")
            return 0
        self.n_bytes = size
        self.address = address
        return address

    def reallocate_output_async(self, tensor_name, memory, size, alignment, stream) -> int:
        if self.log:
            print(f"[MyOutputAllocator::reallocate_output_async] {tensor_name=}, old_address={memory}, {size=}, {alignment=}, {stream=}")
        if size <= self.n_bytes:
            return memory
        if memory != 0:
            status = cudart.cudaFreeAsync(memory, stream)
            if status != cudart.cudaError_t.cudaSuccess:
                if self.log:
                    print("Fail freeing old memory")
                return 0
        status, address = cudart.cudaMallocAsync(size, stream)
        if status != cudart.cudaError_t.cudaSuccess:
            if self.log:
                print("Fail allocating new memory")
            return 0
        self.n_bytes = size
        self.address = address
        return address

    def notify_shape(self, tensor_name, shape):
        if self.log:
            print(f"[MyOutputAllocator::notify_shape] {tensor_name=}, {shape=}")
        self.shape = shape
        return

class MyAlgorithmSelector(trt.IAlgorithmSelector):

    def __init__(self, iStrategy=0, log=False) -> None:  # initialize with a number of our customerized strategies to select algorithm
        if log:
            print("[MyAlgorithmSelector::__init__]")
        super(MyAlgorithmSelector, self).__init__()
        self.iStrategy = iStrategy
        self.log = log

    def select_algorithms(self, layerAlgorithmContext, layerAlgorithmList) -> List[int]:
        if self.log:
            print("[MyAlgorithmSelector::select_algorithms]")
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
                info += f"datatype={datatype_to_string(io_info.dtype)},"
                info += f"stride={io_info.strides},"
                info += f"vectorized_dim={io_info.vectorized_dim},"
                info += f"components_per_element={io_info.components_per_element}"

            print(info)

        if self.iStrategy == 0:  # choose the algorithm with shortest time, TensorRT default strategy
            timeList = [algorithm.timing_msec for algorithm in layerAlgorithmList]
            result = [np.argmin(timeList)]

        elif self.iStrategy == 1:  # choose the algorithm with longest time, to get a TensorRT engine with worst performance, just for fun :)
            timeList = [algorithm.timing_msec for algorithm in layerAlgorithmList]
            result = [np.argmax(timeList)]

        elif self.iStrategy == 2:  # choose the algorithm using smallest workspace
            workspaceSizeList = [algorithm.workspace_size for algorithm in layerAlgorithmList]
            result = [np.argmin(workspaceSizeList)]

        elif self.iStrategy == 3:  # choose one certain algorithm we have known
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
            print("[MyAlgorithmSelector::report_algorithms]")
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
                info += f"datatype={datatype_to_string(io_info.dtype)},"
                info += f"stride={io_info.strides},"
                info += f"vectorized_dim={io_info.vectorized_dim},"
                info += f"components_per_element={io_info.components_per_element}"
            print(info)
        return

class MyProgressMonitor(trt.IProgressMonitor):

    def __init__(self, log=False) -> None:
        if log:
            print("[MyProgressMonitor::__init__]")
        trt.IProgressMonitor.__init__(self)
        self.level = 0
        self.log = log

    def phase_start(self, phase_name, parent_phase, num_steps) -> None:
        if self.log:
            print(f"[MyProgressMonitor::phase_start]{phase_name=},{parent_phase=},{num_steps=}")
        print("|   " * self.level + f"Start[{phase_name}]:{parent_phase=},{num_steps=}")
        self.level += 1
        return

    def phase_finish(self, phase_name) -> None:
        if self.log:
            print(f"[MyProgressMonitor::phase_finish]{phase_name=}")
        self.level -= 1
        print("|   " * self.level + f"End  [{phase_name}]")

        return

    def step_complete(self, phase_name, step) -> bool:
        if self.log:
            print(f"[MyProgressMonitor::step_complete]{phase_name=},{step=}")
        print("|   " * self.level + f"Step [{phase_name}]:{step=}")
        return True

class MyCalibratorV1(trt.IInt8EntropyCalibrator2):  # only for one-input-network

    def __init__(self, n_epoch: int = 1, input_shape: list = [], cache_file: Path = None):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.n_epoch = n_epoch
        self.shape = input_shape
        self.cache_file = cache_file
        self.buffer_size = trt.volume(input_shape) * trt.float32.itemsize
        _, self.dIn = cudart.cudaMalloc(self.buffer_size)
        self.count = 0

    def __del__(self):
        cudart.cudaFree(self.dIn)

    def get_batch_size(self):  # necessary API
        return self.shape[0]

    def get_batch(self, nameList=None, inputNodeName=None):  # necessary API
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

    def read_calibration_cache(self):  # necessary API
        if self.cache_file.exists():
            print(f"Succeed finding int8 cache file {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                cache = f.read()
                return cache
        else:
            print(f"Fail finding int8 cache file {self.cache_file}")
            return

    def write_calibration_cache(self, cache):  # necessary API
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"Succeed saving int8 cache file {self.cache_file}")
        return

def unit_test_myCalibrator():
    m = MyCalibratorV1(5, (1, 1, 28, 28), "./test.int8cache")
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
        logger: trt.Logger = None,  # Pass a `trt.Logger` from outside, or we will create one here.
        trt_file: Path = None,  # If we already have a TensorRT engine file, just load it rather than build it from scratch.
        plugin_file_list: list = [],  # If we already have some plugins, just load them.
    ):
        # Create a logger
        if logger is None:
            self.logger = trt.Logger(trt.Logger.Severity.ERROR)
        else:
            self.logger = logger

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
            self.engine = None

        return

    # ================================ Buildtime actions
    def build(self, output_tensor_list: list = []):
        # Mark output tensors of the network and build engine bytes
        for tensor in output_tensor_list:
            self.network.mark_output(tensor)
        self.engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        return

    def serialize_engine(self, trt_file: Path):
        # Save engine bytes as TensorRT engine file
        with open(trt_file, "wb") as f:
            f.write(self.engine_bytes)
        return

    # ================================ Runtime actions
    def setup(self, input_data: dict = {}):
        # Get input data and do preprocess before inference
        if self.engine is None:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(self.engine_bytes)

        self.tensor_name_list = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.n_input = sum([self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT for name in self.tensor_name_list])
        self.n_output = self.engine.num_io_tensors - self.n_input

        self.context = self.engine.create_execution_context()
        for name, data in input_data.items():
            self.context.set_input_shape(name, data.shape)

        # Print information of input / output tensors
        for name in self.tensor_name_list:
            mode = self.engine.get_tensor_mode(name)
            data_type = self.engine.get_tensor_dtype(name)
            buildtime_shape = self.engine.get_tensor_shape(name)
            runtime_shape = self.context.get_tensor_shape(name)
            print(f"{'Input ' if mode == trt.TensorIOMode.INPUT else 'Output'}->{data_type}, {buildtime_shape}, {runtime_shape}, {name}")

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

        return

    def infer(self, print_io: bool = True, is_get_timeline=False):
        # Do inference and print output
        for name in self.tensor_name_list:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                cudart.cudaMemcpy(self.buffer[name][1], self.buffer[name][0].ctypes.data, self.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        self.context.execute_async_v3(0)

        if is_get_timeline:  # do more inference if we want to get a timeline
            for _ in range(10):  # warm up
                self.context.execute_async_v3(0)
            with nvtx.annotate("Inference", color="green"):
                self.context.execute_async_v3(0)

        for name in self.tensor_name_list:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                cudart.cudaMemcpy(self.buffer[name][0].ctypes.data, self.buffer[name][1], self.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        if print_io:
            for name in self.tensor_name_list:
                print(name)
                print(self.buffer[name][0])

        for _, device_buffer, _ in self.buffer.values():
            cudart.cudaFree(device_buffer)

        return

class TRTWrapperDDS(TRTWrapperV1):
    # Override for Data-dependent-Shape (DDS) mode
    # TRTWrapperDDS = TRTWrapperV1 + MyOutputAllocator

    def __init__(self, logger: trt.Logger = None, trt_file: Path = None, plugin_file_list: list = []):
        TRTWrapperV1.__init__(self, logger, trt_file, plugin_file_list)

    def setup(self, input_data: dict = {}):
        # Get input data and do preprocess before inference
        if self.engine is None:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(self.engine_bytes)

        self.tensor_name_list = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.n_input = sum([self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT for name in self.tensor_name_list])
        self.n_output = self.engine.num_io_tensors - self.n_input

        self.context = self.engine.create_execution_context()
        for name, data in input_data.items():
            self.context.set_input_shape(name, data.shape)

        # Print information of input / output tensors
        for name in self.tensor_name_list:
            mode = self.engine.get_tensor_mode(name)
            data_type = self.engine.get_tensor_dtype(name)
            buildtime_shape = self.engine.get_tensor_shape(name)
            runtime_shape = self.context.get_tensor_shape(name)
            print(f"{'Input ' if mode == trt.TensorIOMode.INPUT else 'Output'}->{data_type}, {buildtime_shape}, {runtime_shape}, {name}")

        # Prepare work before inference
        self.buffer = OrderedDict()
        self.output_allocator_map = OrderedDict()
        for name in self.tensor_name_list:
            data_type = self.engine.get_tensor_dtype(name)
            runtime_shape = self.context.get_tensor_shape(name)
            if -1 in runtime_shape:  # for Data-Dependent-Shape (DDS) output, "else" branch for normal output
                n_byte = 0  # self.context.get_max_output_size(name)
                self.output_allocator_map[name] = MyOutputAllocator()
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

        return

    def infer(self, print_io: bool = True):
        # Do inference and print output
        for name in self.tensor_name_list:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                cudart.cudaMemcpy(self.buffer[name][1], self.buffer[name][0].ctypes.data, self.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        self.context.execute_async_v3(0)

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
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                cudart.cudaMemcpy(self.buffer[name][0].ctypes.data, self.buffer[name][1], self.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        if print_io:
            for name in self.tensor_name_list:
                print(name)
                print(self.buffer[name][0])

        for _, device_buffer, _ in self.buffer.values():
            cudart.cudaFree(device_buffer)

        return

class TRTWrapperShapeInput(TRTWrapperV1):
    # Override for model with Shape-Input-Tensor
    # There 5 differences during `setup()` and `infer()`, see the code below

    def __init__(self, logger: trt.Logger = None, trt_file: Path = None, plugin_file_list: list = []):
        TRTWrapperV1.__init__(self, logger, trt_file, plugin_file_list)

    def setup(self, input_data: dict = {}):
        # Get input data and do preprocess before inference
        if self.engine is None:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(self.engine_bytes)

        self.tensor_name_list = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.n_input = sum([self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT for name in self.tensor_name_list])
        self.n_output = self.engine.num_io_tensors - self.n_input

        self.context = self.engine.create_execution_context()
        for name, data in input_data.items():
            # Key difference, use `set_tensor_address()` instead of `set_input_shape()` for shape input tensor
            if self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                self.context.set_input_shape(name, data.shape)
            else:
                self.context.set_tensor_address(name, data)

        # Print information of input / output tensors
        for name in self.tensor_name_list:
            mode = self.engine.get_tensor_mode(name)
            data_type = self.engine.get_tensor_dtype(name)
            buildtime_shape = self.engine.get_tensor_shape(name)
            runtime_shape = self.context.get_tensor_shape(name)
            print(f"{'Input ' if mode == trt.TensorIOMode.INPUT else 'Output'}->{data_type}, {buildtime_shape}, {runtime_shape}, {name}")

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
            # Key difference, we have call `set_tensor_address()` for shape input tensors before
            if self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                self.context.set_tensor_address(name, self.buffer[name][1])
            elif self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.context.set_tensor_address(name, self.buffer[name][0].ctypes.data)

        return

    def infer(self, print_io: bool = True, is_get_timeline=False):
        # Do inference and print output
        for name in self.tensor_name_list:
            # Key difference, need not to copy shape tensor buffer between CPU and GPU
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT and self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                cudart.cudaMemcpy(self.buffer[name][1], self.buffer[name][0].ctypes.data, self.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        self.context.execute_async_v3(0)

        if is_get_timeline:  # do more inference if we want to get a timeline
            for _ in range(10):  # warm up
                self.context.execute_async_v3(0)
            with nvtx.annotate("Inference", color="green"):
                self.context.execute_async_v3(0)

        for name in self.tensor_name_list:
            # Key difference, need not to copy shape tensor buffer between CPU and GPU
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT and self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE:
                cudart.cudaMemcpy(self.buffer[name][0].ctypes.data, self.buffer[name][1], self.buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        if print_io:
            for name in self.tensor_name_list:
                print(name)
                print(self.buffer[name][0])

        for _, device_buffer, _ in self.buffer.values():
            cudart.cudaFree(device_buffer)

        return
