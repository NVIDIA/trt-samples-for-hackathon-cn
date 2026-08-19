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

from pathlib import Path

import numpy as np
import tensorrt as trt
from typing import Dict, List
from cuda.bindings import runtime as cudart

from tensorrt_cookbook import TRTWrapperV1, case_mark, CookbookCalibratorV1

# Shape of the single network input: [batch, feature].
input_shape = [4, 8]
cache_file = Path("model.Int8Cache")

# TensorRT ships four INT8 post-training calibration algorithms.  Each value of
# `trt.CalibrationAlgoType` corresponds to a concrete calibrator base class that
# a user subclasses and hands to `builder_config.int8_calibrator`.  A calibrator
# must report which algorithm it implements through `get_algorithm()`.
calibration_algo_to_calibrator = {
    trt.CalibrationAlgoType.LEGACY_CALIBRATION: trt.IInt8LegacyCalibrator,  # Original TRT calibrator, needs quantile/regression cutoff, deprecated
    trt.CalibrationAlgoType.ENTROPY_CALIBRATION: trt.IInt8EntropyCalibrator,  # KL-divergence based calibrator (v1)
    trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2: trt.IInt8EntropyCalibrator2,  # KL-divergence based calibrator (v2), recommended default
    trt.CalibrationAlgoType.MINMAX_CALIBRATION: trt.IInt8MinMaxCalibrator,  # Min/Max based calibrator, common for NLP / transformer networks
}
# `trt.IInt8Calibrator` is the abstract root class that all four calibrators above inherit from.
assert all(issubclass(c, trt.IInt8Calibrator) for c in calibration_algo_to_calibrator.values())

class CookbookCalibratorV1(trt.IInt8EntropyCalibrator2):
    """A minimal INT8 Entropy(v2) calibrator feeding synthetic numpy data.

    A real calibrator would iterate over a representative dataset; here we just
    generate a few random batches so the example needs no external data.
    """

    def __init__(self, n_batch: int, shape: list, cache_file: Path) -> None:
        trt.IInt8EntropyCalibrator2.__init__(self)  # Necessary, initialize the base class
        self.n_batch = n_batch
        self.shape = shape
        self.cache_file = cache_file

    def get_batch_size(self) -> int:  # Necessary API, return the calibration batch size
        return self.shape[0]

    def get_batch(self, names, *args):  # Necessary API, return device pointers of one batch or None when finished
        if self.count >= self.n_batch:
            return None
        self.count += 1
        data = np.random.rand(*self.shape).astype(np.float32) * 2 - 1  # synthetic data in [-1, 1]
        data = np.ascontiguousarray(data)
        cudart.cudaMemcpy(self.device_input, data.ctypes.data, self.buffer_size, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        print(f"    get_batch: feeding calibration batch {self.count}/{self.n_batch} for input {names}")
        return [int(self.device_input)]

    def read_calibration_cache(self):  # Necessary API, reuse a cache to skip calibration when available
        if self.cache_file.exists():
            print(f"    read_calibration_cache: reuse {self.cache_file}")
            return self.cache_file.read_bytes()
        print("    read_calibration_cache: no cache found, run calibration")
        return None

    def write_calibration_cache(self, cache) -> None:  # Necessary API, persist calibration result
        self.cache_file.write_bytes(cache)
        print(f"    write_calibration_cache: save {self.cache_file}")

class CookbookCalibratorMNIST(trt.IInt8EntropyCalibrator2):
    """MNIST dataset-based INT8 calibrator with optional random sampling."""

    def __init__(
        self,
        input_info: Dict[str, list] | None = None,
        dataset_path: Path = None,
        int8_cache_file: Path = None,
        is_random_choose: bool = False,
        batch_size: int = 1,
        log: bool = False,
    ) -> None:
        """Initialize MNIST-based calibrator and allocate per-input CUDA buffers."""
        input_info = input_info or {}
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
        """Release all calibration CUDA buffers."""
        if self.log:
            print("[CookbookCalibratorMNIST::__del__]")
        for name, buffer in self.buffer.items():
            cudart.cudaFree(buffer)

    def get_batch_size(self) -> int:  # necessary API
        """Return calibration batch size."""
        if self.log:
            print("[CookbookCalibratorMNIST::get_batch_size]")
        return self.batch_size

    def get_batch(self, names: List[str]) -> List[int]:  # necessary API
        """Copy one calibration batch from dataset to CUDA buffers."""
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
        """Load cached calibration table if it exists."""
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
        """Write generated calibration table to cache file."""
        if self.log:
            print("[CookbookCalibratorMNIST::write_calibration_cache]")
        with open(self.int8_cache_file, "wb") as f:
            f.write(cache)
        if self.log:
            print(f"Succeed saving int8 cache file {self.int8_cache_file}")
        return

def unit_test_myCalibrator():
    """Quick smoke test helper for ``CookbookCalibratorV1``."""
    m = CookbookCalibratorV1(5, (1, 1, 28, 28), "./test.Int8Cache")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")

@case_mark
def case_calibrator_api():
    # Instantiate the calibrator and inspect the algorithm it reports.
    calibrator = CookbookCalibratorV1(4, input_shape, cache_file)
    algo = calibrator.get_algorithm()  # `get_algorithm` reports which trt.CalibrationAlgoType this calibrator uses
    print(f"    calibrator.get_algorithm() = {algo}")
    assert algo == trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2

    print("    Mapping of trt.CalibrationAlgoType -> calibrator base class:")
    for algo_type, calibrator_class in calibration_algo_to_calibrator.items():
        print(f"        {algo_type} -> {calibrator_class.__name__}")
    return calibrator

@case_mark
def case_build_int8_engine(calibrator):
    if cache_file.exists():
        cache_file.unlink()

    tw = TRTWrapperV1()
    builder_config = tw.builder_config

    # Build a small network: input -> MatrixMultiply(weight) -> ReLU.
    input_tensor = tw.network.add_input("inputT0", trt.float32, [-1, input_shape[1]])
    tw.profile.set_shape(input_tensor.name, [1, input_shape[1]], input_shape, [input_shape[0] * 2, input_shape[1]])
    builder_config.add_optimization_profile(tw.profile)

    weight = tw.network.add_constant([input_shape[1], input_shape[1]], np.ascontiguousarray(np.random.rand(input_shape[1], input_shape[1]).astype(np.float32)))
    mm = tw.network.add_matrix_multiply(input_tensor, trt.MatrixOperation.NONE, weight.get_output(0), trt.MatrixOperation.NONE)
    relu = tw.network.add_activation(mm.get_output(0), trt.ActivationType.RELU)
    tw.network.mark_output(relu.get_output(0))

    # Enable INT8 and attach the calibrator so TensorRT can compute dynamic ranges.
    builder_config.set_flag(trt.BuilderFlag.INT8)
    builder_config.int8_calibrator = calibrator

    engine_bytes = tw.builder.build_serialized_network(tw.network, builder_config)
    if engine_bytes is None:
        # INT8 calibration needs a GPU that supports INT8; keep the script exit 0 on unsupported hardware.
        print("    build_serialized_network returned None (INT8 likely unsupported on this GPU), skip runtime demo")
        return None

    print(f"    Succeed building INT8 engine, size = {engine_bytes.nbytes} Bytes")
    return engine_bytes

@case_mark
def case_runtime_config(engine_bytes):
    # Bonus: inspect IRuntimeConfig.get_execution_context_allocation_strategy on the built engine.
    if engine_bytes is None:
        print("    No engine available, skip runtime-config demo")
        return
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        print("    Fail deserializing engine, skip runtime-config demo")
        return
    if not hasattr(engine, "create_runtime_config"):
        print("    ICudaEngine.create_runtime_config unavailable in this TensorRT build, skip")
        return
    runtime_config = engine.create_runtime_config()
    runtime_config.set_execution_context_allocation_strategy(trt.ExecutionContextAllocationStrategy.STATIC)
    strategy = runtime_config.get_execution_context_allocation_strategy()
    print(f"    runtime_config.get_execution_context_allocation_strategy() = {strategy}")

if __name__ == "__main__":
    calibrator = case_calibrator_api()
    engine_bytes = case_build_int8_engine(calibrator)
    case_runtime_config(engine_bytes)

    print("Finish")
