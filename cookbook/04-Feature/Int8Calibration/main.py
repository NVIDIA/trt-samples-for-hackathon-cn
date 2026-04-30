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
