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
import hashlib

import numpy as np
import tensorrt as trt
from typing import List
from tensorrt_cookbook import (TRTWrapperV1, case_mark, load_mnist_network_trt, datatype_cast, byte_to_string)

trt_file = Path("deterministic.engine")

class CookbookAlgorithmSelector(trt.IAlgorithmSelector):
    """Algorithm selector example with several strategy modes for tactic selection."""

    def __init__(self, i_strategy=0, log=False) -> None:  # Pass a number on behalf of our customerized strategy to select algorithm
        """Initialize selector with a strategy index and logging option."""
        if log:
            print("[CookbookAlgorithmSelector::__init__]")
        super().__init__()
        self.i_strategy = i_strategy
        self.log = log

    def select_algorithms(self, layerAlgorithmContext: trt.IAlgorithmContext, layerAlgorithmList) -> List[int]:
        """Choose candidate algorithm indices for one layer according to strategy."""
        # `layerAlgorithmContext` is a `trt.IAlgorithmContext` describing the layer being tuned
        # (its name, number of inputs/outputs and their shapes).
        # Each element of `layerAlgorithmList` is a `trt.IAlgorithm`, from which we can query:
        #   - `algorithm.algorithm_variant`      -> `trt.IAlgorithmVariant` (implementation + tactic)
        #   - `algorithm.get_algorithm_io_info(i)` -> `trt.IAlgorithmIOInfo` (dtype / stride of I/O tensor i)
        if self.log:
            print("[CookbookAlgorithmSelector::select_algorithms]")
        # we print the alternative algorithms of each layer here
        nInput = layerAlgorithmContext.num_inputs
        nOutput = layerAlgorithmContext.num_outputs
        if self.log:
            print(f"Layer {layerAlgorithmContext.name}, {nInput=}, {nOutput=}")
            for i in range(nInput + nOutput):
                info = f"    {'Input ' if i < nInput else 'Output'}     {i if i < nInput else i - nInput: 2d}:"
                info += f"shape={layerAlgorithmContext.get_shape(i)}"
                print(info)

            for i, algorithm in enumerate(layerAlgorithmList):
                variant: trt.IAlgorithmVariant = algorithm.algorithm_variant
                info = f"    algorithm{i:4d}:"
                info += f"implementation[{variant.implementation: 10d}],"
                info += f"tactic[{variant.tactic: 20d}],"
                info += f"timing[{algorithm.timing_msec * 1000: 7.3f}us],"
                info += f"workspace[{byte_to_string(algorithm.workspace_size)}]"
                for j in range(nInput + nOutput):
                    io_info: trt.IAlgorithmIOInfo = algorithm.get_algorithm_io_info(j)
                    info += f"\n                  {'Input ' if j < nInput else 'Output'}{j if j < nInput else j - nInput: 2d}:"
                    info += f"datatype={datatype_cast(io_info.dtype, 'str')},"
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

        else:  # Default behavior: keep all algorithms
            result = list(range(len(layerAlgorithmList)))

        return result

    def report_algorithms(self, modelAlgorithmContext, modelAlgorithmList) -> None:  # report the tactic of the whole network
        """Report selected tactics for all layers in the model."""
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
                info += f"datatype={datatype_cast(io_info.dtype, 'str')},"
                info += f"stride={io_info.strides},"
                info += f"vectorized_dim={io_info.vectorized_dim},"
                info += f"components_per_element={io_info.components_per_element}"
            print(info)
        return

def _build_with_strategy(i_strategy: int = 0) -> bytes | None:
    data = {"x": np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)}
    callback_object_dict = {"algorithm_selector": CookbookAlgorithmSelector(i_strategy=i_strategy)}
    tw = TRTWrapperV1(callback_object_dict=callback_object_dict)

    load_mnist_network_trt(tw)

    tw.build()
    return tw.engine_bytes

def _hash_engine(engine_bytes: bytes) -> str:
    return hashlib.sha256(engine_bytes).hexdigest()

@case_mark
def case_compare():
    engine_bytes_0 = _build_with_strategy(0)
    engine_bytes_1 = _build_with_strategy(2)
    engine_bytes_2 = _build_with_strategy(2)
    print(f"Hash of engine with strategy=0        : {_hash_engine(engine_bytes_0)}")
    print(f"Hash of engine with strategy=2 (run 1): {_hash_engine(engine_bytes_1)}")
    print(f"Hash of engine with strategy=2 (run 2): {_hash_engine(engine_bytes_2)}")

if __name__ == "__main__":
    trt_file.unlink(missing_ok=True)

    case_compare()

    print("Finish")
