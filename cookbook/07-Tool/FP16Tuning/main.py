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
"""Find the layers an FP16 engine should keep in FP32, one layer at a time.

`FP16Tuning` lives in `tensorrt_cookbook/utils_class.py`; what is here is the two ways of feeding it
input data - recorded, and random - and the parameters each one passes.

Tuning happens on the **ONNX graph**, not on the TensorRT network:
`BuilderFlag.PREFER_PRECISION_CONSTRAINTS` was removed, so "run in FP16 but keep these layers in
FP32" is expressed by converting the ONNX model to FP16 with those nodes excluded (ModelOpt
autocast) and building a strongly-typed engine from the result. That is why `tune_type_list` holds
ONNX op types such as `Conv` and `Gemm` rather than TensorRT layer types.

Two logs come out of a run, and they are not the same thing:

+ `log-main.py.log` is stdout and stderr - the case markers, the progress bar and TensorRT's own
  messages. This is what a sweep captures.
+ `FP16Tuning.log` is the tool's own narration, one line per step, named by `log_file` below. It is
  deliberately a separate file: tuning every layer of a real model produces far more text than
  anyone wants interleaved with the rest. `FP16Tuning` builds the logger itself from the name.
"""

from pathlib import Path

from tensorrt_cookbook import FP16Tuning, case_mark, cookbook_path

LOG_FILE = Path("FP16Tuning.log")

@case_mark
def case_data():
    """Recorded inputs: `infer_shape` is taken from the data file, so the shapes match what was dumped."""
    FP16Tuning(
        onnx_file=cookbook_path("00-Data", "model", "model-trained.onnx"),
        output_file=Path("FP16Tuning-data-report.md"),
        log_file=LOG_FILE,
        # Shape in trtexec format, for example: x:4x64x64,y:4,z:"
        min_shape="x:1x1x28x28",
        opt_shape="x:2x1x28x28",
        max_shape="x:4x1x28x28",
        data_file=cookbook_path("00-Data", "data", "InferenceData.npz"),
        tune_type_list=["Conv", "Gemm", "MatMul"],
        test_performance=True,
        max_tune_layers=1000,
    ).tune()

@case_mark
def case_random():
    """Random inputs of `infer_shape`, which falls back to `opt_shape` when it is not given."""
    FP16Tuning(
        onnx_file=cookbook_path("00-Data", "model", "model-trained.onnx"),
        output_file=Path("FP16Tuning-random-report.md"),
        log_file=LOG_FILE,
        min_shape="x:1x1x28x28",
        opt_shape="x:2x1x28x28",
        max_shape="x:4x1x28x28",
        # No `data_file`, so random data of `infer_shape` is used
        infer_shape=None,
        tune_type_list=["Conv", "Gemm", "MatMul"],
        test_performance=True,
        max_tune_layers=1000,
        # The layers we want to tune. No other layers will be tuned if this is set
        specify_layer_name_list=[],
        # The layers we never try to set back to FP32 (maybe due to large loss of performance)
        skip_layer_name_list=[],
        # The layers must stay in FP32
        force_fp32_layer_name_list=[],
        # The output tensor used for BestAcc ranking, using model first output when set to None
        focus_tensor=None,
    ).tune()

if __name__ == "__main__":
    LOG_FILE.unlink(missing_ok=True)  # One run, one log

    case_data()
    case_random()

    print("Finish")
