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

# Build a weight-stripped engine and refit it back to a full one.
#
# `BuilderFlag.STRIP_PLAN` builds an engine that keeps the optimized kernels and schedule but drops
# the weights, because the weights are already in the ONNX file being shipped alongside it. The
# engine is then refitted at load time from that same ONNX, with no accuracy loss.
#
# This example measures what the feature actually does rather than asserting it:
#   + how much smaller the stripped engine actually is;
#   + that a stripped engine really has no weights (it outputs zeros before refitting);
#   + that refitting from the ONNX restores the full engine's accuracy.
#
# Related examples: `04-Feature/Refit` (the general refit APIs, including refitting from raw weight
# arrays), `04-Feature/WeightStreaming` (a different problem - weights too large for device memory,
# streamed in at inference time rather than dropped from the plan).

from pathlib import Path

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, byte_to_string, case_mark, check_array, cookbook_path, parse_onnx

shape = [1, 1, 28, 28]
data_path = cookbook_path("00-Data", "data")
model_path = cookbook_path("00-Data", "model")
onnx_file_trained = model_path / "model-trained.onnx"

trt_file_full = Path("model-full.trt")
trt_file_stripped = Path("model-stripped.trt")

data = {"x": np.load(data_path / "InferenceData.npy")}

output_dict = {}  # Keep the outputs out of the case arguments, which `case_mark` echoes

def build_engine(trt_file, *, b_strip_plan, b_refit_identical=False):
    """Build one engine from the trained ONNX, optionally stripping the weights out of the plan."""
    tw = TRTWrapperV1()
    if b_strip_plan:
        # STRIP_PLAN drops the weights from the plan. It implies the engine is refittable, so
        # `BuilderFlag.REFIT` does not need to be set as well.
        tw.builder_config.set_flag(trt.BuilderFlag.STRIP_PLAN)
    if b_refit_identical:
        # REFIT_IDENTICAL promises the builder that the weights refitted later will be *the same*
        # values it saw at build time. That lets it keep weight-dependent optimizations (constant
        # folding, kernel selection based on the actual values), so the refitted engine performs
        # like a normal one. Refitting anything else is undefined behaviour.
        tw.builder_config.set_flag(trt.BuilderFlag.REFIT_IDENTICAL)

    parse_onnx(onnx_file_trained, tw.logger, tw.network, tw.builder_config)

    input_tensor = tw.network.get_input(0)
    tw.profile.set_shape(input_tensor.name, shape, [2] + shape[1:], [4] + shape[1:])

    tw.build()
    tw.serialize_engine(trt_file)
    return trt_file.stat().st_size

def refit_from_onnx(trt_file):
    """Load a weight-stripped engine and pour the weights back in from the original ONNX."""
    tw = TRTWrapperV1(trt_file=trt_file)
    tw.engine = trt.Runtime(tw.logger).deserialize_cuda_engine(tw.engine_bytes)

    refitter = trt.Refitter(tw.engine, tw.logger)
    # `OnnxParserRefitter` walks the ONNX and hands every matching initializer to the refitter, so
    # the weights never have to be extracted by hand.
    parser_refitter = trt.OnnxParserRefitter(refitter, tw.logger)
    assert parser_refitter.refit_from_file(str(onnx_file_trained)), "Fail refitting from ONNX file"
    assert refitter.refit_cuda_engine(), "Fail refitting engine"

    tw.setup(data)
    tw.infer(b_print_io=False)
    return tw.buffer["y"][0].copy()

def run_engine(trt_file):
    tw = TRTWrapperV1(trt_file=trt_file)
    tw.setup(data)
    tw.infer(b_print_io=False)
    return tw.buffer["y"][0].copy()

@case_mark
def case_full_engine():
    """Baseline: a normal engine, weights included."""
    n_byte = build_engine(trt_file_full, b_strip_plan=False)
    print(f"Full engine:     {byte_to_string(n_byte)}")
    output_dict["full"] = run_engine(trt_file_full)
    output_dict["n_byte_full"] = n_byte

@case_mark
def case_stripped_engine():
    """The same network with STRIP_PLAN, run before and after refitting."""
    n_byte = build_engine(trt_file_stripped, b_strip_plan=True)
    n_byte_full = output_dict["n_byte_full"]
    print(f"Stripped engine: {byte_to_string(n_byte)}")
    print(f"Saved {byte_to_string(n_byte_full - n_byte)}, i.e. {(1 - n_byte / n_byte_full) * 100:.1f}% of the full plan")

    # Run it *without* refitting first. The engine loads and executes happily, it just computes
    # with absent weights - the output is all zeros. This is what "stripped" actually means, and
    # why a stripped engine must never be shipped without its ONNX.
    before = run_engine(trt_file_stripped)
    print(f"Output before refitting: {before[0][:5]}")
    print("The next check is EXPECTED to report False - the weights are not in the plan yet:")
    check_array(before, output_dict["full"], True, des=" before refit", error_epsilon=1e-3)

    # Now pour the weights back in from the ONNX the engine was built from.
    output_dict["stripped"] = refit_from_onnx(trt_file_stripped)
    print(f"Output after refitting:  {output_dict['stripped'][0][:5]}")

    # Note the tolerance: refitting restores the *accuracy*, not necessarily bit-identical results.
    # STRIP_PLAN keeps the builder from specializing on weight values, so the stripped plan can end
    # up choosing different kernels than the full one; the two then differ by ordinary
    # floating-point noise (a few 1e-3 on logits of magnitude ~10 here).
    check_array(output_dict["stripped"], output_dict["full"], True, des=" after refit", error_epsilon=1e-2)

@case_mark
def case_strip_and_refit_identical():
    """STRIP_PLAN combined with REFIT_IDENTICAL.

    Plain STRIP_PLAN makes the builder conservative: it cannot fold or specialize on weight values
    it is going to throw away. REFIT_IDENTICAL tells it the same values will come back, so it may
    keep those optimizations. Use it when one set of weights is shipped to several backends or GPU
    architectures; do not use it if the engine will be refitted with *different* weights.
    """
    trt_file = Path("model-stripped-identical.trt")
    n_byte = build_engine(trt_file, b_strip_plan=True, b_refit_identical=True)
    print(f"Stripped + REFIT_IDENTICAL: {byte_to_string(n_byte)}")

    check_array(refit_from_onnx(trt_file), output_dict["full"], True, error_epsilon=1e-2)

if __name__ == "__main__":
    for trt_path in Path(".").glob("*.trt"):
        trt_path.unlink(missing_ok=True)

    case_full_engine()
    case_stripped_engine()
    case_strip_and_refit_identical()

    print("Finish")
