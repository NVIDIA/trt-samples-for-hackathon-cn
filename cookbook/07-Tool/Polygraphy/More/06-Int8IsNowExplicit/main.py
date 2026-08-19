# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
"""INT8 without a calibrator: what replaced `polygraphy.backend.trt.Calibrator`.

Polygraphy still ships a `Calibrator` that wraps TensorRT's implicit-quantization
flow -- feed it a data loader, it handles the device copies and the calibration
cache. **On TensorRT 11 none of that works**, because the API it wraps was
removed along with weak typing:

    trt.BuilderFlag.INT8            gone
    trt.IInt8Calibrator             gone
    trt.IInt8EntropyCalibrator2     gone

`case_the_calibrator_api_is_gone` shows the exact failures, because a dead API
that still imports cleanly is worse than one that does not exist.

The replacement is **explicit quantization**: `QuantizeLinear` / `DequantizeLinear`
pairs are baked into the ONNX by a quantization toolkit (NVIDIA ModelOpt,
`pytorch-quantization`, or QAT), and TensorRT honours what the graph says --
exactly like FP16 moved from a builder flag to a network declaration in
`../05-BuildNetworkByHand/`.

`00-Data/model/model-trained-int8-qat.onnx` is such a model and is used below.
"""

import numpy as np
import onnx
import tensorrt as trt
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, TrtRunner
from polygraphy.logger import G_LOGGER

from tensorrt_cookbook import case_mark, cookbook_path

G_LOGGER.module_severity = G_LOGGER.ERROR

float_onnx = str(cookbook_path("00-Data", "model", "model-trained.onnx"))
qat_onnx = str(cookbook_path("00-Data", "model", "model-trained-int8-qat.onnx"))
data = {"x": np.load(cookbook_path("00-Data", "data", "InferenceData.npy"))}

def calibration_data():
    """What a calibrator used to be fed: a few representative batches."""
    for _ in range(4):
        yield {"x": np.random.rand(1, 1, 28, 28).astype(np.float32)}

@case_mark
def case_the_calibrator_api_is_gone() -> None:
    """Every entry point of the old flow, and how each one fails now.

    Worth listing individually: the imports all still succeed, so the code looks
    fine until it runs. Polygraphy's own `Calibrator` fails deepest -- it raises
    from inside TensorRT rather than reporting a missing feature.
    """
    for name in ["INT8"]:
        print(f"    trt.BuilderFlag.{name:<4}         : {'present' if hasattr(trt.BuilderFlag, name) else 'REMOVED'}")
    for name in ["IInt8Calibrator", "IInt8EntropyCalibrator2", "IInt8MinMaxCalibrator"]:
        print(f"    trt.{name:<24}: {'present' if hasattr(trt, name) else 'REMOVED'}")

    try:
        EngineFromNetwork(NetworkFromOnnxPath(float_onnx), config=CreateConfig(int8=True))()
        print("    CreateConfig(int8=True)      : unexpectedly accepted")
    except Exception as e:
        print(f"    CreateConfig(int8=True)      : {type(e).__name__}: {str(e).splitlines()[0][:62]}")

    try:
        from polygraphy.backend.trt import Calibrator
        calibrator = Calibrator(data_loader=calibration_data())
        EngineFromNetwork(NetworkFromOnnxPath(float_onnx), config=CreateConfig(calibrator=calibrator))()
        print("    CreateConfig(calibrator=...) : unexpectedly accepted")
    except Exception as e:
        print(f"    CreateConfig(calibrator=...) : {type(e).__name__}: {str(e).splitlines()[0][:62]}")
    print("    the import of `Calibrator` succeeds; only using it fails")
    return

@case_mark
def case_quantization_lives_in_the_graph_now() -> None:
    """The replacement: Q/DQ nodes carry the scales, TensorRT reads them.

    Nothing is passed to the builder. The difference between the two engines
    below is entirely a property of the two ONNX files.
    """
    for tag, path in [("float model", float_onnx), ("QAT model  ", qat_onnx)]:
        model = onnx.load(path, load_external_data=False)
        n_quantize = sum(1 for node in model.graph.node if node.op_type == "QuantizeLinear")
        n_dequantize = sum(1 for node in model.graph.node if node.op_type == "DequantizeLinear")
        print(f"    {tag}: {len(model.graph.node):3d} nodes, QuantizeLinear {n_quantize}, DequantizeLinear {n_dequantize}")
    print("    the scales are initializers in the QAT file -- no calibration step at build time")
    return

@case_mark
def case_build_and_compare() -> None:
    """Build both, then ask how far apart they are.

    `CompareFunc.simple` with a default tolerance is the wrong tool here: INT8 is
    *supposed* to differ. The question for a quantized model is whether it still
    ranks the classes the same way, which is what `../02-ComparingBackends/`
    reaches for `PostprocessFunc.top_k` plus `CompareFunc.indices` to answer.
    """
    output = {}
    for tag, path in [("float", float_onnx), ("qat", qat_onnx)]:
        with TrtRunner(EngineFromNetwork(NetworkFromOnnxPath(path), config=CreateConfig()), name=tag) as runner:
            output[tag] = {k: np.array(v) for k, v in runner.infer(data).items()}
        print(f"    {tag:<5} engine built, outputs {[f'{k}{tuple(v.shape)}' for k, v in output[tag].items()]}")

    logit_difference = np.abs(output["float"]["y"] - output["qat"]["y"]).max()
    same_argmax = np.array_equal(output["float"]["z"], output["qat"]["z"])
    print(f"    max |float - qat| on logits : {logit_difference:.4f}   <- INT8 is supposed to differ")
    print(f"    same predicted class        : {same_argmax}   <- the question that actually matters")
    return

if __name__ == "__main__":
    case_the_calibrator_api_is_gone()
    case_quantization_lives_in_the_graph_now()
    case_build_and_compare()

    print("\nFinish")
