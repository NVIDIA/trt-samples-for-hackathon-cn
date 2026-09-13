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
"""Comparing one model across backends, and picking a comparison that means something.

`Comparator.run` executes several runners on the same synthetic inputs;
`Comparator.compare_accuracy` decides whether their outputs agree. The second
half is where the thinking is: `CompareFunc.simple` is a threshold on a number
somebody has to choose, and choosing it badly turns the check into either a
rubber stamp or a permanent red light.

The model here is the cookbook MNIST network, whose real TensorRT-vs-onnxruntime
difference is **7.45e-09** -- ordinary FP32 rounding. Every comparison below is
run against that same number, so the differences between them are visible rather
than asserted.
"""

import numpy as np
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, TrtRunner
from polygraphy.comparator import Comparator, CompareFunc, PostprocessFunc
from polygraphy.logger import G_LOGGER

from tensorrt_cookbook import case_mark, cookbook_path

G_LOGGER.module_severity = G_LOGGER.ERROR

onnx_file = str(cookbook_path("00-Data", "model", "model-trained.onnx"))

def run_both() -> object:
    """Run the same model through TensorRT and onnxruntime on identical inputs."""
    return Comparator.run([
        TrtRunner(EngineFromNetwork(NetworkFromOnnxPath(onnx_file), config=CreateConfig()), name="trt"),
        OnnxrtRunner(SessionFromOnnx(onnx_file), name="ort"),
    ])

def verdict(results, tag: str, compare_func) -> None:
    """Print PASS/FAIL for one comparison function."""
    try:
        passed = bool(Comparator.compare_accuracy(results, compare_func=compare_func))
        print(f"      {tag:<40}: {'PASS' if passed else 'FAIL'}")
    except Exception as e:
        print(f"      {tag:<40}: ERROR {type(e).__name__}: {str(e).splitlines()[0][:50]}")
    return

@case_mark
def case_run_and_measure() -> None:
    """What `Comparator.run` gives back, and how far the two backends actually are.

    `RunResults` maps a runner name to a list of `IterationResult`, one per
    iteration. Reading the arrays out directly is the only way to know what the
    tolerances below are being compared against.
    """
    global results
    results = run_both()
    print(f"    runners: {list(results.keys())}, {len(results['trt'])} iteration(s) each")
    for name in results["trt"][0].keys():
        trt_value = np.asarray(results["trt"][0][name])
        ort_value = np.asarray(results["ort"][0][name])
        if trt_value.dtype.kind == "f":
            print(f"    output {name}: {trt_value.dtype}, max |trt - ort| = {np.abs(trt_value - ort_value).max():.3e}")
        else:
            print(f"    output {name}: {trt_value.dtype}, identical = {np.array_equal(trt_value, ort_value)}")
    print("    that float difference is FP32 rounding, not a bug -- the rest of this example")
    print("    is about tolerances that can tell the two apart")
    return

@case_mark
def case_simple_is_a_threshold_you_choose() -> None:
    """`CompareFunc.simple` compares element-wise against `atol` / `rtol`.

    The default is not "correct"; it is a guess. Straddling the real difference
    shows how little it takes to flip the verdict, which is why a `simple` check
    that passes says nothing until you know what number it used.
    """
    verdict(results, "simple(atol=1e-8)  above the real diff", CompareFunc.simple(atol=1e-8))
    verdict(results, "simple(atol=1e-9)  below the real diff", CompareFunc.simple(atol=1e-9))
    return

@case_mark
def case_metrics_judge_the_whole_tensor() -> None:
    """The alternatives score the tensor instead of thresholding every element.

    `distance_metrics` reports L2 norm and cosine similarity; `quality_metrics`
    reports PSNR and SNR (a signal-processing view, useful for images);
    `perceptual_metrics` is stricter still and takes no arguments here.

    They are not automatically more lenient -- tightened past the real difference
    they fail too. What they buy is a judgement about the tensor as a whole
    rather than about its single worst element.
    """
    verdict(results, "distance_metrics(l2=1e-5, cos=0.99)", CompareFunc.distance_metrics(l2_tolerance=1e-5, cosine_similarity_threshold=0.99))
    verdict(results, "distance_metrics(l2=1e-12, cos=0.99)", CompareFunc.distance_metrics(l2_tolerance=1e-12, cosine_similarity_threshold=0.99))
    verdict(results, "quality_metrics(psnr=50, snr=25)", CompareFunc.quality_metrics(psnr_tolerance=50.0, snr_tolerance=25.0))
    verdict(results, "quality_metrics(psnr=300, snr=25)", CompareFunc.quality_metrics(psnr_tolerance=300.0, snr_tolerance=25.0))
    # `perceptual_metrics` needs the `lpips` package, which is not installed here.
    # Polygraphy logs the missing import and the call still returns a verdict, so
    # the verdict is reported alongside that caveat rather than as a result.
    try:
        import lpips  # noqa: F401
        verdict(results, "perceptual_metrics()", CompareFunc.perceptual_metrics())
    except ImportError:
        print("      perceptual_metrics()                    : SKIPPED, needs `pip install lpips`")
        print("      (it does return a verdict without the package, but that verdict means nothing)")
    return

@case_mark
def case_indices_needs_indices() -> None:
    """The trap: `CompareFunc.indices` is for Top-K outputs, not for logits.

    It compares two results *containing indices*, so applying it to a result that
    still holds float logits compares those floats as if they were index values
    and fails -- even though the model's own `argmax` output is bit-identical
    between the two backends.

    The intended pairing is `PostprocessFunc.top_k` first, then `indices`: that
    answers "do the backends rank the classes the same way", which is usually the
    question that matters for a classifier, and is immune to the FP32 rounding
    that `simple` is so sensitive to.
    """
    verdict(results, "indices() straight on the raw outputs", CompareFunc.indices(index_tolerance=0))

    # Scoped to `y`: `z` is already 1-D argmax output and `top_k` on axis 1 of a
    # 1-D array raises. A dict keyed by output name is how that is expressed.
    top_k = Comparator.postprocess(run_both(), PostprocessFunc.top_k(k={"y": (5, 1)}))
    verdict(top_k, "indices() after PostprocessFunc.top_k", CompareFunc.indices(index_tolerance=0))
    print("    the second one is the useful question: same ranking, regardless of rounding")
    return

@case_mark
def case_fp16_is_refused_not_ignored() -> None:
    """`CreateConfig(fp16=True)` raises on TensorRT 11 instead of doing nothing.

    Networks parsed from ONNX come out **strongly typed**, so the builder has no
    freedom to pick a precision and the flag is meaningless. Polygraphy rejects
    it outright.

    Worth contrasting with Torch-TensorRT, where the equivalent
    `enabled_precisions={torch.float16}` is accepted, ignored, and never warned
    about -- see `06-DLFrameworkTRT/Torch-TensorRT/MixedPrecisionAutocast/`.
    Same underlying change in TensorRT 11, opposite ergonomics.
    """
    # The flag is only validated when the config is actually applied to a
    # network, so this has to be a real build attempt rather than a bare call.
    try:
        EngineFromNetwork(NetworkFromOnnxPath(onnx_file), config=CreateConfig(fp16=True))()
        print("    unexpectedly accepted")
    except Exception as e:
        print(f"    {type(e).__name__}: {str(e).splitlines()[0][:96]}")
    print("    loud refusal beats silent no-op")
    return

if __name__ == "__main__":
    case_run_and_measure()
    case_simple_is_a_threshold_you_choose()
    case_metrics_judge_the_whole_tensor()
    case_indices_needs_indices()
    case_fp16_is_refused_not_ignored()

    print("\nFinish")
