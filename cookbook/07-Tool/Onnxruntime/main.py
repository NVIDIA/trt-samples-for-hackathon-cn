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
"""ONNX Runtime, for people who are actually deploying with TensorRT.

The reason a TensorRT cookbook has an onnxruntime directory is not that ORT is a competitor - it is
that ORT is the **reference you check TensorRT against**, and the thing whose Execution Provider
mechanism people confuse with TensorRT itself. So this example is about the seams:

+ which Execution Provider you *asked* for versus the one you *got* (they differ, silently);
+ ORT as the golden output for a TensorRT engine, and what the remaining difference actually is;
+ ORT's own graph optimizer, and why its output must never be handed to TensorRT.

+ Steps to run.

```bash
python3 main.py
```
"""

import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import tensorrt as trt

from tensorrt_cookbook import TRTWrapperV1, case_mark, cookbook_path, parse_onnx

# ORT logs one `pthread_setaffinity_np failed` line *per thread* on a machine whose CPU affinity it
# cannot set - 100+ red lines before any output. Its own message says how to stop it: set the thread
# count explicitly. Every SessionOptions below does, and this is why.
INTRA_OP_NUM_THREADS = 8
onnxruntime.set_default_logger_severity(3)  # 3 = ERROR; the affinity lines are logged as errors

# ONNX Runtime and TensorRT both write diagnostics straight to stderr, unbuffered, while this
# script's stdout turns block-buffered the moment it is redirected to a log file. Without this the
# whole log opens with a wall of their messages, detached from the case that provoked each one.
sys.stdout.reconfigure(line_buffering=True)

onnx_file = cookbook_path("00-Data", "model", "model-trained.onnx")
data_path = cookbook_path("00-Data", "data", "InferenceData.npy")
input_data = np.load(data_path)
data = {"x": input_data}

output_optimized_file = "model_ort_optimized.onnx"
profile_file_prefix = "ort_profile"

def make_session_options(**kwargs):
    """SessionOptions with the thread count pinned, plus whatever the caller wants."""
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = INTRA_OP_NUM_THREADS
    for key, value in kwargs.items():
        setattr(session_options, key, value)
    return session_options

def try_provider(provider_name):
    """Ask for one Execution Provider and find out what actually happened.

    Asking is not getting, and the failure has two shapes that look nothing alike. A provider whose
    shared library will not load is swapped for CPU at *session creation* (a warning, not an
    exception). A provider that loads but has no kernels for this GPU gets all the way to *run*
    time and raises there. Both were observed on this machine, from two ONNX Runtime builds.
    Only creating the session, running it, and *then* reading `get_providers()` catches both.
    """
    try:
        session = onnxruntime.InferenceSession(str(onnx_file), make_session_options(), providers=[provider_name])
    except Exception as e:
        return "session failed", type(e).__name__, None
    actual_provider_list = session.get_providers()
    try:
        session.run(None, data)
    except Exception as e:
        return "run failed", f"{type(e).__name__}: {str(e).splitlines()[0][-70:]}", None
    if provider_name not in actual_provider_list:
        return "silently fell back", f"got {actual_provider_list}", None
    return "ok", "", session

def measure_latency(session, num_warmup=5, num_run=30):
    """Median wall-clock latency of one `run`, in milliseconds."""
    for _ in range(num_warmup):
        session.run(None, data)
    duration_list = []
    for _ in range(num_run):
        start = time.perf_counter()
        session.run(None, data)
        duration_list.append((time.perf_counter() - start) * 1000)
    return float(np.median(duration_list))

@case_mark
def case_providers():
    """Which Execution Provider you asked for, and which one you got.

    `get_available_providers()` lists what the *wheel was compiled with*, not what can run here.
    A provider whose dependencies are missing is dropped and CPU is used instead - so a script that
    "uses the TensorRT EP" can be running entirely on the CPU and still produce correct answers,
    only slower. Nothing raises. `get_providers()` on the session is the ground truth.
    """
    print(f"    onnxruntime {onnxruntime.__version__}, get_device() = {onnxruntime.get_device()}")
    print(f"    compiled with: {onnxruntime.get_available_providers()}")
    print(f"    this container has TensorRT {trt.__version__}")

    working_session_dict = {}
    for provider_name in onnxruntime.get_available_providers():
        status, detail, session = try_provider(provider_name)
        print(f"    {provider_name:28s} -> {status}{': ' + detail if detail else ''}")
        if session is not None:
            working_session_dict[provider_name] = session

    assert "CPUExecutionProvider" in working_session_dict, "even the CPU provider does not work"

    if "TensorrtExecutionProvider" not in working_session_dict:
        _explain_tensorrt_ep_mismatch()

    print("\n    latency of the providers that actually run (median of 30):")
    for provider_name, session in working_session_dict.items():
        print(f"        {provider_name:28s} {measure_latency(session):7.3f} ms")
    print("    -> always assert on `session.get_providers()`. A benchmark that silently ran on CPU is the")
    print("       single most common way an 'ORT with TensorRT EP is slow' report gets produced.")
    return working_session_dict

def _explain_tensorrt_ep_mismatch():
    """Why the TensorRT EP did not load: it is pinned to a TensorRT *major* version.

    ORT's error text blames PATH / LD_LIBRARY_PATH / GPU support, which sends people looking in the
    wrong place. The real constraint is in the provider library's own DT_NEEDED entries, and it is
    worth printing because it is checkable rather than guessable.
    """
    provider_library = Path(onnxruntime.__file__).parent / "capi" / "libonnxruntime_providers_tensorrt.so"
    if not provider_library.exists():
        return
    process = subprocess.run(["ldd", str(provider_library)], capture_output=True, text=True)
    missing_line_list = [line.strip() for line in process.stdout.splitlines() if "not found" in line]
    print(f"\n    Why: {provider_library.name} cannot resolve its own dependencies.")
    for line in missing_line_list:
        print(f"        {line}")
    print(f"    Two independent major-version mismatches, and the sonames are not compatible across either:")
    print(f"        TensorRT - the wheel wants `.so.10`, this container provides libnvinfer.so.{trt.__version__.split('.')[0]}")
    print(f"        CUDA     - the wheel is a `cu12` build, this container is CUDA {'.'.join(str(x) for x in _cuda_toolkit_version())}")
    print("    ORT's own message blames PATH / LD_LIBRARY_PATH / GPU support, which is misleading: no amount")
    print("    of PATH fixing produces a `.so.10` or a `.so.12` that is not installed here.")
    print("    Nor is upgrading ORT the answer - `pip install -e .` pins this version, because")
    print("    `nvidia-modelopt[onnx]` requires `onnxruntime-gpu~=1.24.2`. Newer ORT wheels are still `cu12`")
    print("    and still TensorRT 10, so on this stack the GPU providers need a source build of ORT.")
    print("    This affects only ORT's *embedded* use of TensorRT; TensorRT itself, and every other example")
    print("    here, is unaffected - `case_reference_for_tensorrt` builds a real engine in this same process.")

def _cuda_toolkit_version() -> tuple:
    """CUDA toolkit version of the container, from `nvcc`, or ('?',) if nvcc is absent."""
    process = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    if process.returncode != 0:
        return ("?", )
    for token in process.stdout.split():
        if token.startswith("V") and token[1:2].isdigit():
            return tuple(token[1:].split(".")[:2])
    return ("?", )

@case_mark
def case_basic_inference():
    """The plain path, with the model's own I/O metadata read off the session."""
    session = onnxruntime.InferenceSession(str(onnx_file), make_session_options(), providers=["CPUExecutionProvider"])

    for i, tensor in enumerate(session.get_inputs()):
        print(f"    Input  {i:2d}: {tensor.name}, {tensor.shape}, {tensor.type}")
    for i, tensor in enumerate(session.get_outputs()):
        print(f"    Output {i:2d}: {tensor.name}, {tensor.shape}, {tensor.type}")

    output_name_list = ["y", "z"]
    output_list = session.run(output_name_list, data)
    # `zip`, not `output_name_list, output_list` - the latter iterates over a 2-tuple of lists and
    # prints the two *names* as one "pair". It is a real bug this example used to have.
    for name, tensor in zip(output_name_list, output_list):
        print(f"    {name}: {np.array2string(tensor, precision=4, max_line_width=120)}")
    print(f"    -> `{session.get_inputs()[0].shape[0]}` in the input shape is a *symbol*, not a number; ORT")
    print("       reads it straight from the ONNX file and accepts any batch (see `case_dynamic_shape`)")

@case_mark
def case_reference_for_tensorrt():
    """ORT as the golden output for a TensorRT engine - and what the leftover difference is.

    This is the actual job ORT does in a TensorRT workflow. Both consume the same ONNX file, so a
    disagreement is TensorRT's build choices rather than a modelling difference. The size of that
    disagreement surprises people, so it is worth attributing rather than waving at.
    """
    session = onnxruntime.InferenceSession(str(onnx_file), make_session_options(), providers=["CPUExecutionProvider"])
    reference_y, reference_z = session.run(["y", "z"], data)

    shape = list(input_data.shape)
    for use_tf32 in [True, False]:
        tw = TRTWrapperV1()
        parse_onnx(onnx_file, tw.logger, tw.network, tw.builder_config)
        tw.profile.set_shape("x", shape, shape, [8] + shape[1:])
        if not use_tf32:
            tw.builder_config.clear_flag(trt.BuilderFlag.TF32)
        tw.build()
        tw.setup(data, b_print_io=False)
        tw.infer(b_print_io=False)

        difference = float(np.abs(reference_y - tw.buffer["y"][0]).max())
        relative = difference / float(np.abs(reference_y).max())
        label = "default (TF32 allowed)" if use_tf32 else "TF32 cleared"
        print(f"    {label:24s} max|ORT - TRT| = {difference:.3e}  ({relative:.2e} relative), argmax equal: {bool((reference_z == tw.buffer['z'][0]).all())}")
        if use_tf32:
            difference_tf32 = difference
        else:
            difference_fp32 = difference

    print(f"    -> clearing one flag moved the disagreement by {difference_tf32 / difference_fp32:.0f}x.")
    print("       `BuilderFlag.TF32` is on by default, so a TensorRT engine built with no precision flags at")
    print("       all is *not* doing FP32 matmuls - it is doing TF32 ones on the tensor cores, with a 10-bit")
    print("       mantissa. Before blaming a conversion bug for a 1e-3 discrepancy, clear TF32 and re-measure:")
    print("       if the gap collapses, there is no bug, and the remaining ~1e-5 is ordinary reassociation.")

@case_mark
def case_graph_optimization():
    """ORT rewrites the graph before running it, and will write the rewrite out for you.

    `optimized_model_filepath` is the best way to see what a runtime *actually* executes. The trap
    is what people do with the file next.
    """
    session_options = make_session_options(
        graph_optimization_level=onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL,
        optimized_model_filepath=output_optimized_file,
    )
    onnxruntime.InferenceSession(str(onnx_file), session_options, providers=["CPUExecutionProvider"])

    original_model = onnx.load(onnx_file)
    optimized_model = onnx.load(output_optimized_file)
    print(f"    original : {len(original_model.graph.node):2d} nodes {[n.op_type for n in original_model.graph.node]}")
    print(f"    optimized: {len(optimized_model.graph.node):2d} nodes {[n.op_type for n in optimized_model.graph.node]}")

    non_standard_domain_list = sorted({n.domain for n in optimized_model.graph.node if n.domain not in ("", "ai.onnx")})
    print(f"    the three `Relu`s are gone (folded into Conv / Gemm), and `FusedGemm` + `ReorderOutput` appeared")
    print(f"    domains now in use: {non_standard_domain_list or ['(none)']}")

    logger = trt.Logger(trt.Logger.INTERNAL_ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network()
    onnx_parser = trt.OnnxParser(network, logger)
    ok = onnx_parser.parse(optimized_model.SerializeToString())
    error_list = [onnx_parser.get_error(i).desc().splitlines()[0] for i in range(onnx_parser.num_errors)]
    print(f"    TensorRT on the optimized file: ok={ok}, {error_list[0] if error_list else ''}")
    assert not ok, "TensorRT was expected to reject ORT's optimized graph"
    print("    -> **do not feed this file to TensorRT.** `com.microsoft.nchwc` ops are ONNX Runtime's private")
    print("       layout-aware kernels; to TensorRT they are unknown ops, so it reports them as missing")
    print("       plugins. Dump the optimized model to *read* it; always build TensorRT from the original.")

@case_mark
def case_profiling():
    """ORT's built-in profiler: per-node timings, as JSON, with no external tool."""
    session_options = make_session_options(enable_profiling=True, profile_file_prefix=profile_file_prefix)
    session = onnxruntime.InferenceSession(str(onnx_file), session_options, providers=["CPUExecutionProvider"])
    for _ in range(10):
        session.run(None, data)
    profile_file = session.end_profiling()

    event_list = json.loads(Path(profile_file).read_text())
    kernel_event_list = [e for e in event_list if e.get("cat") == "Node" and e["name"].endswith("_kernel_time")]
    duration_by_op = defaultdict(int)
    for event in kernel_event_list:
        duration_by_op[event["args"]["op_name"]] += event["dur"]
    total_duration = sum(duration_by_op.values())

    print(f"    {profile_file}: {len(event_list)} events, {len(kernel_event_list)} node kernel timings")
    for op_name, duration in sorted(duration_by_op.items(), key=lambda kv: -kv[1])[:5]:
        print(f"        {op_name:16s} {duration:6d} us  {duration / total_duration:5.1%}")
    print(f"    -> the op names are the *optimized* ones from the previous case, so this is what really ran.")
    print("       The file is Chrome-trace format: open it in `chrome://tracing` or Perfetto for a timeline.")

@case_mark
def case_dynamic_shape():
    """One session, many batch sizes - no optimization profile, no rebuild.

    Worth doing next to TensorRT, because the contrast is the point: TensorRT needs an
    `IOptimizationProfile` declared up front and refuses shapes outside it (see
    `../../08-Advance/MultiOptimizationProfile/`), while ORT just takes whatever arrives.
    """
    session = onnxruntime.InferenceSession(str(onnx_file), make_session_options(), providers=["CPUExecutionProvider"])
    for batch_size in [1, 4, 8, 37]:
        batched_data = np.repeat(input_data, batch_size, axis=0)
        output_y, output_z = session.run(["y", "z"], {"x": batched_data})
        print(f"    batch {batch_size:3d}: y{output_y.shape}, z{output_z.shape}")
    print("    -> 37 works as readily as 1. That flexibility is exactly the freedom TensorRT trades away for")
    print("       the ability to pick kernels and memory layouts for a known shape range.")

@case_mark
def case_io_binding():
    """`IOBinding` + `OrtValue`: place the buffers yourself instead of letting `run()` copy.

    `session.run()` takes numpy arrays and returns numpy arrays, which means a copy in and a copy
    out on every call. On a GPU provider those become host-device transfers that can dominate a
    small model. `IOBinding` is ORT's answer, and the same idea as binding device pointers with
    `IExecutionContext::setTensorAddress` in TensorRT.
    """
    session = onnxruntime.InferenceSession(str(onnx_file), make_session_options(), providers=["CPUExecutionProvider"])
    device_name = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"

    ort_value = onnxruntime.OrtValue.ortvalue_from_numpy(input_data, device_name, 0)
    print(f"    input OrtValue lives on {ort_value.device_name()}, shape {ort_value.shape()}, {ort_value.data_type()}")

    io_binding = session.io_binding()
    io_binding.bind_ortvalue_input("x", ort_value)
    for name in ["y", "z"]:
        io_binding.bind_output(name, device_name)
    session.run_with_iobinding(io_binding)
    output_list = io_binding.copy_outputs_to_cpu()

    reference_list = session.run(["y", "z"], data)
    for bound, reference in zip(output_list, reference_list):
        assert np.array_equal(bound, reference)
    print(f"    outputs via binding: {[o.shape for o in output_list]}, identical to `run()`")
    print(f"    -> on this machine the provider is {session.get_providers()[0]}, so there is nothing to save;")
    print("       the win appears when the provider is on a device and the caller already has the data there.")

if __name__ == "__main__":
    for stale_file in list(Path(".").glob("*.onnx")) + list(Path(".").glob(f"{profile_file_prefix}*.json")):
        stale_file.unlink(missing_ok=True)

    # Which Execution Provider did you actually get?
    case_providers()
    # The plain path
    case_basic_inference()
    # ORT as the golden output for a TensorRT engine
    case_reference_for_tensorrt()
    # ORT rewrites the graph, and the rewrite is not portable
    case_graph_optimization()
    # Per-node timings without an external tool
    case_profiling()
    # One session, any batch size
    case_dynamic_shape()
    # Bind the buffers yourself
    case_io_binding()

    print("Finish")
