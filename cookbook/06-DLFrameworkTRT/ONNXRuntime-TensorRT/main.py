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

import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL_FILE = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "model" / "model-trained.onnx"
DATA_FILE = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data" / "InferenceData.npy"
WARMUP = 50
STEPS = 200

def assert_session_uses_provider(session: ort.InferenceSession, expected_provider: str) -> None:
    active_providers = session.get_providers()
    if expected_provider not in active_providers:
        raise RuntimeError(f"Expect session to use {expected_provider}, but active providers are {active_providers}. "
                           "This usually means ONNX Runtime fell back because CUDA/TensorRT runtime libraries are missing.")

def build_session(provider_name: str, trt_engine_cache_dir: Path | None = None) -> ort.InferenceSession:
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3

    providers: list[str | tuple[str, dict[str, int | str]]] = []
    if provider_name == "TensorrtExecutionProvider":
        trt_provider_options: dict[str, int | str] = {
            "trt_fp16_enable": 0,
            "trt_engine_cache_enable": 1,
            "trt_engine_cache_path": str(trt_engine_cache_dir),
        }
        providers.append(("TensorrtExecutionProvider", trt_provider_options))
        providers.append("CUDAExecutionProvider")
    elif provider_name == "CUDAExecutionProvider":
        providers.append("CUDAExecutionProvider")
    else:
        raise ValueError(f"Unsupported provider_name={provider_name}")

    return ort.InferenceSession(str(MODEL_FILE), sess_options=sess_options, providers=providers)

def run_once(session: ort.InferenceSession, data: np.ndarray) -> list[np.ndarray]:
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: data})

def benchmark(session: ort.InferenceSession, data: np.ndarray, warmup: int = WARMUP, steps: int = STEPS) -> float:
    for _ in range(warmup):
        _ = run_once(session, data)

    t0 = time.perf_counter()
    for _ in range(steps):
        _ = run_once(session, data)
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / steps

def assert_provider_available(provider_name: str) -> None:
    available = ort.get_available_providers()
    if provider_name not in available:
        raise RuntimeError(f"{provider_name} is not available in this ONNX Runtime build. "
                           f"Available providers: {available}")

def main() -> None:
    assert_provider_available("CUDAExecutionProvider")
    assert_provider_available("TensorrtExecutionProvider")

    input_data = np.load(DATA_FILE).astype(np.float32)

    cache_dir = Path("trt_engine_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cuda_session = build_session("CUDAExecutionProvider")
    trt_session = build_session("TensorrtExecutionProvider", trt_engine_cache_dir=cache_dir)

    assert_session_uses_provider(cuda_session, "CUDAExecutionProvider")
    assert_session_uses_provider(trt_session, "TensorrtExecutionProvider")

    cuda_outputs = run_once(cuda_session, input_data)
    trt_outputs = run_once(trt_session, input_data)

    np.testing.assert_allclose(trt_outputs[0], cuda_outputs[0], rtol=1e-2, atol=1e-2)
    np.testing.assert_array_equal(trt_outputs[1], cuda_outputs[1])

    trt_latency_ms = benchmark(trt_session, input_data)
    cuda_latency_ms = benchmark(cuda_session, input_data)

    print(f"ONNX Runtime TensorRT EP latency: {trt_latency_ms:.3f} ms")
    print(f"ONNX Runtime CUDA EP latency: {cuda_latency_ms:.3f} ms")
    faster = "TensorRT EP" if trt_latency_ms < cuda_latency_ms else "CUDA EP"
    print(f"Lower latency: {faster}")
    print("Finish")

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise RuntimeError(f"{error}\n"
                           "Tips:\n"
                           "1) Install CUDA 12.x and cuDNN 9.x that match your onnxruntime-gpu build.\n"
                           "2) Ensure TensorRT runtime libraries are installed.\n"
                           "3) Export runtime paths, e.g. LD_LIBRARY_PATH includes CUDA and TensorRT lib directories.") from error
