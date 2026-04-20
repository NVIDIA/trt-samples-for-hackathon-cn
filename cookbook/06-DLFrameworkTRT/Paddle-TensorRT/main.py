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

try:
    import paddle
    from paddle import inference
except ImportError as error:
    raise RuntimeError("This sample requires PaddlePaddle. Install `paddlepaddle-gpu` first.") from error

DATA_FILE = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data" / "InferenceData.npy"
MODEL_DIR = Path("paddle_infer_model")
MODEL_FILE = MODEL_DIR / "model.pdmodel"
PARAMS_FILE = MODEL_DIR / "model.pdiparams"
INPUT_SHAPE = [1, 1, 28, 28]
WARMUP = 50
STEPS = 200

def export_inference_model_if_needed() -> None:
    if MODEL_FILE.exists() and PARAMS_FILE.exists():
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    paddle.enable_static()
    paddle.seed(97)

    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    with paddle.static.program_guard(main_program, startup_program):
        x = paddle.static.data(name="x", shape=[None, 1, 28, 28], dtype="float32")

        conv1 = paddle.static.nn.conv2d(x, num_filters=32, filter_size=5, padding=2)
        relu1 = paddle.nn.functional.relu(conv1)
        pool1 = paddle.nn.functional.max_pool2d(relu1, kernel_size=2, stride=2)

        conv2 = paddle.static.nn.conv2d(pool1, num_filters=64, filter_size=5, padding=2)
        relu2 = paddle.nn.functional.relu(conv2)
        pool2 = paddle.nn.functional.max_pool2d(relu2, kernel_size=2, stride=2)

        flatten = paddle.flatten(pool2, start_axis=1)
        fc1 = paddle.static.nn.fc(flatten, size=1024)
        relu3 = paddle.nn.functional.relu(fc1)
        logits = paddle.static.nn.fc(relu3, size=10)
        probs = paddle.nn.functional.softmax(logits, axis=1)
        pred = paddle.argmax(probs, axis=1)

    place = paddle.CUDAPlace(0)
    executor = paddle.static.Executor(place)
    executor.run(startup_program)

    paddle.static.save_inference_model(
        path_prefix=str(MODEL_DIR / "model"),
        feed_vars=[x],
        fetch_vars=[logits, pred],
        executor=executor,
        program=main_program,
    )

def build_predictor(enable_trt: bool):
    config = inference.Config(str(MODEL_FILE), str(PARAMS_FILE))
    config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
    config.switch_ir_optim(True)
    config.enable_memory_optim()

    if enable_trt:
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=3,
            precision_mode=inference.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False,
        )
        config.set_trt_dynamic_shape_info(
            {"x": INPUT_SHAPE},
            {"x": INPUT_SHAPE},
            {"x": INPUT_SHAPE},
        )

    return inference.create_predictor(config)

def run_once(predictor, data: np.ndarray) -> list[np.ndarray]:
    input_name = predictor.get_input_names()[0]
    input_handle = predictor.get_input_handle(input_name)
    input_handle.copy_from_cpu(data)

    predictor.run()

    output_values: list[np.ndarray] = []
    for output_name in predictor.get_output_names():
        output_handle = predictor.get_output_handle(output_name)
        output_values.append(output_handle.copy_to_cpu())

    return output_values

def benchmark(predictor, data: np.ndarray, warmup: int = WARMUP, steps: int = STEPS) -> float:
    for _ in range(warmup):
        _ = run_once(predictor, data)

    t0 = time.perf_counter()
    for _ in range(steps):
        _ = run_once(predictor, data)
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / steps

def main() -> None:
    if not paddle.is_compiled_with_cuda():
        raise RuntimeError("This sample requires a CUDA-enabled PaddlePaddle build.")

    export_inference_model_if_needed()

    input_data = np.load(DATA_FILE).astype(np.float32)

    gpu_predictor = build_predictor(enable_trt=False)
    trt_predictor = build_predictor(enable_trt=True)

    gpu_outputs = run_once(gpu_predictor, input_data)
    trt_outputs = run_once(trt_predictor, input_data)

    np.testing.assert_allclose(trt_outputs[0], gpu_outputs[0], rtol=1e-2, atol=1e-2)
    np.testing.assert_array_equal(trt_outputs[1], gpu_outputs[1])

    trt_latency_ms = benchmark(trt_predictor, input_data)
    gpu_latency_ms = benchmark(gpu_predictor, input_data)

    print(f"Paddle Inference TensorRT latency: {trt_latency_ms:.3f} ms")
    print(f"Paddle Inference GPU latency: {gpu_latency_ms:.3f} ms")

    faster = "TensorRT" if trt_latency_ms < gpu_latency_ms else "GPU"
    print(f"Lower latency: {faster}")
    print("Finish")

if __name__ == "__main__":
    main()
