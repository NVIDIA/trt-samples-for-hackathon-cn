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
"""Deploy a TensorRT engine on Triton Inference Server, end to end, from one script.

The four stages are:

1. build the engine and lay out the model repository Triton expects,
2. run the same input through the engine locally, to have a reference answer,
3. start `tritonserver`, wait until it reports ready, send an inference request over the
   KServe v2 HTTP protocol, and compare the reply against the reference,
4. shut the server down and leave no process behind.

Stages 1 and 2 need `tensorrt` (+ `cuda-python`); stage 3 needs the `tritonserver` binary, which
only ships inside the Triton container. Each stage checks what it needs and says what is missing
instead of failing halfway, so this script is useful in a plain TensorRT container too.

Deliberately dependency-light (no `tensorrt_cookbook`, no `tritonclient`): it is meant to be copied
into a Triton container, where neither is installed. The KServe v2 protocol is plain JSON over
HTTP, so `requests` is enough. See README.md for the `tritonclient` equivalent.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_REPOSITORY = PROJECT_ROOT / "model_repository"
MODEL_NAME = "mnist-cnn"
MODEL_VERSION = 1

API_HOST = "127.0.0.1"
HTTP_PORT, GRPC_PORT, METRICS_PORT = 8000, 8001, 8002
SERVER_LOG = PROJECT_ROOT / "tritonserver.log"

# The optimization profile of the engine, which `config.pbtxt` has to stay inside of.
MIN_BATCH, OPT_BATCH, MAX_BATCH = 1, 4, 16
N_BATCH = 4  # Batch size actually used by the request below

def case_mark(func):
    """Print a banner around a case, like `tensorrt_cookbook.case_mark` does.

    Re-implemented rather than imported: this script has to run inside the Triton container, where
    the cookbook package is not installed.
    """

    def wrapper(*args, **kwargs):
        print(f"\n{'=' * 30} Start [{func.__name__}]")
        result = func(*args, **kwargs)
        print(f"{'=' * 30} End   [{func.__name__}]")
        return result

    return wrapper

def find_onnx_file(argument: str | None) -> Path:
    """Locate the trained MNIST ONNX: the command line wins, then `TRT_COOKBOOK_PATH`, then guessing."""
    if argument is not None:
        return Path(argument)
    cookbook_path = os.environ.get("TRT_COOKBOOK_PATH")
    if cookbook_path:
        return Path(cookbook_path) / "00-Data" / "model" / "model-trained.onnx"
    return PROJECT_ROOT.parent.parent / "00-Data" / "model" / "model-trained.onnx"

# ================================ Stage 1: model repository

@case_mark
def case_build_model_repository(onnx_file: Path) -> Path:
    """Build the engine into `model_repository/<name>/<version>/model.plan` and write `config.pbtxt`.

    The layout is fixed by Triton:

    ```txt
        model_repository/
        └── mnist-cnn/
            ├── config.pbtxt
            └── 1/
                └── model.plan      <- the file name is fixed for the `tensorrt_plan` platform
    ```
    """
    import tensorrt as trt

    version_path = MODEL_REPOSITORY / MODEL_NAME / str(MODEL_VERSION)
    version_path.mkdir(parents=True, exist_ok=True)
    plan_file = version_path / "model.plan"

    logger = trt.Logger(trt.Logger.Severity.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network()  # Strongly typed by default on TensorRT 11
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_file)):
        raise RuntimeError(f"Failed to parse {onnx_file}: {parser.get_error(0)}")

    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    # Triton picks the batch size at run time, so the profile has to cover `max_batch_size` below.
    profile.set_shape("x", [MIN_BATCH, 1, 28, 28], [OPT_BATCH, 1, 28, 28], [MAX_BATCH, 1, 28, 28])
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Failed to build the engine")
    plan_file.write_bytes(engine_bytes)
    print(f"    Engine: {plan_file.relative_to(PROJECT_ROOT)} ({plan_file.stat().st_size} B)")

    # `max_batch_size > 0` makes Triton own the first dimension, so every `dims` below is the shape
    # of ONE sample. `z` is a scalar per sample, which cannot be spelled as an empty `dims`, hence
    # `dims: [1]` plus a `reshape` that drops it again: this is the usual trip-up of the format.
    config_text = f"""name: "{MODEL_NAME}"
platform: "tensorrt_plan"
max_batch_size: {MAX_BATCH}
input [
  {{
    name: "x"
    data_type: TYPE_FP32
    dims: [1, 28, 28]
  }}
]
output [
  {{
    name: "y"
    data_type: TYPE_FP32
    dims: [10]
  }},
  {{
    name: "z"
    data_type: TYPE_INT64
    dims: [1]
    reshape {{ shape: [] }}
  }}
]
instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]
"""
    config_file = MODEL_REPOSITORY / MODEL_NAME / "config.pbtxt"
    config_file.write_text(config_text)
    print(f"    Config: {config_file.relative_to(PROJECT_ROOT)}")
    print("    Repository:")
    for path in sorted(MODEL_REPOSITORY.rglob("*")):
        print(f"        {path.relative_to(MODEL_REPOSITORY.parent)}")
    return plan_file

# ================================ Stage 2: local reference

@case_mark
def case_reference_inference(plan_file: Path, input_data: np.ndarray) -> dict | None:
    """Run the engine here, without Triton, so the served answer has something to be compared to.

    Skipped rather than fatal when `cuda-python` is missing: the deployment itself does not need it.
    """
    try:
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
    except ImportError as e:
        print(f"    skipped, {e}")
        return None

    logger = trt.Logger(trt.Logger.Severity.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan_file.read_bytes())
    context = engine.create_execution_context()
    context.set_input_shape("x", input_data.shape)

    name_list = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    buffer = {}
    for name in name_list:
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        shape = context.get_tensor_shape(name)
        host = np.ascontiguousarray(input_data) if name == "x" else np.empty(shape, dtype=dtype)
        device = cudart.cudaMalloc(host.nbytes)[1]
        buffer[name] = (host, device)
        context.set_tensor_address(name, device)

    cudart.cudaMemcpy(buffer["x"][1], buffer["x"][0].ctypes.data, buffer["x"][0].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.execute_async_v3(0)
    cudart.cudaDeviceSynchronize()

    output = {}
    for name in name_list:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            host, device = buffer[name]
            cudart.cudaMemcpy(host.ctypes.data, device, host.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            output[name] = host
    for _, device in buffer.values():
        cudart.cudaFree(device)

    print(f"    y[0] = {np.array2string(output['y'][0], precision=4, max_line_width=200)}")
    print(f"    z    = {output['z'].tolist()}")
    return output

# ================================ Stage 3: serve and query

def find_tritonserver() -> str | None:
    """`tritonserver` only exists inside the Triton container, so its absence is expected here."""
    from shutil import which
    return os.environ.get("TRITONSERVER_BIN") or which("tritonserver")

def start_server(binary: str) -> subprocess.Popen:
    """Start `tritonserver` on the model repository built above, with its log going to a file."""
    command = [
        binary,
        f"--model-repository={MODEL_REPOSITORY}",
        f"--http-port={HTTP_PORT}",
        f"--grpc-port={GRPC_PORT}",
        f"--metrics-port={METRICS_PORT}",
        "--log-verbose=0",
    ]
    # The backend directory defaults to the hard-coded `/opt/tritonserver/backends`, which only
    # exists inside the container. An install unpacked somewhere else (see README.md, "without
    # docker") loads no backend at all and fails with `unable to find backend library for backend
    # 'tensorrt'`, which does not mention the directory. Derive it from the binary instead.
    backend_path = Path(binary).resolve().parent.parent / "backends"
    if backend_path.is_dir() and backend_path != Path("/opt/tritonserver/backends"):
        command.append(f"--backend-directory={backend_path}")
    print(f"    {' '.join(command)}")
    log_file = SERVER_LOG.open("w")
    return subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)

def wait_for_server(process: subprocess.Popen, timeout: int = 300) -> None:
    """Poll the KServe v2 readiness endpoint until the model is loaded.

    The process is checked on every round as well: a server that died on a bad `config.pbtxt` would
    otherwise be waited for until the timeout, and the real error is in `tritonserver.log`.
    """
    import requests

    start_time = time.time()
    while time.time() - start_time < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"tritonserver exited with code {process.returncode}, see {SERVER_LOG}")
        try:
            response = requests.get(f"http://{API_HOST}:{HTTP_PORT}/v2/health/ready", timeout=2)
            if response.status_code == 200:
                print(f"    Server ready after {time.time() - start_time:.1f} s")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"Timeout waiting for tritonserver, see {SERVER_LOG}")

def infer_over_http(input_data: np.ndarray) -> dict:
    """One inference request in the KServe v2 HTTP protocol, which is plain JSON."""
    import requests

    payload = {
        "inputs": [{
            "name": "x",
            "shape": list(input_data.shape),
            "datatype": "FP32",
            "data": input_data.reshape(-1).tolist(),
        }],
        "outputs": [{
            "name": "y"
        }, {
            "name": "z"
        }],
    }
    url = f"http://{API_HOST}:{HTTP_PORT}/v2/models/{MODEL_NAME}/infer"
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()

    output = {}
    for item in response.json()["outputs"]:
        dtype = {"FP32": np.float32, "INT64": np.int64}[item["datatype"]]
        output[item["name"]] = np.array(item["data"], dtype=dtype).reshape(item["shape"])
    return output

@case_mark
def case_serve_and_query(input_data: np.ndarray, reference: dict | None) -> None:
    """Start the server, query it, compare with the reference, then always stop the server."""
    binary = find_tritonserver()
    if binary is None:
        print("    skipped, no `tritonserver` binary in PATH (set TRITONSERVER_BIN to override).")
        print("    The model repository above is ready to be served, see README.md for how to run it:")
        print(f"        tritonserver --model-repository={MODEL_REPOSITORY}")
        return
    try:
        import requests  # noqa: F401
    except ImportError:
        print("    skipped, `requests` is not installed (pip install requests)")
        return

    process = start_server(binary)
    try:
        wait_for_server(process)

        # What the server says about itself, which is the first thing to check in a deployment
        metadata = requests.get(f"http://{API_HOST}:{HTTP_PORT}/v2/models/{MODEL_NAME}", timeout=10).json()
        print(f"    Model metadata: {json.dumps(metadata)}")

        start_time = time.time()
        output = infer_over_http(input_data)
        print(f"    Inference over HTTP: {(time.time() - start_time) * 1000:.2f} ms")
        print(f"    y[0] = {np.array2string(output['y'][0], precision=4, max_line_width=200)}")
        print(f"    z    = {output['z'].reshape(-1).tolist()}")

        if reference is not None:
            difference = float(np.max(np.abs(output["y"].reshape(-1) - reference["y"].reshape(-1))))
            b_same_class = np.array_equal(output["z"].reshape(-1), reference["z"].reshape(-1))
            print(f"    Compared with the local reference: max |diff| = {difference:.3e}, same argmax = {b_same_class}")
            assert b_same_class and difference < 1e-5, "Triton and the local engine disagree"
    finally:
        # Teardown that runs even if the request failed, so no server is left listening on 8000
        print("    Stopping the server")
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        print(f"    Server stopped with code {process.returncode}")
    return

def main() -> int:
    """Build, serve, query, clean up."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx", default=None, help="Input ONNX file, default `$TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx`")
    parser.add_argument("--no-serve", action="store_true", help="Only build the model repository, do not start a server")
    args = parser.parse_args()

    onnx_file = find_onnx_file(args.onnx)
    if not onnx_file.exists():
        print(f"Input ONNX not found: {onnx_file}")
        print("Pass --onnx, or set TRT_COOKBOOK_PATH, or run `00-Data/main.py` to create it.")
        return 1

    plan_file = case_build_model_repository(onnx_file)

    # A fixed input, so the printed numbers are the same on every run
    input_data = np.load(onnx_file.parent.parent / "data" / "InferenceData.npy") if (onnx_file.parent.parent / "data" / "InferenceData.npy").exists() else None
    if input_data is None:
        input_data = np.random.default_rng(31193).random([1, 1, 28, 28]).astype(np.float32)
    input_data = np.ascontiguousarray(np.repeat(input_data.reshape(1, 1, 28, 28), N_BATCH, axis=0).astype(np.float32))
    print(f"\nInput: {input_data.shape} {input_data.dtype}")

    reference = case_reference_inference(plan_file, input_data)
    if not args.no_serve:
        case_serve_and_query(input_data, reference)

    print("\nFinish")
    return 0

if __name__ == "__main__":
    sys.exit(main())
