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
"""Exercise the client half of `main.py` without a Triton Inference Server.

`tritonserver` only exists inside the Triton container, so on a plain TensorRT container stage 3 of
`main.py` skips itself and stays untested — which is how examples rot. This test replaces the
server with a stub that speaks just enough KServe v2 to drive the real client code:

+ `/v2/health/ready` answers 503 twice before 200, so the readiness polling is actually exercised;
+ `/v2/models/<name>` returns metadata;
+ `/v2/models/<name>/infer` parses the request and answers with the reference outputs, so the
  request encoding, the response decoding and the comparison in `main.py` are all covered.

What it cannot cover is whether Triton accepts `config.pbtxt`; for that, `main.py` has to be run
inside the Triton container. As a partial stand-in, the last case cross-checks `config.pbtxt`
against the engine's own tensor shapes.

Returns a non-zero exit code on any failure.
"""

import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

import main as deploy

N_SAMPLE = 4
REFERENCE = {
    "y": np.arange(N_SAMPLE * 10, dtype=np.float32).reshape(N_SAMPLE, 10),
    "z": np.full([N_SAMPLE], 8, dtype=np.int64),
}

n_fail = 0
received_request = {}

def check(b_ok: bool, message: str) -> None:
    """Print one check line and count the failures."""
    global n_fail
    n_fail += not b_ok
    print(f"    [{'PASS' if b_ok else 'FAIL'}] {message}")
    return

class StubTritonHandler(BaseHTTPRequestHandler):
    """The smallest server that the client of `main.py` cannot tell from Triton."""

    n_health_request = 0

    def log_message(self, format, *args):  # noqa: A002 - signature fixed by the base class
        """Silence the default per-request logging."""
        return

    def _reply(self, code: int, body: dict | None = None) -> None:
        payload = json.dumps(body or {}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        """Readiness and metadata."""
        if self.path == "/v2/health/ready":
            StubTritonHandler.n_health_request += 1
            # Not ready for the first two polls, so `wait_for_server` has to loop
            self._reply(200 if StubTritonHandler.n_health_request > 2 else 503)
        elif self.path == f"/v2/models/{deploy.MODEL_NAME}":
            self._reply(200, {
                "name": deploy.MODEL_NAME,
                "platform": "tensorrt_plan",
                "inputs": [{
                    "name": "x",
                    "datatype": "FP32",
                    "shape": [-1, 1, 28, 28]
                }],
                "outputs": [{
                    "name": "y",
                    "datatype": "FP32",
                    "shape": [-1, 10]
                }, {
                    "name": "z",
                    "datatype": "INT64",
                    "shape": [-1]
                }],
            })
        else:
            self._reply(404)

    def do_POST(self) -> None:
        """Inference: record what the client sent, answer with the reference."""
        if self.path != f"/v2/models/{deploy.MODEL_NAME}/infer":
            self._reply(404)
            return
        body = self.rfile.read(int(self.headers["Content-Length"]))
        received_request.update(json.loads(body))
        self._reply(200, {
            "model_name": deploy.MODEL_NAME,
            "outputs": [
                {
                    "name": "y",
                    "datatype": "FP32",
                    "shape": [N_SAMPLE, 10],
                    "data": REFERENCE["y"].reshape(-1).tolist()
                },
                {
                    "name": "z",
                    "datatype": "INT64",
                    "shape": [N_SAMPLE, 1],
                    "data": REFERENCE["z"].reshape(-1).tolist()
                },
            ],
        })

class StubProcess:
    """Stands in for the `subprocess.Popen` of `tritonserver`, so the teardown path is exercised too."""

    def __init__(self, server: HTTPServer) -> None:
        self.server = server
        self.returncode = None
        self.b_terminated = False

    def poll(self):
        """`None` while running, the exit code afterwards, same contract as `Popen`."""
        return self.returncode

    def terminate(self) -> None:
        """What `case_serve_and_query` calls in its `finally` block."""
        self.b_terminated = True
        self.server.shutdown()
        self.returncode = 0

    def wait(self, timeout=None):
        """The server is already down by the time this is called."""
        return self.returncode

    def kill(self) -> None:
        """Only reached if `terminate` timed out, which cannot happen here."""
        self.terminate()

def case_client_against_stub() -> None:
    """Drive `case_serve_and_query` against the stub and check what crossed the wire."""
    print("case_client_against_stub")
    server = HTTPServer((deploy.API_HOST, deploy.HTTP_PORT), StubTritonHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    process = StubProcess(server)
    # Replace only the two things that need a real Triton: finding and starting the binary.
    deploy.find_tritonserver = lambda: "/stub/tritonserver"
    deploy.start_server = lambda binary: process

    input_data = np.ascontiguousarray(np.random.default_rng(31193).random([N_SAMPLE, 1, 28, 28]).astype(np.float32))
    deploy.case_serve_and_query(input_data, REFERENCE)

    check(StubTritonHandler.n_health_request > 2, f"readiness was polled until ready ({StubTritonHandler.n_health_request} requests)")
    check(bool(received_request), "an inference request was sent")
    if received_request:
        tensor = received_request["inputs"][0]
        check(tensor["name"] == "x", f"input name is `x`, got `{tensor['name']}`")
        check(tensor["datatype"] == "FP32", f"input datatype is FP32, got {tensor['datatype']}")
        check(tensor["shape"] == [N_SAMPLE, 1, 28, 28], f"input shape carries the batch dimension, got {tensor['shape']}")
        check(len(tensor["data"]) == input_data.size, f"input data is flattened, got {len(tensor['data'])} of {input_data.size}")
        check(np.allclose(np.array(tensor["data"], dtype=np.float32), input_data.reshape(-1)), "input data survived the JSON round trip")
        check([o["name"] for o in received_request["outputs"]] == ["y", "z"], "both outputs were requested")
    check(process.b_terminated, "the server was terminated in the `finally` block")
    server.server_close()
    return

def case_config_matches_engine() -> None:
    """`config.pbtxt` must describe the same tensors as the engine, minus the batch dimension."""
    print("case_config_matches_engine")
    config_file = deploy.MODEL_REPOSITORY / deploy.MODEL_NAME / "config.pbtxt"
    plan_file = deploy.MODEL_REPOSITORY / deploy.MODEL_NAME / str(deploy.MODEL_VERSION) / "model.plan"
    if not config_file.exists() or not plan_file.exists():
        print("    skipped, run `python3 main.py` first")
        return
    try:
        import tensorrt as trt
    except ImportError as e:
        print(f"    skipped, {e}")
        return

    text = config_file.read_text()
    check(f'name: "{deploy.MODEL_NAME}"' in text, "config name matches the directory name")
    check('platform: "tensorrt_plan"' in text, "platform is `tensorrt_plan`")
    check(plan_file.name == "model.plan", "the plan file is called `model.plan`, as the platform requires")

    max_batch_size = int(re.search(r"max_batch_size:\s*(\d+)", text).group(1))
    engine = trt.Runtime(trt.Logger(trt.Logger.Severity.ERROR)).deserialize_cuda_engine(plan_file.read_bytes())
    profile_shape = engine.get_tensor_profile_shape("x", 0)
    check(max_batch_size <= profile_shape[2][0], f"max_batch_size {max_batch_size} is within the profile max {profile_shape[2][0]}")

    # With `max_batch_size > 0` Triton owns the leading dimension, so every `dims` in the config is
    # the shape of one sample. Getting this wrong is the classic `config.pbtxt` mistake.
    for name, dims in re.findall(r'name:\s*"(\w+)"\s*\n\s*data_type:\s*\w+\s*\n\s*dims:\s*\[([^\]]*)\]', text):
        shape_in_config = [int(x) for x in dims.split(",") if x.strip()]
        shape_in_engine = list(engine.get_tensor_shape(name))
        b_ok = shape_in_config == shape_in_engine[1:] or (name == "z" and shape_in_config == [1] and len(shape_in_engine) == 1)
        check(b_ok, f"`{name}`: config dims {shape_in_config} vs engine shape {shape_in_engine} (batch dimension excluded)")
    return

if __name__ == "__main__":
    case_client_against_stub()
    case_config_matches_engine()
    print(f"\n{'Finish' if n_fail == 0 else f'{n_fail} check(s) FAILED'}")
    sys.exit(1 if n_fail else 0)
