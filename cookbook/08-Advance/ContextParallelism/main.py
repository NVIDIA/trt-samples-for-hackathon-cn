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
"""Context parallelism: split one attention model across GPUs along the sequence axis.

`02-API/Layer/DistCollective` shows the collective operations one at a time on a
toy tensor. This example is the whole picture: a real self-attention block is
sharded across 2 GPUs, so that each GPU owns half of the sequence, and the
`DistCollective` operations that make the maths still come out right are
inserted **into the ONNX graph** rather than written by hand.

Pipeline:

1. `case_build_model` - build a single-device self-attention ONNX with
   onnx-graphsurgeon (Q/K/V projection, RMSNorm on Q/K, scaled dot-product
   attention, output projection).
2. `case_shard_model` - `polygraphy multi-device shard` reads a `hint.json`
   describing where the attention is, and rewrites the graph with
   `ReduceScatter` / `AllGather` nodes.
3. `case_single_device` - build and run the unsharded model on one GPU, this is
   the numerical reference.
4. `case_context_parallel` - spawn one process per rank, each builds the sharded
   model, hands TensorRT a NCCL communicator via
   `IExecutionContext.set_communicator`, and runs it. Rank 0 compares its output
   against the single-device reference.

Why every rank is fed the **whole** input: the first collective is a
`ReduceScatter` with `reduce_op=max`. All ranks hold identical data, so the
reduction is the identity and the operation degenerates into "scatter the
sequence". The trailing `AllGather` puts the full output back together, so both
the single-device and the context-parallel model have the same I/O contract.

MPI is not needed: like `02-API/Layer/DistCollective`, this example re-launches
itself once per rank and passes the NCCL unique id through a file.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from ctypes import c_char_p, c_void_p, py_object, pythonapi
from pathlib import Path

import cuda.bindings.runtime as cudart
import nccl.core as nccl
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt

from tensorrt_cookbook import case_mark

np.random.seed(31193)

REQUIRED_WORLD_SIZE = 2  # Number of GPUs (= number of ranks) the model is sharded across
N_HEAD = 8  # Number of attention heads
HEAD_DIM = 64  # Width of one head
HIDDEN_DIM = N_HEAD * HEAD_DIM  # 512
N_SEQUENCE = int(os.getenv("TRT_N_SEQUENCE", "16384"))  # Sequence length, must be divisible by REQUIRED_WORLD_SIZE (`sweep.py` varies it)
N_BATCH = 1
N_WARMUP = 10
N_INFERENCE = 50
OPSET = 17

output_path = Path(__file__).parent
onnx_file_sd = output_path / "model-single_device.onnx"
onnx_file_md = output_path / "model-multi_device.onnx"
hint_file = output_path / "hint.json"
input_file = output_path / "data-input.npy"
reference_file = output_path / "data-reference.npy"
result_file = output_path / "data-result.json"

# ================================================================ ONNX model

def register_operator(op_type: str) -> None:
    """Expose one ONNX operator as a `gs.Graph` method returning its output tensor.

    This is the onnx-graphsurgeon "layer API": `graph.matmul(a, b)` instead of
    building a node and wiring tensors by hand.
    """

    def method(self, *input_list, **attribute):
        return self.layer(op=op_type, inputs=list(input_list), attrs=attribute, outputs=[f"{op_type.lower()}_out"])[0]

    method.__name__ = op_type.lower()
    setattr(gs.Graph, method.__name__, method)

for _op_type in ["MatMul", "Mul", "Div", "Add", "Pow", "Sqrt", "Cast", "Transpose", "Reshape", "Softmax", "ReduceMean"]:
    register_operator(_op_type)

def build_attention_onnx() -> None:
    """A self-attention block: [sequence, batch, hidden] -> [sequence, batch, hidden], float16.

    Shapes are kept **static** here. The sharding tool only needs to find the Q
    tensor that feeds QK^T, which is named `q_scaled` below and referenced from
    `hint.json`.
    """
    graph = gs.Graph(opset=OPSET)

    def weight(*shape):
        return (np.random.rand(*shape).astype(np.float32) * 2 - 1).astype(np.float16) * 0.05

    input_tensor = gs.Variable("input", dtype=np.float16, shape=[N_SEQUENCE, N_BATCH, HIDDEN_DIM])
    graph.inputs = [input_tensor]

    # The sequence dimension is written as `-1` on purpose. After sharding, every rank
    # only holds `N_SEQUENCE / world_size` tokens, so a hard-coded `N_SEQUENCE` here
    # makes the parser reject the sharded model with "reshape changes volume" (it sees
    # a Reshape of [8192, 1, 512] into [16384, 1, 8, 64]). The upstream
    # sample solves the same problem with a Shape/Gather/Concat chain, `-1` is 12 nodes
    # cheaper and says the same thing.
    shape_4d = np.array([-1, N_BATCH, N_HEAD, HEAD_DIM], dtype=np.int64)
    q = graph.reshape(graph.matmul(input_tensor, weight(HIDDEN_DIM, HIDDEN_DIM)), shape_4d)
    k = graph.reshape(graph.matmul(input_tensor, weight(HIDDEN_DIM, HIDDEN_DIM)), shape_4d)
    v = graph.reshape(graph.matmul(input_tensor, weight(HIDDEN_DIM, HIDDEN_DIM)), shape_4d)

    def rms_norm(x):
        # Computed in float32, exactly as a real model would
        x32 = graph.cast(x, to=onnx.TensorProto.FLOAT)
        mean = graph.reducemean(graph.pow(x32, np.array([2.0], dtype=np.float32)), axes=[-1], keepdims=1)
        rms = graph.sqrt(graph.add(mean, np.array([1e-6], dtype=np.float32)))
        normed = graph.div(x32, rms)
        return graph.mul(graph.cast(normed, to=onnx.TensorProto.FLOAT16), weight(1, 1, 1, HEAD_DIM))

    # [sequence, batch, head, head_dim] -> [batch, head, sequence, head_dim]
    q = graph.transpose(rms_norm(q), perm=[1, 2, 0, 3])
    k = graph.transpose(rms_norm(k), perm=[1, 2, 0, 3])
    v = graph.transpose(v, perm=[1, 2, 0, 3])

    # The 1/sqrt(head_dim) factor is split over Q and K so that neither product overflows float16
    scale = np.array([(1.0 / HEAD_DIM) ** 0.25], dtype=np.float16)
    q_scaled = graph.mul(q, scale)
    q_scaled.name = "q_scaled"  # Referenced by `hint.json`
    k_scaled = graph.mul(graph.transpose(k, perm=[0, 1, 3, 2]), scale)

    attention = graph.matmul(graph.softmax(graph.matmul(q_scaled, k_scaled), axis=-1), v)

    # [batch, head, sequence, head_dim] -> [sequence, batch, hidden]
    attention = graph.reshape(graph.transpose(attention, perm=[2, 0, 1, 3]), np.array([-1, N_BATCH, HIDDEN_DIM], dtype=np.int64))
    output_tensor = graph.matmul(attention, weight(HIDDEN_DIM, HIDDEN_DIM))
    output_tensor.name = "output"
    output_tensor.dtype = np.float16
    output_tensor.shape = [N_SEQUENCE, N_BATCH, HIDDEN_DIM]
    graph.outputs = [output_tensor]

    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, onnx_file_sd)
    return

# ================================================================ Thermal guard
#
# This matters more than it looks. Measured on this machine with one **identical**
# engine (same 18 layers, same 322 MiB device memory), sequence 32768:
#
#     GPU 0, already at 88-91 C : 3.6 -> 11.0 ms over five rounds, SM clock 1432 -> 352 MHz
#     GPU 2, starting at 36 C   : 2.769 - 2.815 ms, spread 1.7%,   SM clock ~1897 MHz
#
# `clocks_throttle_reasons.active` reads `0x20` (SW thermal slowdown) in the first
# case. A benchmark that ignores this does not measure TensorRT, it measures how
# long the GPU has been busy, and it will happily report a 4x difference that is
# entirely thermal. So: run on the coolest devices, wait for them, and print the
# clock next to every latency.

TEMPERATURE_LIMIT = 70  # Celsius, start measuring only below this
COOLDOWN_TIMEOUT = 300  # Seconds

def query_gpu(device: int) -> tuple:
    """Return `(sm_clock_MHz, max_sm_clock_MHz, temperature_C)` of one device."""
    command = ["nvidia-smi", "-i", str(device), "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu", "--format=csv,noheader,nounits"]
    process = subprocess.run(command, capture_output=True, text=True)
    return tuple(int(value) for value in process.stdout.strip().split(","))

def pick_device_list(n_device: int) -> list:
    """The `n_device` coolest GPUs, so a previous run's heat does not distort this one."""
    _, device_count = cudart.cudaGetDeviceCount()
    temperature_list = sorted((query_gpu(i)[2], i) for i in range(device_count))
    return sorted(index for _, index in temperature_list[:n_device])

def wait_until_cool(device_list: list) -> None:
    """Block until every device is below `TEMPERATURE_LIMIT`, or give up and say so."""
    t0 = time.time()
    while time.time() - t0 < COOLDOWN_TIMEOUT:
        temperature_list = [query_gpu(device)[2] for device in device_list]
        if max(temperature_list) < TEMPERATURE_LIMIT:
            return
        time.sleep(5)
    print(f"    WARNING: GPU {device_list} still at {[query_gpu(d)[2] for d in device_list]} C after {COOLDOWN_TIMEOUT} s, "
          f"the latency below is thermally throttled and must not be compared against anything")
    return

# The parent chooses the devices once and passes them down, so that every rank and
# the single-device reference all agree on which GPUs this run owns.
DEVICE_LIST = [int(value) for value in os.environ["TRT_DEVICE_LIST"].split(",")] if os.getenv("TRT_DEVICE_LIST") else []

# ================================================================ TensorRT helpers

def build_engine_bytes(onnx_file: Path) -> bytes:
    """Parse an ONNX file into a strongly-typed network and build the engine."""
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_file)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"Failed parsing {onnx_file}")

    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    engine_bytes = builder.build_serialized_network(network, builder_config)
    if engine_bytes is None:
        raise RuntimeError(f"Failed building engine from {onnx_file}")
    return bytes(engine_bytes), network.num_layers

def run_engine(engine_bytes: bytes, input_data: np.ndarray, communicator_capsule=None) -> tuple:
    """Deserialize, optionally attach a NCCL communicator, then time `N_INFERENCE` runs.

    Returns `(output, median_latency_ms, sm_clock_MHz, temperature_C)`; the last two
    are what tells you whether the latency is worth anything.
    """
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    if communicator_capsule is not None:
        # The only extra runtime API needed for multi-device: hand TensorRT the
        # already-initialized NCCL communicator, it drives the collectives itself.
        assert context.set_communicator(communicator_capsule), "Failed setting the NCCL communicator"

    name_list = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    buffer = OrderedDict()
    for name in name_list:
        shape = context.get_tensor_shape(name)
        data_type = engine.get_tensor_dtype(name)
        n_byte = trt.volume(shape) * data_type.itemsize
        host_buffer = np.empty(shape, dtype=trt.nptype(data_type))
        device_buffer = cudart.cudaMalloc(max(n_byte, 1))[1]
        buffer[name] = [host_buffer, device_buffer, n_byte]
        context.set_tensor_address(name, device_buffer)

    input_name = name_list[0]
    output_name = name_list[-1]
    cudart.cudaMemcpy(buffer[input_name][1], np.ascontiguousarray(input_data).ctypes.data, buffer[input_name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    for _ in range(N_WARMUP):
        context.execute_async_v3(0)
    cudart.cudaStreamSynchronize(0)

    # Time each iteration separately and take the **median**. A mean over one block is
    # too noisy here: the single-device engine's latency moved between 4.8 and 12.1 ms
    # across rebuilds at sequence 32768, because the builder does not always pick the
    # same attention tactic.
    latency_list = []
    for _ in range(N_INFERENCE):
        t0 = time.time()
        context.execute_async_v3(0)
        cudart.cudaStreamSynchronize(0)
        latency_list.append((time.time() - t0) * 1000)
    latency_ms = float(np.median(latency_list))
    sm_clock, _, temperature = query_gpu(cudart.cudaGetDevice()[1])

    cudart.cudaMemcpy(buffer[output_name][0].ctypes.data, buffer[output_name][1], buffer[output_name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    output = buffer[output_name][0].copy()

    for _, device_buffer, _ in buffer.values():
        cudart.cudaFree(device_buffer)
    return output, latency_ms, sm_clock, temperature

# ================================================================ NCCL helpers

def nccl_unique_id_via_file(rank: int):
    """Rank 0 creates the NCCL unique id and writes it out, the other ranks poll for it.

    Same trick as `02-API/Layer/DistCollective`: it keeps the example runnable as a
    plain `python3 main.py`, with no `mpirun` in the loop.
    """
    file_path = Path(os.environ["TRT_NCCL_ID_FILE"])
    if rank == 0:
        unique_id = nccl.get_unique_id()
        raw = unique_id.as_bytes() if callable(getattr(unique_id, "as_bytes", None)) else bytes(unique_id)
        file_path.write_text(raw.hex())
        return unique_id

    for _ in range(3000):
        if file_path.exists():
            text = file_path.read_text().strip()
            if len(text) == 256:
                return nccl.UniqueId.from_bytes(bytes.fromhex(text))
        time.sleep(0.01)
    raise TimeoutError("Timeout waiting for the NCCL unique id file from rank 0")

def communicator_to_capsule(communicator):
    """`set_communicator` wants a `PyCapsule` named `ncclComm_t`, not an integer."""
    assert int(communicator.ptr) != 0, "NCCL communicator has already been destroyed"
    py_capsule_new = pythonapi.PyCapsule_New
    py_capsule_new.restype = py_object
    py_capsule_new.argtypes = [c_void_p, c_char_p, c_void_p]
    return py_capsule_new(c_void_p(int(communicator.ptr)), b"ncclComm_t", None)

# ================================================================ Cases

@case_mark
def case_build_model() -> None:
    """Build the single-device attention model."""
    build_attention_onnx()
    model = onnx.load(onnx_file_sd)
    print(f"    {onnx_file_sd.name}: node={len(model.graph.node)}, initializer={len(model.graph.initializer)}")
    print(f"    input={[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in model.graph.input]}")
    print(f"    output={[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in model.graph.output]}")
    return

@case_mark
def case_shard_model() -> None:
    """Rewrite the model for context parallelism with `polygraphy multi-device shard`."""
    # The `polygraphy_class` keys are **mandatory**: polygraphy deserializes this file
    # through its own JSON machinery, which dispatches on that key. Dropping them (as
    # the upstream sample README does when it explains the format) fails with
    # "Provided JSON cannot be decoded into a ShardHints".
    hint = {
        "parallelism": "CP",  # Context parallelism, i.e. split along the sequence axis
        "attention_layers": [{
            "q": "q_scaled",  # The Q tensor feeding QK^T, this is how the tool finds the attention
            "gather_kv": True,  # Every rank needs the full K/V to attend over the whole sequence
            "gather_q": False,  # Q stays local, each rank only produces its own slice of the output
            "replace": None,
            "polygraphy_class": "AttentionLayerHint",
        }],
        "dist_collectives": {
            "group_size": 0,
            "root": -1,
            "nb_rank": REQUIRED_WORLD_SIZE,
            "reduce_op": "max",  # Identity here, see the module docstring
            "groups": [],
            "polygraphy_class": "DistCollective",
        },
        "inputs": [{
            "name": "input",
            "seq_len_idx": 0,
            "rank": 3,
            "polygraphy_class": "ShardTensor"
        }],
        "outputs": [{
            "name": "output",
            "seq_len_idx": 0,
            "rank": 3,
            "polygraphy_class": "ShardTensor"
        }],
        "k_seq_len_idx": 3,  # K has been transposed to [batch, head, head_dim, sequence]
        "v_seq_len_idx": 2,  # V is still [batch, head, sequence, head_dim]
        "kv_rank": 4,
        "polygraphy_class": "ShardHints",
    }
    hint_file.write_text(json.dumps(hint, indent=4))

    command = ["polygraphy", "multi-device", "shard", str(onnx_file_sd), "-s", str(hint_file), "-o", str(onnx_file_md)]
    process = subprocess.run(command, capture_output=True, text=True)
    assert process.returncode == 0, f"polygraphy multi-device shard failed:\n{process.stderr}"

    model = onnx.load(onnx_file_md)
    print(f"    {onnx_file_md.name}: node={len(model.graph.node)} (+{len(model.graph.node) - len(onnx.load(onnx_file_sd).graph.node)})")
    print(f"    {'Name':<12}{'Operation':<16}{'ReduceOp':<10}{'Rank':>6}")
    for node in model.graph.node:
        if node.op_type == "DistCollective":
            attribute = {a.name: (a.s.decode() if a.type == onnx.AttributeProto.STRING else a.i) for a in node.attribute}
            print(f"    {node.name:<12}{attribute['collective_operation']:<16}{attribute['reduce_op']:<10}{attribute['nb_rank']:>6}")
    # The sharding tool renames the graph output, the I/O contract is otherwise unchanged
    print(f"    output name: {[o.name for o in onnx.load(onnx_file_sd).graph.output]} -> {[o.name for o in model.graph.output]}")
    return

@case_mark
def case_single_device() -> None:
    """Numerical and latency reference: the unsharded model on one GPU.

    Runs on `DEVICE_LIST[0]`, the same device rank 0 of the parallel run will use,
    so the two latencies come from the same silicon.
    """
    wait_until_cool(DEVICE_LIST[:1])
    cudart.cudaSetDevice(DEVICE_LIST[0])
    input_data = (np.random.rand(N_SEQUENCE, N_BATCH, HIDDEN_DIM).astype(np.float32) * 2 - 1).astype(np.float16)
    np.save(input_file, input_data)

    engine_bytes, n_layer = build_engine_bytes(onnx_file_sd)
    output, latency_ms, sm_clock, temperature = run_engine(engine_bytes, input_data)
    np.save(reference_file, output)

    print(f"    GPU {DEVICE_LIST[0]}: TRT-layer={n_layer}, engine={len(engine_bytes) / (1 << 20):.2f} MiB, latency={latency_ms:.3f} ms, SM-clock={sm_clock} MHz, {temperature} C")
    # `n_sequence` is stamped in so that `sweep.py` can refuse a stale file
    result_file.write_text(json.dumps({"n_sequence": N_SEQUENCE, "device_list": DEVICE_LIST, "single_device": {"n_layer": n_layer, "n_byte": len(engine_bytes), "latency_ms": latency_ms, "sm_clock": sm_clock, "temperature": temperature}}))
    return

@case_mark
def case_context_parallel() -> None:
    """Run the sharded model, one process per rank."""
    wait_until_cool(DEVICE_LIST)
    with tempfile.TemporaryDirectory(prefix="trt_context_parallel_") as temporary_directory:
        process_list = []
        for rank in range(REQUIRED_WORLD_SIZE):
            environment = os.environ.copy()
            environment["TRT_MY_RANK"] = str(rank)
            environment["TRT_WORLD_SIZE"] = str(REQUIRED_WORLD_SIZE)
            environment["TRT_NCCL_ID_FILE"] = str(Path(temporary_directory) / "nccl_id.txt")
            environment["TRT_DEVICE_LIST"] = ",".join(str(d) for d in DEVICE_LIST)
            process_list.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve())], env=environment, cwd=str(output_path), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
        # Collect the children's output instead of letting it race with the parent's
        output_list = [process.communicate()[0] for process in process_list]
        exit_code_list = [process.returncode for process in process_list]
    for text in output_list:
        print("\n".join(line for line in text.splitlines() if line.startswith("    [Rank")))
    assert all(code == 0 for code in exit_code_list), f"Child rank process failed, exit codes={exit_code_list}\n{''.join(output_list)}"
    return

def run_one_rank() -> None:
    """Body of a child process, one per rank."""
    rank = int(os.environ["TRT_MY_RANK"])
    world_size = int(os.environ["TRT_WORLD_SIZE"])
    device = DEVICE_LIST[rank]  # Rank i runs on the i-th *chosen* device, not necessarily device i
    cudart.cudaSetDevice(device)

    communicator = nccl.Communicator.init(nranks=world_size, rank=rank, unique_id=nccl_unique_id_via_file(rank))

    # Every rank is given the *whole* input, the leading ReduceScatter splits it
    input_data = np.load(input_file)
    engine_bytes, n_layer = build_engine_bytes(onnx_file_md)
    output, latency_ms, sm_clock, temperature = run_engine(engine_bytes, input_data, communicator_to_capsule(communicator))

    reference = np.load(reference_file).astype(np.float32)
    difference = np.abs(output.astype(np.float32) - reference)
    relative = np.max(difference) / max(np.max(np.abs(reference)), 1e-6)
    print(f"    [Rank {rank}] GPU {device}: TRT-layer={n_layer}, engine={len(engine_bytes) / (1 << 20):.2f} MiB, latency={latency_ms:.3f} ms, "
          f"SM-clock={sm_clock} MHz, {temperature} C, max|diff|={np.max(difference):.3e}, relative={relative:.3e}")
    assert relative < 1e-2, f"[Rank {rank}] Output does not match the single-device reference"

    if rank == 0:
        result = json.loads(result_file.read_text())
        result["context_parallel"] = {"n_layer": n_layer, "n_byte": len(engine_bytes), "latency_ms": latency_ms, "sm_clock": sm_clock, "temperature": temperature, "max_abs_diff": float(np.max(difference)), "relative": float(relative)}
        result_file.write_text(json.dumps(result))
    return

# ================================================================ Entrance

def main() -> None:
    case_build_model()
    case_shard_model()
    case_single_device()
    case_context_parallel()

    result = json.loads(result_file.read_text())
    print("\n" + "=" * 78)
    print(f"{'Case':<20}{'GPU':>5}{'TRTLayer':>10}{'Engine(MiB)':>14}{'Latency(ms)':>14}{'RelDiff':>14}")
    print("-" * 78)
    single = result["single_device"]
    parallel = result["context_parallel"]
    print(f"{'single_device':<20}{1:>5}{single['n_layer']:>10}{single['n_byte'] / (1 << 20):>14.2f}{single['latency_ms']:>14.3f}{0.0:>14.3e}")
    print(f"{'context_parallel':<20}{REQUIRED_WORLD_SIZE:>5}{parallel['n_layer']:>10}{parallel['n_byte'] / (1 << 20):>14.2f}{parallel['latency_ms']:>14.3f}{parallel['relative']:>14.3e}")
    print("=" * 78)
    print(f"Speedup: {single['latency_ms'] / parallel['latency_ms']:.2f}x on {REQUIRED_WORLD_SIZE} GPUs")
    print("Note: the two engines have the same I/O contract, the sharded one additionally")
    print("      carries 4 DistCollective layers and holds half of the sequence per rank.")
    return

if __name__ == "__main__":

    # Child process, one rank of the context-parallel run
    if os.getenv("TRT_MY_RANK") is not None:
        run_one_rank()
        exit(0)

    # Parent process
    _, device_count = cudart.cudaGetDeviceCount()
    if device_count < REQUIRED_WORLD_SIZE:
        print(f"Skip since no enough GPU is ready (need {REQUIRED_WORLD_SIZE}, get {device_count})")
        exit(0)

    DEVICE_LIST = pick_device_list(REQUIRED_WORLD_SIZE)
    print(f"Using GPU {DEVICE_LIST}, currently at {[query_gpu(device)[2] for device in DEVICE_LIST]} C "
          f"(coolest {REQUIRED_WORLD_SIZE} of {device_count}; see the thermal-guard note in this file)")

    main()
    print("\nFinish")
