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

import os
import subprocess
import sys
import tempfile
import time
from ctypes import c_char_p, c_void_p, py_object, pythonapi
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, case_mark, check_api_coverage, check_array, print_enumerated_members
import cuda.bindings.runtime as cudart
import nccl.core as nccl

REQUIRED_WORLD_SIZE = 2

COLLECTIVE_ARGS_MAP = {
    "all_reduce": (trt.CollectiveOperation.ALL_REDUCE, trt.ReduceOperation.SUM, -1),
    "all_gather": (trt.CollectiveOperation.ALL_GATHER, trt.ReduceOperation.NONE, -1),
    "broadcast": (trt.CollectiveOperation.BROADCAST, trt.ReduceOperation.NONE, 0),
    "reduce": (trt.CollectiveOperation.REDUCE, trt.ReduceOperation.SUM, 0),
    "reduce_scatter": (trt.CollectiveOperation.REDUCE_SCATTER, trt.ReduceOperation.SUM, -1),
}

@dataclass(frozen=True)
class CollectiveTestConfig:
    rank0_input: list[float]
    rank1_input: list[float]
    rank0_expected_output: list[float]
    rank1_expected_output: list[float]

def _get_test_config(op_name: str) -> CollectiveTestConfig:
    if op_name == "all_gather":
        expected = [1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0, 10.0, 50.0, 90.0, 20.0, 60.0, 100.0, 30.0, 70.0, 110.0, 40.0, 80.0, 120.0]
        return CollectiveTestConfig(
            rank0_input=[1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0],
            rank1_input=[10.0, 50.0, 90.0, 20.0, 60.0, 100.0, 30.0, 70.0, 110.0, 40.0, 80.0, 120.0],
            rank0_expected_output=expected,
            rank1_expected_output=expected,
        )

    if op_name == "all_reduce":
        expected = [11.0, 55.0, 99.0, 22.0, 66.0, 110.0, 33.0, 77.0, 121.0, 44.0, 88.0, 132.0]
        return CollectiveTestConfig(
            rank0_input=[1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0],
            rank1_input=[10.0, 50.0, 90.0, 20.0, 60.0, 100.0, 30.0, 70.0, 110.0, 40.0, 80.0, 120.0],
            rank0_expected_output=expected,
            rank1_expected_output=expected,
        )

    if op_name == "broadcast":
        expected = [1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0]
        return CollectiveTestConfig(
            rank0_input=expected,
            rank1_input=[99.0] * 12,
            rank0_expected_output=expected,
            rank1_expected_output=expected,
        )

    if op_name == "reduce":
        return CollectiveTestConfig(
            rank0_input=[1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0],
            rank1_input=[10.0, 50.0, 90.0, 20.0, 60.0, 100.0, 30.0, 70.0, 110.0, 40.0, 80.0, 120.0],
            rank0_expected_output=[11.0, 55.0, 99.0, 22.0, 66.0, 110.0, 33.0, 77.0, 121.0, 44.0, 88.0, 132.0],
            rank1_expected_output=[],
        )

    if op_name == "reduce_scatter":
        return CollectiveTestConfig(
            rank0_input=[1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0],
            rank1_input=[10.0, 50.0, 90.0, 20.0, 60.0, 100.0, 30.0, 70.0, 110.0, 40.0, 80.0, 120.0],
            rank0_expected_output=[11.0, 55.0, 99.0, 22.0, 66.0, 110.0],
            rank1_expected_output=[33.0, 77.0, 121.0, 44.0, 88.0, 132.0],
        )

    assert False, f"Unknown operation: {op_name}"

# NCCL related
def _hex_char_to_int(c: str) -> int:
    if "0" <= c <= "9":
        return ord(c) - ord("0")
    if "a" <= c <= "f":
        return ord(c) - ord("a") + 10
    if "A" <= c <= "F":
        return ord(c) - ord("A") + 10
    return -1

def _nccl_id_to_hex(unique_id) -> str:
    if hasattr(unique_id, "as_bytes"):
        as_bytes = unique_id.as_bytes
        raw = as_bytes() if callable(as_bytes) else as_bytes
    else:
        raw = bytes(unique_id)
    return "".join([f"{byte:02x}" for byte in raw])

def _hex_to_nccl_id(hex_str: str):
    raw = bytearray(128)
    for i in range(128):
        high = _hex_char_to_int(hex_str[2 * i])
        low = _hex_char_to_int(hex_str[2 * i + 1])
        raw[i] = (high << 4) | low
    return nccl.UniqueId.from_bytes(bytes(raw))

def _get_nccl_id_via_file(rank: int):
    file_path_env = os.getenv("TRT_NCCL_ID_FILE")
    file_path = Path(file_path_env)
    expected_hex_len = 2 * 128
    poll_interval_s = 0.01
    timeout_s = 30.0

    if rank == 0:
        unique_id = nccl.get_unique_id()
        hex_str = _nccl_id_to_hex(unique_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(hex_str)
        return unique_id

    elapsed = 0.0
    while elapsed < timeout_s:
        if file_path.exists():
            hex_str = file_path.read_text().strip()
            if len(hex_str) == expected_hex_len:
                return _hex_to_nccl_id(hex_str)
        time.sleep(poll_interval_s)
        elapsed += poll_interval_s
    assert False, "Timeout waiting for NCCL ID file from rank 0"

def run_collective_test():
    world_size = int(os.getenv("TRT_WORLD_SIZE", "1"))
    rank = int(os.getenv("TRT_MY_RANK", "0"))
    op_name = os.getenv("TRT_COLLECTIVE_OP", "none")

    cudart.cudaSetDevice(rank)

    nccl_comm = nccl.Communicator.init(nranks=world_size, rank=rank, unique_id=_get_nccl_id_via_file(rank))
    assert nccl_comm is not None and hasattr(nccl_comm, "ptr") and int(nccl_comm.ptr) != 0, "Failed to initialize NCCL communicator"

    config = _get_test_config(op_name)
    input_chunk = np.asarray(config.rank0_input if rank == 0 else config.rank1_input, dtype=np.float32).reshape(4, 3)

    collective_op, reduce_op, root = COLLECTIVE_ARGS_MAP[op_name]

    tw = TRTWrapperV1()
    tw.builder_config.set_preview_feature(trt.PreviewFeature.MULTIDEVICE_RUNTIME_10_16, True)
    tw.network = tw.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    input_tensor = tw.network.add_input("input", trt.float32, (4, 3))
    layer = tw.network.add_dist_collective(input_tensor, collective_op, reduce_op, root, list(range(world_size)))
    # Input: Tensor of type T with shape [d0, d1, ..., dn] where n >= 1. For BROADCAST and SCATTER, only the root rank provides meaningful data.
    # Outputs: Tensor of type T. ALL_GATHER/GATHER produce [nb_rank * d0, d1, ..., dn]; REDUCE_SCATTER/SCATTER produce [d0/nb_rank, d1, ..., dn]; REDUCE produces output only on root rank.
    # Data type: float32, float16, bfloat16, float8, int64, int32, int8, uint8, bool.
    # Shape: Input [d0, d1, ..., dn], n >= 1. Output shape depends on operation (see above). First dimension must be divisible by nb_rank where applicable.
    # Volume limits: None specified.
    layer.num_ranks = world_size  # [Optional] Default: 1, number of participating ranks (> 1 enables multi-device execution)

    output_tensor = layer.get_output(0)
    output_tensor.name = "output"

    tw.build([output_tensor])
    tw.setup({"input": input_chunk})

    py_capsule_new = pythonapi.PyCapsule_New
    py_capsule_new.restype = py_object
    py_capsule_new.argtypes = [c_void_p, c_char_p, c_void_p]
    communicator_capsule = py_capsule_new(c_void_p(int(nccl_comm.ptr)), b"ncclComm_t", None)
    tw.context.set_communicator(communicator_capsule)

    tw.infer(b_print_io=False)

    output = tw.buffer["output"][0]
    expected = config.rank0_expected_output if rank == 0 else config.rank1_expected_output
    if len(expected) != 0:
        output_flat = output.astype(np.float32).reshape(-1)
        expected_flat = np.asarray(expected, dtype=np.float32)
        check_array(output_flat, expected_flat)

    print(f"Rank {rank} PASSED")

    check_api_coverage(layer)  # Sanity check, unnecessary in normal workflow

@case_mark
def case_normal(op_name: str):

    script_path = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="trt_dist_collective_") as tmp_dir:
        nccl_id_path = Path(tmp_dir) / "nccl_id.txt"
        procs = []
        for rank in range(REQUIRED_WORLD_SIZE):
            env = os.environ.copy()
            env["TRT_COLLECTIVE_OP"] = op_name
            env["TRT_LAUNCHED_CHILD"] = "1"
            env["TRT_MY_RANK"] = str(rank)
            env["TRT_NCCL_ID_FILE"] = str(nccl_id_path)
            env["TRT_WORLD_SIZE"] = str(REQUIRED_WORLD_SIZE)
            procs.append(subprocess.Popen([sys.executable, str(script_path)], env=env, cwd=str(script_path.parent)))

        exit_codes = [proc.wait() for proc in procs]
        assert not any(code != 0 for code in exit_codes), f"Child rank process failed, exit codes={exit_codes}"

if __name__ == "__main__":

    print_enumerated_members(trt.CollectiveOperation)
    print_enumerated_members(trt.ReduceOperation)

    # Child process
    if all(os.getenv(key) is not None for key in ["TRT_MY_RANK", "TRT_WORLD_SIZE", "TRT_NCCL_ID_FILE"]):
        run_collective_test()
        print("Finish child process")
        exit(0)

    # Parent process
    _, device_count = cudart.cudaGetDeviceCount()
    if device_count < REQUIRED_WORLD_SIZE:
        print(f"Skip since no enough GPU is ready (need {REQUIRED_WORLD_SIZE}, get {device_count})")
        exit(0)

    for op_name in ["all_reduce", "all_gather", "broadcast", "reduce", "reduce_scatter"]:
        case_normal(op_name)

    print("Finish parent process")
