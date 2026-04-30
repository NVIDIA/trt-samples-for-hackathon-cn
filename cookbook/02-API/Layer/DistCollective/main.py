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
from ctypes import c_char_p, c_void_p, py_object, pythonapi

import numpy as np
import tensorrt as trt
import torch
import torch.distributed as dist
from tensorrt_cookbook import TRTWrapperV1, case_mark, datatype_cast
import nccl.core as nccl  # need package `nccl4py`

@case_mark
def case_simple():

    if torch.cuda.device_count() < 2:
        print(f"Skip this test since DistCollectiveLayer requires at least 2 GPUs, but only {torch.cuda.device_count()} is available.")
        return

    required_env = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
    if len([key for key in required_env if key not in os.environ]) > 0:
        assert False, f"DistCollective should be launched with torchrun"

    # Initialize torch distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    print(f"{rank = }, {world_size = }, {local_rank = }")

    torch.cuda.set_device(local_rank)

    # Initialize NCCL communicator
    unique_id = nccl.get_unique_id() if rank == 0 else None
    payload = [unique_id]
    dist.broadcast_object_list(payload, src=0)
    nccl_comm = nccl.Communicator.init(nranks=world_size, rank=rank, unique_id=payload[0])
    if nccl_comm is None or not hasattr(nccl_comm, "ptr"):
        raise TypeError("Expect a valid nccl.core.Communicator")
    if nccl_comm.ptr == 0:
        raise ValueError("NCCL communicator has been destroyed")

    print(f"{unique_id = }")

    # Get corresponding operator name
    op_map = {
        "all_reduce": (trt.CollectiveOperation.ALL_REDUCE, trt.ReduceOperation.SUM, -1),
        "all_gather": (trt.CollectiveOperation.ALL_GATHER, trt.ReduceOperation.NONE, -1),
        "broadcast": (trt.CollectiveOperation.BROADCAST, trt.ReduceOperation.NONE, 0),
        "reduce": (trt.CollectiveOperation.REDUCE, trt.ReduceOperation.SUM, 0),
        "reduce_scatter": (trt.CollectiveOperation.REDUCE_SCATTER, trt.ReduceOperation.SUM, -1),
    }
    op_name = "all_reduce"
    collective_op, reduce_op, root = op_map[op_name]
    groups = list(range(world_size))

    data = {"input": np.full((4, ), rank + 1, dtype=np.float32)}

    tw = TRTWrapperV1()
    tw.network = tw.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    tensor = tw.network.add_input("input", datatype_cast(data["input"].dtype, "trt"), data["input"].shape)
    layer = tw.network.add_dist_collective(tensor, collective_op, reduce_op, root, groups)
    layer.metadata = f"dist-collective:{op_name}"
    layer.num_ranks = world_size

    print("here")

    if not tw.build([layer.get_output(0)]):
        print(f"[Rank {rank}] build failed")
        return

    tw.setup(data)

    py_capsule_new = pythonapi.PyCapsule_New
    py_capsule_new.restype = py_object
    py_capsule_new.argtypes = [c_void_p, c_char_p, c_void_p]
    capsule = py_capsule_new(c_void_p(nccl_comm.ptr), b"ncclComm_t", None)

    if not tw.context.set_communicator(capsule):
        print(f"[Rank {rank}] set_communicator failed")
        return

    tw.infer(b_print_io=False)

    print(f"[Rank {rank}] input={data['input']}, output={tw.buffer['output'][0]}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":

    case_simple()  # TODO: correct this example

    print("Finish")
