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
from ctypes import c_char_p, c_void_p, py_object, pythonapi

import numpy as np
import tensorrt as trt
import torch
import torch.distributed as dist
from tensorrt_cookbook import TRTWrapperV1, case_mark, datatype_cast

try:
    import nccl.core as nccl  # need package `nccl4py`
    HAS_NCCL_CORE = True
except Exception:
    HAS_NCCL_CORE = False

def init_torch_dist():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def init_nccl_comm(rank, world_size):
    unique_id = nccl.get_unique_id() if rank == 0 else None
    payload = [unique_id]
    dist.broadcast_object_list(payload, src=0)
    return nccl.Communicator.init(nranks=world_size, rank=rank, unique_id=payload[0])

def parse_op(op_name: str):
    op_name = op_name.lower()
    if op_name == "all_reduce":
        return trt.CollectiveOperation.ALL_REDUCE, trt.ReduceOperation.SUM, -1
    if op_name == "all_gather":
        return trt.CollectiveOperation.ALL_GATHER, trt.ReduceOperation.NONE, -1
    if op_name == "broadcast":
        return trt.CollectiveOperation.BROADCAST, trt.ReduceOperation.NONE, 0
    if op_name == "reduce":
        return trt.CollectiveOperation.REDUCE, trt.ReduceOperation.SUM, 0
    if op_name == "reduce_scatter":
        return trt.CollectiveOperation.REDUCE_SCATTER, trt.ReduceOperation.SUM, -1
    raise ValueError(f"Unsupported op: {op_name}")

@case_mark
def case_simple():

    op_name: str = "all_reduce"  # all_reduce, all_gather, broadcast, reduce, reduce_scatter

    rank, world_size, _ = init_torch_dist()
    nccl_comm = init_nccl_comm(rank, world_size)

    collective_op, reduce_op, root = parse_op(op_name)
    groups = list(range(world_size))

    data = {"input": np.full((4, ), rank + 1, dtype=np.float32)}

    tw = TRTWrapperV1()
    tw.network = tw.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    tensor = tw.network.add_input("input", datatype_cast(data["input"].dtype, "trt"), data["input"].shape)
    layer = tw.network.add_dist_collective(tensor, collective_op, reduce_op, root, groups)
    if layer is None:
        print(f"[Rank {rank}] add_dist_collective failed")
        return

    layer.metadata = f"dist-collective:{op_name}"
    layer.num_ranks = world_size

    output_tensor = layer.get_output(0)
    output_tensor.name = "output"
    if not tw.build([output_tensor]):
        print(f"[Rank {rank}] build failed")
        return

    tw.setup(data)

    if nccl_comm is None or not hasattr(nccl_comm, "ptr"):
        raise TypeError("Expect a valid nccl.core.Communicator")
    if nccl_comm.ptr == 0:
        raise ValueError("NCCL communicator has been destroyed")
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

    case_simple()  # TODO: check this

    print("Finish")
