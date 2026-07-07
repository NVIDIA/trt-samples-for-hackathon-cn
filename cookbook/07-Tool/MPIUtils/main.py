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

import numpy as np

from tensorrt_cookbook import (
    local_mpi_rank,
    local_mpi_size,
    mpi_allgather,
    mpi_barrier,
    mpi_broadcast,
    mpi_rank,
    mpi_recv,
    mpi_recv_object,
    mpi_send,
    mpi_send_object,
    mpi_world_size,
)

def main():
    rank = mpi_rank()
    world = mpi_world_size()
    local_rank = local_mpi_rank()
    local_size = local_mpi_size()

    print(f"[rank {rank}] world={world}, local_rank={local_rank}, local_size={local_size}")

    # 1) Broadcast from root
    config = {"batch": 8, "dtype": "fp16"} if rank == 0 else None
    config = mpi_broadcast(config, root=0)
    print(f"[rank {rank}] broadcast config = {config}")

    # 2) All-gather simple object
    gathered = mpi_allgather({"rank": rank, "value": rank * rank})
    print(f"[rank {rank}] allgather = {gathered}")

    # 3) Point-to-point array/object transfer (rank0 -> rank1)
    if world > 1:
        tag_array = 11
        tag_obj = 12
        if rank == 0:
            array = np.arange(8, dtype=np.float32) + 100
            payload = {"msg": "hello from rank0", "shape": array.shape}
            mpi_send(array, dest=1, tag=tag_array)
            mpi_send_object(payload, dest=1, tag=tag_obj)
            print(f"[rank {rank}] send array+object to rank 1")
        elif rank == 1:
            recv = np.empty(8, dtype=np.float32)
            mpi_recv(recv, source=0, tag=tag_array)
            payload = mpi_recv_object(source=0, tag=tag_obj)
            print(f"[rank {rank}] recv array={recv}, object={payload}")

        # Synchronize all ranks before exit
        mpi_barrier()

    print(f"[rank {rank}] finish")

if __name__ == "__main__":
    main()
