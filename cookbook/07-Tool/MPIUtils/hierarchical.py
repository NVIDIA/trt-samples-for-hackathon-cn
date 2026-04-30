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

import socket

from tensorrt_cookbook import (
    local_mpi_comm,
    local_mpi_rank,
    mpi_allgather,
    mpi_barrier,
    mpi_broadcast,
    mpi_rank,
    mpi_world_size,
)

def main():
    rank = mpi_rank()
    world = mpi_world_size()
    host = socket.gethostname()

    # Step 1) Node-local aggregation using local communicator
    local_value = rank + 1
    local_values = local_mpi_comm().allgather(local_value)
    local_sum = sum(local_values)

    is_local_leader = local_mpi_rank() == 0

    # Step 2) Global gather of node-leader partial sums
    leader_payload = {
        "rank": rank,
        "host": host,
        "local_sum": local_sum,
    } if is_local_leader else None

    leader_payloads = mpi_allgather(leader_payload)
    leader_payloads = [item for item in leader_payloads if item is not None]

    # Step 3) Root builds final plan and broadcasts
    plan = None
    if rank == 0:
        global_sum = sum(item["local_sum"] for item in leader_payloads)
        plan = {
            "n_ranks": world,
            "n_hosts": len({item["host"]
                            for item in leader_payloads}),
            "leader_payloads": leader_payloads,
            "global_sum": global_sum,
        }
    plan = mpi_broadcast(plan, root=0)

    print(f"[rank {rank}] host={host}, local_value={local_value}, "
          f"local_sum={local_sum}, is_local_leader={is_local_leader}, global_sum={plan['global_sum']}")

    mpi_barrier()
    print(f"[rank {rank}] finish")

if __name__ == "__main__":
    main()
