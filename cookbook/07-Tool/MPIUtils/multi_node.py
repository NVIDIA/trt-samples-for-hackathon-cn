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

import torch

from tensorrt_cookbook import (
    local_mpi_comm,
    local_mpi_rank,
    local_mpi_size,
    mpi_allgather,
    mpi_barrier,
    mpi_broadcast,
    mpi_isend_object,
    mpi_rank,
    mpi_recv_object,
    mpi_world_size,
)

def bind_device_by_local_rank() -> int:
    if not torch.cuda.is_available():
        return -1
    n_device = torch.cuda.device_count()
    if n_device <= 0:
        return -1
    device_id = local_mpi_rank() % n_device
    torch.cuda.set_device(device_id)
    return device_id

def get_host_leaders(rank_host_local):
    host_to_leader = {}
    for item in rank_host_local:
        rank = int(item["rank"])
        host = str(item["host"])
        if host not in host_to_leader:
            host_to_leader[host] = rank
        else:
            host_to_leader[host] = min(host_to_leader[host], rank)
    return sorted(host_to_leader.values())

def main():
    rank = mpi_rank()
    world = mpi_world_size()
    host = socket.gethostname()

    local_rank = local_mpi_rank()
    local_size = local_mpi_size()
    local_comm_rank = local_mpi_comm().Get_rank()
    local_comm_size = local_mpi_comm().Get_size()

    device_id = bind_device_by_local_rank()
    print(f"[rank {rank}] host={host}, world={world}, local_rank={local_rank}, "
          f"local_size={local_size}, local_comm_rank={local_comm_rank}/{local_comm_size}, cuda_device={device_id}")

    # Gather global topology snapshot
    topology = mpi_allgather({
        "rank": rank,
        "host": host,
        "local_rank": local_rank,
        "device": device_id,
    })

    # Root builds host-leader plan and broadcasts to all ranks
    plan = None
    if rank == 0:
        leaders = get_host_leaders(topology)
        plan = {
            "n_hosts": len({item["host"]
                            for item in topology}),
            "leaders": leaders,
            "topology": topology,
        }
    plan = mpi_broadcast(plan, root=0)

    if rank == 0:
        print(f"[rank 0] host leaders = {plan['leaders']}")

    # Example: host-leader ring token pass (across hosts)
    leaders = plan["leaders"]
    if len(leaders) > 1 and rank in leaders:
        i = leaders.index(rank)
        dst = leaders[(i + 1) % len(leaders)]
        src = leaders[(i - 1 + len(leaders)) % len(leaders)]

        req = mpi_isend_object({"from": rank, "host": host}, dest=dst, tag=701)
        token = mpi_recv_object(source=src, tag=701)
        if req is not None:
            req.wait()
        print(f"[leader {rank}] recv token from leader {src}: {token}")

    mpi_barrier()
    print(f"[rank {rank}] finish")

if __name__ == "__main__":
    main()
