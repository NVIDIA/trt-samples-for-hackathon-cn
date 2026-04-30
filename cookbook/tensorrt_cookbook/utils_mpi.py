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
import socket
import threading
import linecache
from functools import wraps
import torch
import trace
import sys
import traceback
try:
    from mpi4py import MPI
    from mpi4py.util import pkl5

    _MPI_AVAILABLE = True
except Exception:
    MPI = None
    pkl5 = None
    _MPI_AVAILABLE = False

OMPI_COMM_TYPE_HOST = 9  # mpi4py doesn't export this OpenMPI constant.

class _SingleProcessRequest:

    def wait(self):
        return None

class _SingleProcessComm:

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Barrier(self):
        return None

    def bcast(self, obj, root=0):
        return obj

    def allgather(self, obj):
        return [obj]

    def Isend(self, buf, dest=0, tag=0):
        return _SingleProcessRequest()

    def Send(self, buf, dest=0, tag=0):
        return None

    def Recv(self, buf, source=0, tag=0):
        return None

    def send(self, obj, dest=0, tag=0):
        return None

    def isend(self, obj, dest=0, tag=0):
        return _SingleProcessRequest()

    def recv(self, source=0, tag=0):
        return None

    def Split_type(self, split_type=OMPI_COMM_TYPE_HOST):
        return self

_fallback_comm = _SingleProcessComm()
comm = pkl5.Intracomm(MPI.COMM_WORLD) if _MPI_AVAILABLE else _fallback_comm
thread_local_comm = threading.local()
local_comm = comm.Split_type(split_type=OMPI_COMM_TYPE_HOST)

def set_mpi_comm(new_comm):
    global comm
    comm = new_comm

def set_thread_local_mpi_comm(new_comm):
    thread_local_comm.value = new_comm

def mpi_comm():
    if hasattr(thread_local_comm, "value") and thread_local_comm.value is not None:
        return thread_local_comm.value
    return comm

def local_mpi_comm():
    return local_comm

def mpi_disabled() -> bool:
    return os.environ.get("TRT_COOKBOOK_DISABLE_MPI", "0") == "1"

def global_mpi_rank() -> int:
    if mpi_disabled() or not _MPI_AVAILABLE:
        return 0
    return MPI.COMM_WORLD.Get_rank()

def global_mpi_size() -> int:
    if mpi_disabled() or not _MPI_AVAILABLE:
        return 1
    return MPI.COMM_WORLD.Get_size()

def mpi_rank() -> int:
    if mpi_disabled():
        try:
            return torch.distributed.get_rank()
        except Exception:
            return 0
    return mpi_comm().Get_rank()

def mpi_world_size() -> int:
    if mpi_disabled() or not _MPI_AVAILABLE:
        return 1
    return mpi_comm().Get_size()

def local_mpi_rank() -> int:
    if mpi_disabled() or not _MPI_AVAILABLE:
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        return 0
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return mpi_comm().Get_rank() % torch.cuda.device_count()
    return mpi_comm().Get_rank()

def local_mpi_size() -> int:
    if mpi_disabled() or not _MPI_AVAILABLE:
        return 1
    return local_comm.Get_size()

def mpi_barrier():
    mpi_comm().Barrier()

def local_mpi_barrier():
    local_mpi_comm().Barrier()

def mpi_broadcast(obj, root=0):
    return mpi_comm().bcast(obj, root=root) if mpi_world_size() > 1 else obj

def mpi_allgather(obj):
    return mpi_comm().allgather(obj) if mpi_world_size() > 1 else [obj]

def mpi_isend(buf, dest, tag=0):
    if mpi_world_size() > 1:
        return mpi_comm().Isend(buf, dest=dest, tag=tag)
    return None

def mpi_send(buf, dest, tag=0):
    if mpi_world_size() > 1:
        mpi_comm().Send(buf, dest=dest, tag=tag)
    return None

def mpi_recv(buf, source, tag=0):
    if mpi_world_size() > 1:
        return mpi_comm().Recv(buf, source=source, tag=tag)
    return None

def mpi_send_object(obj, dest, tag=0):
    if mpi_world_size() > 1:
        mpi_comm().send(obj, dest=dest, tag=tag)

def mpi_isend_object(obj, dest, tag=0):
    if mpi_world_size() > 1:
        return mpi_comm().isend(obj, dest=dest, tag=tag)
    return None

def mpi_recv_object(source, tag=0):
    if mpi_world_size() > 1:
        return mpi_comm().recv(source=source, tag=tag)
    return None

def get_free_ports(num=1) -> list[int]:
    sockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(num)]
    for s in sockets:
        s.bind(("", 0))
    ports = [s.getsockname()[1] for s in sockets]
    for s in sockets:
        s.close()
    return ports

def get_free_port() -> int:
    return get_free_ports(1)[0]

def default_gpus_per_node() -> int:
    num_gpus = torch.cuda.device_count()
    num_ranks = local_mpi_size()
    if num_gpus <= 0:
        return 0
    if num_ranks > num_gpus:
        print(f"[WARNING] {num_ranks} MPI ranks will share {num_gpus} GPUs.")
    return min(num_ranks, num_gpus)

def print_stacks():
    """Print stack traces for all threads."""
    for thread_id, frame in sys._current_frames().items():
        print(f"Thread {thread_id} stack trace:\n{''.join(traceback.format_stack(frame))}")

def is_trace_enabled(env_var: str):
    value = os.environ.get(env_var, "-1")
    if value == "ALL":
        return True
    if value == "-1":
        return False
    try:
        return int(value) == global_mpi_rank()
    except ValueError:
        return False

def trace_func(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        def globaltrace(frame, why, arg):
            if why == "call":
                code = frame.f_code
                filename = frame.f_globals.get("__file__", None)
                if filename:
                    modulename = trace._modname(filename)
                    if modulename is not None:
                        ignore_it = tracer.ignore.names(filename, modulename)
                        if not ignore_it:
                            print(f"[rank{rank}] --- path: {filename} , funcname: {code.co_name}")
                            return localtrace
                else:
                    return None

        def localtrace(frame, why, arg):
            if why == "line":
                filename = frame.f_code.co_filename
                lineno = frame.f_lineno
                bname = os.path.basename(filename)
                print(f"[rank{rank}] {bname}:{lineno}: {linecache.getline(filename, lineno)}", end="")
            return localtrace

        ignoredirs = [os.path.dirname(package.__file__) for package in [os, torch, trace]]
        tracer = trace.Trace(trace=1, count=0, ignoredirs=ignoredirs)
        rank = global_mpi_rank()
        tracer.globaltrace = globaltrace
        tracer.localtrace = localtrace
        result = tracer.runfunc(func, *args, **kwargs)
        return result

    return wrapper
