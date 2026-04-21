# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import ctypes.util
import os
import subprocess
from pathlib import Path

class NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_ubyte * 128)]

def get_nccl_unique_id_hex() -> str:
    lib_name = ctypes.util.find_library("nccl")
    if lib_name is None:
        raise RuntimeError("Cannot find libnccl")

    lib = ctypes.CDLL(lib_name)
    fn = lib.ncclGetUniqueId
    fn.argtypes = [ctypes.POINTER(NcclUniqueId)]
    fn.restype = ctypes.c_int

    uid = NcclUniqueId()
    ret = fn(ctypes.byref(uid))
    if ret != 0:
        raise RuntimeError(f"ncclGetUniqueId failed, code={ret}")
    return bytes(bytearray(uid.internal)).hex()

def main() -> None:
    workdir = Path(__file__).resolve().parent
    uid_hex = get_nccl_unique_id_hex()

    env_base = os.environ.copy()
    env_base["NCCL_WORLD_SIZE"] = "2"
    env_base["CUDA_VISIBLE_DEVICES"] = "0"
    env_base.setdefault("NCCL_IGNORE_DUPLICATE_RANK", "1")
    env_base.setdefault("NCCL_SOCKET_IFNAME", "lo")
    env_base.setdefault("NCCL_P2P_DISABLE", "1")

    for rank in (0, 1):
        env = env_base.copy()
        env["NCCL_RANK"] = str(rank)
        env["NCCL_UID_HEX"] = uid_hex
        env["NCCL_PREPARE_ONLY"] = "1"
        prep = subprocess.run(
            ["./main.exe"],
            cwd=workdir,
            env=env,
        )
        if prep.returncode != 0:
            raise SystemExit(f"Prepare rank {rank} failed")

    procs = []
    for rank in (0, 1):
        env = env_base.copy()
        env["NCCL_RANK"] = str(rank)
        env["NCCL_UID_HEX"] = uid_hex
        procs.append(subprocess.Popen(
            ["./main.exe"],
            cwd=workdir,
            env=env,
        ))

    all_ok = True
    for p in procs:
        p.wait()
        if p.returncode != 0:
            all_ok = False

    if not all_ok:
        raise SystemExit("NCCLPlugin 双进程示例失败")

    print("NCCLPlugin 单卡双进程示例完成")

if __name__ == "__main__":
    main()
