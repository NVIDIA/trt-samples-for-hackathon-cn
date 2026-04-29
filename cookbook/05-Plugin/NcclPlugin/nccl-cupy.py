import multiprocessing as mp

import cupy as cp
from cupy.cuda import nccl

def worker(rank: int, world_size: int, unique_id, device_id: int):

    cp.cuda.Device(device_id).use()

    comm = nccl.NcclCommunicator(world_size, unique_id, rank)

    send = cp.full((4, ), rank + 1, dtype=cp.float32)
    recv = cp.zeros_like(send)

    stream = cp.cuda.Stream.null
    comm.allReduce(
        send.data.ptr,
        recv.data.ptr,
        send.size,
        nccl.NCCL_FLOAT32,
        nccl.NCCL_SUM,
        stream.ptr,
    )
    stream.synchronize()

    print(f"rank={rank}, send={send.get().tolist()}, recv={recv.get().tolist()}")

if __name__ == "__main__":

    device_count = cp.cuda.runtime.getDeviceCount()

    print("=== 1) 单进程单卡 ===")
    cp.cuda.Device(0).use()
    unique_id = nccl.get_unique_id()
    comm = nccl.NcclCommunicator(1, unique_id, 0)

    send = cp.array([1, 2, 3, 4], dtype=cp.float32)
    recv = cp.zeros_like(send)

    stream = cp.cuda.Stream.null
    comm.allReduce(
        send.data.ptr,
        recv.data.ptr,
        send.size,
        nccl.NCCL_FLOAT32,
        nccl.NCCL_SUM,
        stream.ptr,
    )
    stream.synchronize()

    print("[single] rank=0")
    print(f"[single] send={send.get().tolist()}")
    print(f"[single] recv={recv.get().tolist()} (world_size=1 时应与 send 相同)")

    print("=== 3) 双卡双进程 ===")
    world_size = 2
    unique_id = nccl.get_unique_id()

    mp.set_start_method("spawn", force=True)
    procs = [mp.Process(target=worker, args=(rank, world_size, unique_id, rank)) for rank in range(world_size)]

    for p in procs:
        p.start()
    for p in procs:
        p.join()

    if any(p.exitcode != 0 for p in procs):
        raise SystemExit("[multi-two-gpu] 子进程执行失败，请检查 NCCL/CUDA 环境。")

    print("[multi-two-gpu] 2 进程 2 GPU allReduce 示例执行完成。")
