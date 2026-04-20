import multiprocessing as mp
import socket
import traceback
from contextlib import closing
import torch

def worker(rank: int, world_size: int, device_id: int, master_port: int):
    import torch
    import torch.distributed as dist

    try:
        torch.cuda.set_device(device_id)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{master_port}",
            world_size=world_size,
            rank=rank,
        )

        send = torch.full((4, ), rank + 1, dtype=torch.float32, device=f"cuda:{device_id}")
        recv = torch.zeros_like(send)

        # rank0 -> rank1, then rank1 -> rank0 (ack)
        if rank == 0:
            dist.send(send, dst=1)
            dist.recv(recv, src=1)
            torch.cuda.synchronize(device_id)
            print(f"rank={rank}, device={device_id}, send={send.cpu().tolist()}, recv_ack={recv.cpu().tolist()}")
        else:
            dist.recv(recv, src=0)
            ack = recv + 10
            dist.send(ack, dst=0)
            torch.cuda.synchronize(device_id)
            print(f"rank={rank}, device={device_id}, recv={recv.cpu().tolist()}, send_ack={ack.cpu().tolist()}")

    except Exception as error:
        print(f"[worker-rank{rank}] 失败: {error}")
        traceback.print_exc()
        raise
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

if __name__ == "__main__":

    device_count = torch.cuda.device_count()
    assert device_count >= 2, f"Only {device_count} GPU is available."

    world_size = 2
    device_map = [0, 1]
    title = "multi-two-gpu"

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        master_port = int(sock.getsockname()[1])

    mp.set_start_method("spawn", force=True)

    procs = [mp.Process(target=worker, args=(rank, world_size, device_map[rank], master_port)) for rank in range(world_size)]

    for process in procs:
        process.start()

    for process in procs:
        process.join()

    if any(process.exitcode != 0 for process in procs):
        raise SystemExit(f"Subprocess {title} failed.")

    print("Finish")
