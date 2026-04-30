# DistCollective Layer

+ End-to-end example for `IDistCollectiveLayer` with NCCL communicator integration.

+ Steps to run (multi-process with torch.distributed + NCCL).

```bash
torchrun --nproc_per_node=2 main.py --op all_reduce
```
