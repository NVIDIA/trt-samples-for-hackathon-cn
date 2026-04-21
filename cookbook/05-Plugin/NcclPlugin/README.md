# NcclPlugin

+ Minimal TensorRT `PluginV3` + NCCL `send/recv` example.
+ Two GPUs, two processes, each process builds and runs one engine.

## Files

+ `NcclSendRecvPlugin.h/.cu`: One PluginV3 implementation, controlled by `mode`:
  + `mode=0`: `ncclSend`
  + `mode=1`: `ncclRecv`
+ `main.py`: Spawn 2 processes (`rank0` on GPU0 send, `rank1` on GPU1 recv), each process builds/loads `model-rank{rank}.trt` and runs inference.
+ `Makefile`: Build `NcclSendRecvPlugin.so`.

## Build and run

```bash
cd cookbook/05-Plugin/NcclPlugin
make -j
python main.py
```

## Notes

+ This example follows the organization style of `05-Plugin/BasicExample`.
+ NCCL runtime requirements apply; if environment is incomplete, child processes may fail.
+ Requires at least 2 visible GPUs.
