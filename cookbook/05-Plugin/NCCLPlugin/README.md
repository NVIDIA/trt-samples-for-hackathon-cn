# NCCL Plugin

+ 极简 TensorRT `PluginV3` + NCCL `allReduce(sum)` 示例。
+ 参考 TensorRT-LLM 的 NCCL 插件思路：在插件中持有 communicator，并在 enqueue 里发起通信。
+ 示例目标：**单卡双进程**（2 个 rank 都绑到 GPU0）。

## 目录结构

+ `NCCLAllReducePlugin.h/.cu`：插件实现（仅支持 `float32` + `kLINEAR`）。
+ `main.cpp`：构建/加载 TensorRT 引擎并执行一次推理。
+ `launch_two_process.py`：生成共享 `ncclUniqueId` 并启动 2 个进程。

## 运行

```bash
make build
python3 launch_two_process.py
```

启动脚本会先顺序做一次 rank0/rank1 的 engine prepare，再并发启动两个进程做实际推理。

预期每个 rank 输出类似：

```text
[rank 0] output=3,3,3,3, expected=3, PASS
[rank 1] output=3,3,3,3, expected=3, PASS
```

## 说明

+ 此示例是教学最小版，省略了 TensorRT-LLM 里大量健壮性与多类型支持。
+ NCCL 官方推荐一进程一卡；同卡多 rank 在部分驱动/环境下可能失败。
+ 启动脚本默认设置了 `NCCL_IGNORE_DUPLICATE_RANK=1`、`NCCL_SOCKET_IFNAME=lo`、`NCCL_P2P_DISABLE=1`。
+ 若仍失败，请检查 GPU compute mode 不是 `EXCLUSIVE_PROCESS`（可用 `nvidia-smi -q` 查看）。
