# launch-PMU 测试包：任何驱动版本都能跑的发射开销测量

## 为什么有这个包

前一个 `perfbugs-kit` 需要 **driver ≥ 570**（它引用了 `cuCtxCreate_v4` 等较新的 API），
在老驱动的机器上根本起不来。本包只用 Linux 的 PMU 计数器和一个自带的小探针，
**任何驱动版本都能跑**，不需要 root，不需要改驱动。

一次测量同时给出每次 kernel launch 的：**指令数 / 周期数 / 耗时 / IPC / 实际运行频率**。

## 怎么跑

```bash
tar xzf launchpmu-kit.tar.gz
cd launchpmu-kit
bash run_launch_pmu.sh <给这台机器起个标签>
```

约 1 分钟。跑完把 `result_launchpmu_*` 整个目录打包回传。

可选：`GPU=0 REPEAT=3 bash run_launch_pmu.sh <标签>`

## 前置条件

| | 说明 |
|---|---|
| `nvcc` | 用来编译探针。机器上原有的 CUDA 工具链即可，版本不限 |
| `cc` / `gcc` | 用来编译 PMU 采集器 |
| `perf_event_paranoid ≤ 3` | 或容器带 `CAP_PERFMON`。脚本会在 `env.txt` 里记录当前值 |

**不需要** root，**不需要**改驱动，**不需要**装新的 CUDA。

## 原理

用 PMU 采集器包住探针，分别发射 `T1` 和 `T2` 次，两点差分：

```
每次 launch 的指令数 = (指令_T2 − 指令_T1) / (T2 − T1)
```

差分把 CUDA 初始化、建 context、预热等固定开销全部消掉。
探针按 1024 次一批同步，避免发射队列填满后 CPU 被背压阻塞（那会把指令数也带偏）。

## ⚠️ 一条重要的口径限制

**本包的指令数不能和 `perfbugs-kit` 的指令数直接比较。**
本包走 CUDA runtime API（`cudaLaunchKernel` → `cuLaunchKernel`），
perf_bugs 直接调 driver API，中间差一层包装——本地实验室同机实测差约 **792 条**。

→ **跨机器比较时，请所有机器都用本包**，不要和 perf_bugs 的数混用。

## 参考值（本地实验室，供对照）

| | Pro5000 48G / driver 615.71.09 / Xeon w5-3425 |
|---|---|
| 每次 launch 指令数 | **9081** |
| 每次 launch 周期数 | **5622** |
| 每次 launch 耗时 | **1389 ns** |
| IPC | **1.615** |
| 实测频率 | **4.05 GHz** |

交叉验证：耗时与 perf_bugs 的 1367 ns 差 1.6%，
隐含频率与独立测得的 4.01 GHz 及系统报告的 4.10 GHz 一致。

**重复性**：指令数的轮间波动约 ±0.5%（差分法天然比 perf_bugs 的 ±0.01% 噪声大，
因为是两个大数相减）。要分辨的效应在 5% 以上，这个精度够用。

## 结果怎么看

- **指令数相同、耗时更长** → 驱动跑的是同一段代码，慢在 CPU/内存子系统
- **指令数不同** → 驱动走了不同的代码路径（版本、GPU 架构、CPU 特性都可能导致）
- **隐含频率** 就是发射循环期间 CPU 的实际运行频率，
  虚拟机里 `/proc/cpuinfo` 的 `cpu MHz` 往往是标称值，不可信，请以这个为准
