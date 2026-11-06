# GpuLaunchDiag

+ 一套回答「CPU 把一个 kernel 发射到 GPU 要花多少代价、这台机器为什么更贵」的诊断工具。

> 改这个文件的开头前先知道：`build-Copyright-and-README.py` 会把**前三行**抄进 `90-Misc/README.md`，
> 一级标题直接成为那里的小节名。所以 h1 必须在同一个父目录下**唯一**（`DeviceInfo` 与 `LibraryInfo`
> 曾经都叫 `# get_device_info.py`，父级因此出现两个同名小节），第 3 行也要是**一句完整的话**，断在
> 折行处会在父级留下残句。h1 不必等于目录名 —— 仓库里多数 README 用的是可读标题，如 `02-API/Dims`
> 的 `# Dims API`。

由 2026-09 ~ 2026-10 一次「RTX PRO 5000 vs L20 发射开销」调查沉淀而来，现作为通用工具维护。
被测对象可以是任何 GPU / 驱动 / CPU / 容器组合，与当初那次调查无关。

## 与 DeviceInfo / LibraryInfo 的分工

同在 `90-Misc/` 下的三者回答的是不同问题，不要混用：

| 工具 | 回答 | 数据来源 |
| --- | --- | --- |
| `DeviceInfo` | 这块 GPU 的**架构参数**是什么（SM 数、共享内存、算力） | `cudaGetDeviceProperties` / torch |
| `LibraryInfo` | 装了**哪些版本**的软件（驱动、CUDA、cuDNN、TensorRT） | `nvidia-smi` + 头文件 |
| **本工具** `collect_env.sh` | 这台机器**此刻处于什么状态**，以及实际加载的是哪个 libcuda | `/sys`、`/proc`、`lscpu`、`dlopen` |

三者有少量重叠（GPU 名称、驱动版本、torch 版本），**是刻意保留的**：`collect_env.sh` 的价值在于它是
一个自包含、只读、10 秒跑完的单文件快照，能直接附进 bug 报告，也是 `make_*_pkg.sh` 那套独立小包
能成立的前提。让它去调用另外两个 Python 脚本会毁掉这个属性，省下的不过五行。

## ⚠️ 关于 `bin/perf_bugs`

`perf_bugs` 是 NVIDIA 内部驱动微基准，**不随本仓库发布**（见 `cookbook/.gitignore`）。公开检出里
它不存在，受影响的脚本会**跳过并提示"请与作者联系"，而不是报错**，退出码为 0：

| 脚本 | 没有 perf_bugs 时 |
| --- | --- |
| `run_perf_bugs.sh`（T5） | 整项跳过，提示改用 `run_launch_pmu.sh`（T4，测同一件事，任何驱动版本可跑） |
| `make_perfbugs_pkg.sh` | 跳过，提示改用 `make_launchpmu_pkg.sh` |
| `make_summary_pkg.sh` | 照常打包，但包内不含该二进制，结尾据实说明 |
| `collect_env.sh` | 自动改用 python `dlopen` 探针解析 libcuda，结果与用 perf_bugs 探测一致 |

**其余全部测项（T0~T4、T6~T7）都不依赖它。**

---

## 文档入口

| 文件 | 内容 |
| --- | --- |
| **[诊断手册.md](诊断手册.md)** | ★ 主文档 ★ 每个测项**测什么 / 预期多少 / 偏了说明什么**，含诊断决策树 |
| **[基线参考值.md](基线参考值.md)** | 全部实测基线数据（41 项全表、队列深度、指令数横向对比、拓扑扫描） |
| **[baselines/README.md](baselines/README.md)** | 已验证的参考运行（2026-10-30 实跑，41 项全部 ±4.1% 内复现） |
| `docs/launchpmu-README.md` | T4 独立小包的对外说明书 |
| `docs/perfbugs-README.md` | T5 独立小包的对外说明书 |

---

## 最快上手

```bash
# 0. 环境采集（10 秒，永远先跑）
bash collect_env.sh > 01_env.txt

# 1. 一次跑完全景（约 20 分钟）
./run_all.sh <给这台机器起个唯一标签>

# 2. 怀疑驱动有深队列问题时（5 分钟）
bash run_queue_depth.sh <标签>
NOCOMPAT=1 bash run_queue_depth.sh <标签>-host     # 测老驱动必须加

# 3. 需要指令数（与 CPU 主频无关的诊断量）时（1 分钟）
bash run_launch_pmu.sh <标签>
```

**跳过耗时步骤**：`SKIP_PERCORE=1 SKIP_SLOT=1 ./run_all.sh <标签>`

**前置**：`nvidia-smi` + `nvcc` 必需；`cc`、免密 `sudo`、PMU 权限可选（没有也能跑，功能递减）。
**全程只读，不修改任何系统配置。**

---

## 三条纪律（违反任意一条都会得到错误结论）

1. **丢弃第一次运行** —— 冷启动惩罚实测高达 **2.05 倍**
2. **看 min / p50，绝不看 mean** —— 抢占只会让结果变慢；见过单次 0.88 ms 的调度打嗝
3. **核对实际加载的 `libcuda`** —— 容器里 `cuda-compat` 会偷换 UMD，
   我们自己就把「580 的测量」做成了「610 的测量」

详见 `诊断手册.md` 第二节。

---

## 目录结构

```
GpuLaunchDiag/
├── README.md               本文件
├── 诊断手册.md             ★ 主文档：测项 / 预期 / 偏差诊断 / 决策树
├── 基线参考值.md           ★ 全部实测基线数据
│
├── run_all.sh              主入口 —— T0~T2 + T6 一次跑完
├── collect_env.sh          T0 环境采集（只读）★含 libcuda 实际解析路径★
├── measure_launch.sh       T1 完整发射基准（4 趟，丢弃 run1）
├── run_roundtrip.sh        T2 往返三段拆分 + 纯 CPU 发射成本
├── run_queue_depth.sh      T3 队列深度扫描 ★判定驱动有无深队列缺陷★
├── run_launch_pmu.sh       T4 PMU 两点差分（任意驱动版本）
├── run_perf_bugs.sh        T5 perf_bugs 运行器（需 driver ≥570）
├── percore_scan.sh         T6 逐核扫描（NUMA 诊断）
├── gpu_slot_scan.sh        T6 槽位 / 可见设备数扫描
├── kvm_mmio_host.sh        T7 KVM 宿主机侧 mmio_exits 采样（需 root）
│
├── pick_core.sh            NUMA 感知选核（所有脚本共用）
├── compare_results.py      对比两份基准 CSV（先校验元数据是否可比）
├── make_*_pkg.sh           生成对外的独立小包
│
├── baselines/              ★ 已验证的参考运行（换新环境时拿来对照，已脱敏）
│
├── bench/
│   ├── kernel_launch_bench_split.cu   T1：41 个测项
│   ├── launch_roundtrip.cu            T2：往返拆分 + issue 扫描
│   ├── launch_pmu_probe.cu            T3/T4：精确发射 T 次
│   ├── freq_monitor.c                 PMU 读 cycles/instructions，测实际主频
│   └── mmio_probe.cu                  T7：精确发射 N 次
├── bin/
│   ├── perf_bugs                      ★ 内部工具，被 .gitignore 排除，公开检出里没有 ★
│   └── *_testlist.txt
└── docs/
```

> ⚠️ **`bin/perf_bugs` 是 NVIDIA 内部驱动微基准，不得外传**，因此不随本仓库发布，详见开头那一节。
> 对外分享请用 `make_launchpmu_pkg.sh` 生成只含源码的 ~10 KB 小包，
> 它不依赖 perf_bugs，且任何驱动版本都能跑。
>
> ⚠️ **T8（真实模型 QPS）没有可运行的压测器。** 原压测器源自客户项目，不适合随这个公开仓库
> 发布，已整体移除。方法、以及给 AOTI 模型套 CUDA Graph 的四条前提，保留在
> `诊断手册.md` 第十一节 —— 要复现得自己实现一个压测器。

## 本目录不参与 `tests/run_tests.py` 的默认扫描

目录里放了 `.skip_unit_test`。`run_all.sh` 要跑约 20 分钟、需要现场 `nvcc` 编译，部分测项还要免密
`sudo` 与 PMU 权限 —— 这些都不该出现在一次常规 sweep 里。要跑就按上面的「最快上手」手动跑。
