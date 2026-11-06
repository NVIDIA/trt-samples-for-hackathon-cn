# report-v6：perf_bugs kernel launch 延迟测试包

本包用 **NVIDIA 内部的 CUDA 驱动微基准 `perf_bugs`** 来测 kernel launch 延迟。
它和我们前几轮自己写的 `kernel_launch_bench` / `launch_roundtrip` 是**两套互相独立的实现**，
用它来交叉验证之前的结论；更重要的是它能额外报出 **每次 `cuLaunchKernel` 执行了多少条 CPU 指令**，
这是前几轮拿不到的信息。

---

## 一、为什么要加测"指令数"

前几轮已经定位到：远端环境 Pro5000 升级驱动到 615.71.09 之后，和 本地实验室 Pro5000 相比
还残留约 **300 ns** 的 `issue`（纯 CPU 侧发射）差距，而 `host`（同步后的墙上时间）两边完全一致。
我们已经排除了 GPU 硬件、CPU 型号/主频、绑核、NUMA、PCIe 槽位、可见卡数、CUDA Toolkit 版本、
gcc/glibc、可执行文件本身、以及虚拟化本身。**时间**这一个维度已经问不出更多东西了。

指令数是一个**与主频无关**的量，正好能把剩下的可能性一刀切开：

| 观察到的现象 | 结论 |
|---|---|
| 指令数相同，时间更长 | 驱动跑的是**同一段代码**，慢在 CPU/内存子系统（cache miss、TLB、MMIO 被 trap、频率…） |
| 指令数更多 | 驱动走了**不同的代码路径**（版本差异、不同的 fallback 分支、额外的检查） |

所以这一轮的核心产出是：**同一张表里同时看时间和指令数**。

---

## 二、包里有什么

```
report-v6/
├── README.md             本文件
├── run_perf_bugs.sh      主脚本，一条命令跑完
├── pick_core.sh          选绑定核（自动避开跨 NUMA）
└── bin/
    ├── perf_bugs                  可执行文件 (7.8 MB)
    ├── launch_latency_testlist.txt
    └── init_testlist.txt
```

**依赖极少**：`perf_bugs` 只链接 `libcuda.so.1` + 标准 C/C++ 库。
**不需要安装 CUDA Toolkit，不需要换驱动，不需要 root，不需要编译。**
机器上有 NVIDIA 驱动就能跑。

---

## 三、怎么跑

```bash
tar xzf report-v6.tar.gz
cd report-v6
bash run_perf_bugs.sh <给这台机器起个标签>
```

例如：

```bash
bash run_perf_bugs.sh remote-pro5000-615.71
```

跑完约 **1 分钟**，结果全部在 `result_<标签>/` 目录下，终端最后会打印一张汇总表。
然后把整个目录打包回传：

```bash
tar czf result_<标签>.tar.gz result_<标签>
```

可选环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `GPU` | `0` | 用哪张卡（`nvidia-smi` 里的 index） |
| `COUNT` | `50000` | 每轮的 launch 次数 |
| `REPEAT` | `3` | 正式测量轮数（另有 1 轮丢弃） |
| `PREWARM` | `4` | CPU 预热秒数 |
| `FREQ` | `1` | 顺带实测 CPU **实际运行频率**（见下）。无 `cc` 或无 PMU 权限时自动跳过 |
| `NOCOMPAT` | `0` | `=1` 时从 `LD_LIBRARY_PATH` 剔除 `.../cuda/compat/lib`，强制回落到宿主机注入的 libcuda |

### 容器内做 UMD A/B（强烈推荐，只要一分多钟）

很多 CUDA 镜像自带 `cuda-compat` 包，容器里**同时存在两个驱动用户态库**：
compat 包提供的新版，和 nvidia-container-toolkit 从宿主机注入的旧版。
`NOCOMPAT=1` 可以在**同一个容器里**切换两者，做一组除 UMD 外完全无差异的对照：

```bash
bash run_perf_bugs.sh <标签>-compat            # A: 用 compat 的 UMD
NOCOMPAT=1 bash run_perf_bugs.sh <标签>-host   # B: 用宿主机的 UMD
```

`env.txt` 的 `### libcuda` 段会打印实际加载的 `libcuda.so.<版本号>`，可直接核对切换是否生效。
**如果两次跑出来的版本号相同**，说明该镜像没装 compat 包，这组对照做不了（只交 A 即可）。

```bash
GPU=3 COUNT=100000 bash run_perf_bugs.sh remote-pro5000-gpu3
```

---

## 四、脚本替 perf_bugs 补了哪些坑

`perf_bugs` 是内部测试工具，直接裸跑会踩两个坑，本脚本都已经处理：

1. **它默认把自己钉在 0 号核上。**
   0 号核通常在处理中断，而且在双路机器上大概率和 GPU 不在同一个 NUMA 节点。
   我们实测跨 NUMA 会让 launch 多付 **200~410 ns**（和我们要找的 300 ns 残差是同一个量级，
   不排除会直接把结论带偏）。
   → 脚本用 `--dont-pin-threads` 解除它的自绑定，再用 `pick_core.sh` 选一个**和 GPU 同 NUMA 节点**的核，
   用 `taskset` 绑上去。

2. **它不做 CPU 预热，也不丢弃首轮。**
   在 `ondemand` / `powersave` governor 下，CPU 从空闲爬到满频要几十毫秒，
   而 perf_bugs 自己的 warmup 远不够。我们实测**不预热会让结果虚高 23%**
   （1684 ns vs 1366 ns，同一台机器同一张卡）。
   → 脚本先在目标核上空转 4 秒把频率拉满，再丢弃一轮，然后才正式测 3 轮。

> ⚠️ **请不要跳过 `run_perf_bugs.sh` 直接裸跑 `bin/perf_bugs`**，
> 否则拿到的数会同时混进上面两个误差，没法和别的机器比。

---

## 五、怎么拿到指令数（三条路，任选其一）

指令数通过 Linux 的 PMC（性能计数器）读取。内核对此有两道门，**满足任意一道即可**：

### 路线 0：裸机 / 不用容器 —— 直接用 root 跑

```bash
sudo bash run_perf_bugs.sh <标签>
```

root 默认就持有 `CAP_PERFMON`，直接绕过 paranoid 检查，**不需要改任何参数、不需要任何 flag**。
只有在容器里（能力被裁剪过）才需要下面的路线 A。

### 路线 A：容器创建时加两个 flag（**不需要宿主机 sudo，已实测验证**）

```bash
docker run --cap-add=PERFMON --user 0 ...
```

**两个 flag 缺一不可**，这是实测踩出来的：
`--cap-add` 只把能力放进 bounding + permitted 集；容器内若以**非 root 用户**运行，
permitted 实际为空，能力完全拿不到，`perf_event_open` 仍然返回 `Permission denied`。
加上 `--user 0` 之后一次成功。

> ⚠️ 必须在**创建容器**时加，已经在跑的容器改不了（加到 `docker exec` 上无效）。
> 如果 docker 版本不认 `PERFMON`（需内核 5.8+），改用 `--cap-add=SYS_ADMIN`。

### 路线 B：宿主机上改一个内核参数（需要宿主机 sudo）

```bash
cat /proc/sys/kernel/perf_event_paranoid        # 先记下原值
sudo sysctl -w kernel.perf_event_paranoid=3     # 临时，重启后失效
# 测完改回原值
```

该值只影响"非特权用户能不能读性能计数器"，不改变任何调度/内存/驱动行为，不重启、不影响业务。

### 路线 C：两条都走不通 —— 直接跑，只看时间

`perf_bugs` 在拿不到 PMC 时会打印 `Failed to enable PMC to count instructions`，
然后**正常跑完并正常报出所有时间指标**。时间是我们全部跨机器对比的基础，本身完整可用。
**指令数是加分项，不是前提。不要因为这一个参数卡住整轮测试。**

### 出问题时怎么自查

脚本会把诊断信息写进 `result_<标签>/env.txt` 的"身份与 capabilities"段
（uid / CapPrm / CapEff / CapBnd / CapAmb / CAP_PERFMON 是否在 Eff 集 / Seccomp 状态），
启动时也会直接提示。另外看 `host_launch_latency_run1.txt` 里的报错码：

| 报错 | 含义 |
|---|---|
| `perf_event_open ...: Permission denied`（EACCES） | 内核 paranoid 检查没过 → 走路线 A 或 B |
| EPERM | seccomp 拦截 → 加 `--security-opt seccomp=unconfined` |
| `No such file or directory`（ENOENT）/ 指令数恒为 0 | **没有可用的 PMU**，见下面的虚拟机说明 |

### ⚠️ 虚拟机（KVM）里的特别说明

指令数靠的是 CPU 的硬件性能计数器（PMU）。**KVM 默认不给虚拟机暴露 vPMU**，
这种情况下无论容器加什么 flag、宿主机 sysctl 怎么改，指令数都拿不到——
这是虚拟化层的限制，不是权限问题。

需要在**宿主机**上给这台虚拟机开 vPMU：

- QEMU 直接起：`-cpu host,pmu=on`
- libvirt：domain XML 的 `<cpu>` 里加 `<feature policy='require' name='pmu'/>`

另外即使开了 vPMU，虚拟机里的计数值也可能不如裸机精确。
**所以"裸机 vs KVM"这组对照，请以【时间】为主要依据，指令数只作参考。**

## 五·五、CPU 实际运行频率（默认已开）

kernel launch 是纯 CPU 的活，**主频直接决定成绩**。但在虚拟机里
`/proc/cpuinfo` 的 `cpu MHz` 和 `cpufreq` 往往是宿主机的标称值，不反映真实运行频率。

脚本会用 PMU 的 `CPU_CYCLES` 计数器除以 CPU 占用时间来实测，
**与 `/proc/cpuinfo`、cpufreq、MSR、TSC 全都无关**，虚拟机里同样有效。
结果打印在汇总表上方（`★CPU 实测运行频率★`），原始时间序列在 `result_*/freq.txt`。

取的是**稳态中位数**（丢掉开头几个采样点，避开 governor 的爬升过程），
因为整段平均会被 perf_bugs 的初始化阶段拉低。

前提：机器上有 `cc`/`gcc`，以及和指令数相同的 PMU 权限。
任一条件不满足会自动跳过，不影响其它指标。

> 虚拟机里如果报 `ENOENT`，说明 guest 没有可用的 vPMU，
> 需要宿主机给这台虚机开（QEMU `-cpu host,pmu=on`）。

参考值：本地实验室 Xeon w5-3425 在负载下实测 **4.03 GHz**
（与 `scaling_cur_freq` 报的 4100 MHz 一致，可互相印证）。

---

## 六、怎么读结果

终端最后的汇总表（也存在 `result_<标签>/summary.txt`）长这样：

```
指标                                   run1       run2       run3
cuLaunchKernel 耗时 (ns)             1350.4     1379.7     1346.4
cuLaunchKernel 指令数 (条)           8334.2     8334.6     8334.3
mean_host_latency (ns)                 1404       1377       1394
mean_total_latency (ns)                3207       3184       3197
device_launch_latency (ns)             1803       1807       1803
```

各指标含义：

| 指标 | 含义 | 对应我们之前的哪个量 |
|---|---|---|
| **cuLaunchKernel 耗时** | 纯 CPU 侧，连续发射不同步，单次 `cuLaunchKernel` 的平均开销 | ≈ 我们的 **`issue`** |
| **cuLaunchKernel 指令数** | 单次 `cuLaunchKernel` 在 CPU 上执行的指令条数 | 新增，之前没有 |
| **mean_host_latency** | 同上，另一个测项的独立实现 | ≈ `issue`，用来自洽校验 |
| **mean_total_latency** | 从 CPU 发射到 GPU 上 kernel 真正开始跑的端到端延迟 | ≈ 我们的 **`wall`/`event`** |
| **device_launch_latency** | GPU 侧看到的那一段 | ≈ `event − issue` |

三轮之间的波动正常在 **±1~3%** 以内。如果某一项三轮之间差了 10% 以上，
说明机器上有别的负载在抢 CPU，请在空闲时重测。

### 已有的基准值（用来对照）

| | 本地 Pro5000 48G<br>driver **615.44**<br>Xeon w5-3425 单路裸机 | 第三方环境 Pro5000 72G<br>UMD **610.43.02** (compat)<br>EPYC 9575F 双路 | 第三方环境 Pro5000 72G<br>UMD **580.95.05**<br>同一台同一容器 |
|---|---|---|---|
| cuLaunchKernel 耗时 | **1363 ns** | **1397 ns** | **1980 ns** |
| cuLaunchKernel 指令数 | **8334 条** | **8728 条** | **9921 条** |
| mean_host_latency | 1385 | 1275 | 1436 |
| mean_total_latency | 3190 | 3355 | 4089 |
| device_launch_latency | 1805 | 2081 | 2652 |

最右两列是**同一台机器、同一个容器、同一个 CPU 核、同一个内核驱动(KMD)**，
只切换了用户态驱动库(UMD)。这组对照的意义见下。

### 怎么看"时间 + 指令数"这一对

右边两列的差距可以干净地分解成两部分：

```
时间 1.42×  =  指令数 1.14×  ×  每条指令耗时 1.25×
```

- **指令数多了 14%** → 旧 UMD 走的是**不同的代码路径**，不是同一段代码跑慢了。
- **每条指令还更慢 25%** → 多出来的那段代码访存行为也更差（cache miss / 锁竞争 / 等待）。

还有一个信号：**新 UMD 的指令数三轮完全一致**（8728.25 / 8728.19 / 8728.22，离散度 ~0），
而**旧 UMD 的指令数三轮波动 3.8%**（9964.7 / 9712.1 / 10086.1）。
指令数不稳定说明旧路径里有**数据相关的分支或重试循环**——
这类循环正是"有时多转几圈"的典型特征。

所以拿到你机器上的数之后，先看指令数：

| 和参照列比 | 说明 |
|---|---|
| 指令数接近 8300~8800 | 驱动是健康的代码路径 |
| 指令数接近 9900 且三轮不稳 | 命中了已知的旧 UMD 回归 |
| 指令数正常但时间明显偏长 | 不是驱动代码路径问题，去查 CPU 频率 / NUMA / 虚拟化 |

## 七、这一轮的三步走（① 已完成）

| 顺序 | 机器 | 状态 | 要回答的问题 |
|---|---|---|---|
| ① | 第三方环境（580 KMD，容器内 compat 610.43.02） | ✅ **已完成** | 有问题的 UMD 长什么样？→ **9921 条指令 / 1980 ns，且指令数不稳定** |
| ② | 同事的裸金属 / KVM，driver 615 | ⬜ 待做 | 同一驱动下裸金属 vs KVM 的指令数是否一致？（锁定虚拟化层有无额外开销） |
| ③ | 远端环境实际掉速环境，driver 615.71 | ⬜ 待做 | 和 本地实验室 615.44（8334 条 / 1363 ns）比，是"指令一样但更慢"还是"指令更多"？ |

③ 是最终要回答的那一问，① 已经把参照系建好了。

**给 ②③ 的具体建议**：

- 容器创建时务必带上 `--cap-add=PERFMON --user 0`（见第五章），
  指令数这一维度不拿到的话，这轮测试的价值会打对折。
- 如果镜像自带 `cuda-compat`，**请把 `NOCOMPAT=1` 那一组也跑了**（见第三章），
  一共才多花一分钟，但能在你自己的机器上就地得到一组干净的 UMD 对照。
- 跑完把 `result_*` 整个目录打包回传，不要只截汇总表 —— 原始文件里有诊断信息。

## 八、已知限制

- `perf_bugs --print-all-nodes` 可以打印更细的子节点（`launchCheck` / `launchInitialize` /
  `launchFinalizeQMD` 等），但那需要一个用 `CUDA_PERFORMANCE=1` 编译的插桩版 `libcuda`。
  **正式发布的驱动上只能看到 `cuLaunchKernel` 这一层**，再往下拆不开。
  如果最后需要拆到子节点，得走 NVIDIA 内部驱动团队的路子。
- 汇总表里还有几行 `*_sync` 变体（约 5300 ns），那是"每发射一次就同步一次"的测法，
  包含了 GPU 往返，**不要拿它和上面的 1360 ns 直接比**。
