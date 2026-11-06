#!/usr/bin/env bash
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

# =============================================================
#  perf_bugs kernel launch 延迟测试   (report-v6)
#
#  用法:  bash run_perf_bugs.sh [标签]
#         GPU=0 COUNT=50000 REPEAT=3 bash run_perf_bugs.sh my-machine
#
#  不需要装 CUDA Toolkit，只需要机器上已有 NVIDIA 驱动 (libcuda.so.1)。
#  不需要 root。全部结果写到 result_<标签>/ 目录。
# =============================================================
set -u
cd "$(dirname "$0")"
HERE=$(pwd)

# T5 是唯一硬依赖 bin/perf_bugs 的测项，而 perf_bugs 是 NVIDIA 内部驱动微基准，不随本仓库发布
# （见 cookbook/.gitignore）。公开检出里它不存在，这里跳过而不是报错 —— 退出码 0，好让
# run_all.sh 之类的调用方继续往下跑。先于 mkdir "$OUT" 判断，免得在公开检出里留下空目录。
if [ ! -x "$HERE/bin/perf_bugs" ]; then
    echo "跳过 T5：本测项依赖 NVIDIA 内部工具 perf_bugs，此处不公开，请与作者联系。"
    echo "  不含该工具时的替代：bash run_launch_pmu.sh <标签>"
    echo "  T4 用 PMU 两点差分测同一件事，不依赖 perf_bugs，且任何驱动版本都能跑。"
    exit 0
fi

TAG="${1:-$(hostname | tr -cd 'A-Za-z0-9_-')}"
GPU="${GPU:-0}"
COUNT="${COUNT:-50000}"      # 每次测试的 launch 次数
REPEAT="${REPEAT:-3}"        # 正式测量轮数（另有 1 轮丢弃）
PREWARM="${PREWARM:-4}"      # CPU 预热秒数
FREQ="${FREQ:-1}"            # =1 时顺带实测 CPU 实际运行频率（需要 gcc 与 PMU 权限）
NOCOMPAT="${NOCOMPAT:-0}"    # =1 时从 LD_LIBRARY_PATH 里剔除 .../cuda/compat/lib，
                             #    强制回落到宿主机注入的 libcuda，用来在同一个容器里做 UMD A/B

if [ "$NOCOMPAT" = 1 ]; then
    export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
        | grep -v 'cuda/compat' | paste -sd: -)
fi
OUT="$HERE/result_perfbugs_${TAG}"
mkdir -p "$OUT"
LOG="$OUT/run.log"
exec > >(tee "$LOG") 2>&1

echo "=============================================="
echo " perf_bugs launch latency  —  标签: $TAG"
echo " 时间: $(date -Is)"
echo "=============================================="

# ---------- 0. 基本检查 ----------
BIN="$HERE/bin/perf_bugs"   # 存在性已在脚本开头判过，那里会跳过而非失败
if ! ldd "$BIN" 2>/dev/null | grep -q 'libcuda.so.1 =>.*/'; then
    echo "❌ 找不到 libcuda.so.1，请确认本机装了 NVIDIA 驱动"
    ldd "$BIN" | grep -i cuda
    exit 1
fi

# ---------- 1. 环境采集 ----------
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
DRVMAJ=${DRV%%.*}
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 99)
# CAP_PERFMON 是 38 号能力；十六进制 CapEff 从右数第 10 个 nibble 的 bit2
CAPEFF=$(awk '/^CapEff/{print $2}' /proc/self/status 2>/dev/null)
PERFMON_EFF=未知
if [ -n "$CAPEFF" ]; then
    NIB=$(printf '%s' "$CAPEFF" | tail -c 10 | head -c 1)
    if [ -n "$NIB" ] && [ $(( 0x$NIB & 4 )) -ne 0 ] 2>/dev/null; then PERFMON_EFF=是; else PERFMON_EFF=否; fi
fi
{
  echo "### hostname / kernel"
  hostname; uname -a
  echo; echo "### driver / GPU"
  nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version,memory.total --format=csv 2>&1
  echo; echo "### libcuda  ★本项最关键★"
  ldd "$BIN" | grep -i cuda
  LC=$(ldd "$BIN" | sed -n 's/.*libcuda\.so\.1 => \(\S*\).*/\1/p')
  [ -n "$LC" ] && { ls -l "$LC"; ls -l "$(dirname "$LC")"/libcuda.so.* 2>/dev/null; }
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-(未设置)}"
  echo; echo "### CPU"
  lscpu 2>/dev/null | grep -Ei 'model name|^CPU\(s\)|Thread|Core|Socket|NUMA|MHz'
  echo; echo "### governor / 当前频率"
  cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c
  grep -m4 'cpu MHz' /proc/cpuinfo 2>/dev/null
  echo; echo "### clocksource"
  cat /sys/devices/system/clocksource/clocksource0/current_clocksource 2>/dev/null
  echo; echo "### 虚拟化"
  V=$(systemd-detect-virt 2>/dev/null); echo "systemd-detect-virt: ${V:-(不可用)}"
  grep -m1 -o 'hypervisor' /proc/cpuinfo 2>/dev/null || echo "(无 hypervisor flag)"
  echo; echo "### perf_event_paranoid"
  echo "$PARANOID   (99 = 读不到)"
  echo; echo "### 身份与 capabilities (指令数拿不到时看这里)"
  echo "uid=$(id -u)  euid=$(id -u -r 2>/dev/null)  user=$(id -un 2>/dev/null)"
  grep -E '^Cap(Prm|Eff|Bnd|Amb)' /proc/self/status 2>/dev/null
  echo "CAP_PERFMON 是否在 Eff 集: $PERFMON_EFF"
  command -v capsh >/dev/null && capsh --decode=$(awk '/^CapEff/{print $2}' /proc/self/status) 2>/dev/null | tr ',' '\n' | grep -i 'perfmon\|sys_admin'
  echo; echo "### seccomp (0=关闭 2=filter)"
  grep -E '^Seccomp:' /proc/self/status 2>/dev/null
} > "$OUT/env.txt" 2>&1
echo "[1/4] 环境已采集 → $OUT/env.txt"
grep -E 'Model name|driver_version|^[0-9]+,' "$OUT/env.txt" | head -5 | sed 's/^/      /'

if [ "$PARANOID" -le 3 ] 2>/dev/null; then
    echo "      perf_event_paranoid=$PARANOID → ✅ 可以拿到指令数"
else
    echo "      perf_event_paranoid=$PARANOID, CAP_PERFMON在Eff集=$PERFMON_EFF"
    if [ "$PERFMON_EFF" = 是 ]; then
        echo "      → 有 CAP_PERFMON，应该能拿到指令数"
    else
        echo "      → ⚠️  拿不到指令数（uid=$(id -u)；--cap-add 对非 root 用户无效，需 --user 0 或 ambient cap）"
        echo "         但【时间】照常测，结果依然有效"
    fi
fi

# perf_bugs 引用了 cuCtxCreate_v4 (CUDA 12.5+) 与 cuEventElapsedTime_v2 (CUDA 12.8+)，
# 老驱动里没有这些符号，程序会在惰性绑定时被动态链接器直接杀掉。
if [ -n "$DRVMAJ" ] && [ "$DRVMAJ" -lt 570 ] 2>/dev/null; then
    echo
    echo "❌ 这台机器的驱动是 $DRV，太老，跑不了 perf_bugs。"
    echo "   perf_bugs 是预编译的内部二进制，引用了只有较新驱动才提供的 API："
    echo "     cuCtxCreate_v4        （CUDA 12.5 / driver 555 起）"
    echo "     cuEventElapsedTime_v2 （CUDA 12.8 / driver 570 起）"
    echo "   没有源码，无法重编到老 API 上，所以这台机器上这个测试做不了。"
    echo
    echo "   👉 如果你要的是这台机器的【CPU 实际运行频率】，那不需要 perf_bugs："
    echo "      cc -O2 -o freq_monitor freq_monitor.c"
    echo "      taskset -c <核> ./freq_monitor -i 150 -- taskset -c <核> \\"
    echo "        bash -c 'end=\$((SECONDS+5)); while [ \$SECONDS -lt \$end ]; do :; done'"
    echo "      这个只用 PMU，和驱动版本完全无关。"
    echo
    exit 4
fi

# ---------- 2. 选核 ----------
CORE=$(bash "$HERE/pick_core.sh" --explain "$GPU" 2>"$OUT/pick_core.txt")
[ -z "$CORE" ] && CORE=0
echo "[2/4] 绑定 CPU 核 = $CORE"
sed 's/^/      /' "$OUT/pick_core.txt"

# ---------- 3. CPU 预热（关键！） ----------
# perf_bugs 自身不做预热也不丢弃首轮；在 ondemand/powersave governor 下
# 冷启动会把 launch 延迟抬高 20%~100%。必须先把这个核跑热。
echo "[3/4] CPU 预热 ${PREWARM}s (核 $CORE) ..."
taskset -c "$CORE" timeout "$PREWARM" bash -c 'while :; do :; done' 2>/dev/null

# ---------- 4. 跑测试 ----------
run_one() {  # $1=测项  $2=输出文件
    CUDA_VISIBLE_DEVICES="$GPU" taskset -c "$CORE" \
      "$BIN" -t "$1" --launch_count "$COUNT" --dont-pin-threads > "$2" 2>&1
}

# ---------- 3.5 编译频率探针 ----------
FM=""
if [ "$FREQ" = 1 ]; then
    SRC="$HERE/bench/freq_monitor.c"; [ -f "$SRC" ] || SRC="$HERE/freq_monitor.c"
    if [ -f "$SRC" ] && command -v cc >/dev/null 2>&1; then
        if cc -O2 -o "$OUT/freq_monitor" "$SRC" 2>"$OUT/freq_build.log"; then
            FM="$OUT/freq_monitor"
        else
            echo "      ⚠️ 频率探针编译失败，跳过（见 $OUT/freq_build.log）"
        fi
    else
        echo "      ⚠️ 无 cc 或找不到 freq_monitor.c，跳过频率测量"
    fi
fi

echo "[4/4] 开始测量 (launch_count=$COUNT, 丢弃首轮 + 正式 $REPEAT 轮)"
for T in host_launch_latency total_launch_latency; do
    echo "--- $T ---"
    run_one "$T" "$OUT/${T}_warmup.txt"
    for i in $(seq 1 "$REPEAT"); do
        run_one "$T" "$OUT/${T}_run${i}.txt"
        if [ "$T" = host_launch_latency ]; then
            # 值是 COUNT 次的总纳秒数，要除以 COUNT
            awk -v c="$COUNT" '/-time/ && /cuLaunchKernel/ {printf "      run'"$i"'  %-62s %8.1f ns\n", $3, $4/c}' "$OUT/${T}_run${i}.txt"
            awk '/-instructions/ {printf "      run'"$i"'  %-62s %8.1f 条指令\n", $3, $4}' "$OUT/${T}_run${i}.txt"
        else
            # mean_host_latency / device_launch_latency 本身就是 ns/次
            awk '/-time/ {printf "      run'"$i"'  %-62s %8s ns\n", $3, $4}' "$OUT/${T}_run${i}.txt"
            awk '/device_launch_latency|cpu-gpu_tick_scale|StdDev/ {printf "      run'"$i"'  %-62s %8s\n", $1, $3}' "$OUT/${T}_run${i}.txt"
        fi
    done
    if ! grep -q 'PASSED' "$OUT/${T}_run${REPEAT}.txt"; then
        echo "      ⚠️ 该测项未 PASS。实际报错："
        grep -m3 -iE 'symbol lookup error|undefined symbol|error|failed|cannot' \
            "$OUT/${T}_run${REPEAT}.txt" | sed 's/^/        /' || echo "        (无输出，程序可能被直接终止)"
        echo "      请把 $OUT 整个目录回传"
    fi
done

# ---------- 4.5 实测 CPU 实际运行频率 ----------
# 为什么要测：虚拟机里 /proc/cpuinfo 的 "cpu MHz" 和 cpufreq 往往不可信，
# 而 kernel launch 是纯 CPU 的活，主频直接决定成绩。这里用 PMU 的
# CPU_CYCLES 计数器除以 CPU 占用时间，与 TSC/cpufreq/MSR 都无关。
FREQ_RESULT="(未测)"
if [ -n "$FM" ]; then
    echo "--- CPU 实际频率 ---"
    taskset -c "$CORE" timeout 2 bash -c 'while :; do :; done' 2>/dev/null
    CUDA_VISIBLE_DEVICES="$GPU" taskset -c "$CORE" "$FM" -i 150 -- \
        taskset -c "$CORE" "$BIN" -t host_launch_latency --launch_count 200000 \
        --dont-pin-threads > "$OUT/freq.txt" 2>&1
    # 取稳态区间：丢掉头 3 个和尾 2 个采样点，再取中位数
    FREQ_RESULT=$(awk '/^ /{print $2}' "$OUT/freq.txt" | head -n -2 | tail -n +4 \
        | sort -n | awk '{v[NR]=$1} END{if(NR)printf "%.2f GHz (稳态中位，%d 个采样点)", v[int((NR+1)/2)], NR; else print "(采样不足)"}')
    echo "      $FREQ_RESULT"
    grep -E '统计范围|CPU 占用|IPC' "$OUT/freq.txt" | sed 's/^/      /'
    grep -q '❌' "$OUT/freq.txt" && { FREQ_RESULT="(PMU 不可用)"; sed -n '/❌/,+2p' "$OUT/freq.txt" | sed 's/^/      /'; }
fi

# ---------- 5. 汇总 ----------
{
echo "=========== 关键指标汇总  ($TAG) ==========="
echo "驱动: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed -n "$((GPU+1))p")"
echo "CPU : $(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1)"
echo "绑核: $CORE   launch_count=$COUNT   perf_event_paranoid=$PARANOID"
echo "★CPU 实测运行频率★: $FREQ_RESULT"
echo
printf "%-34s %10s %10s %10s\n" "指标" "run1" "run2" "run3"
row() { # $1=标题 $2=文件后缀 $3=grep模式 $4=awk取值表达式
    printf "%-34s" "$1"
    for i in $(seq 1 "$REPEAT"); do
        V=$(grep -F "$3" "$OUT/${2}_run${i}.txt" 2>/dev/null | head -1 | awk "$4")
        printf "%10s" "${V:--}"
    done; echo
}
row "cuLaunchKernel 耗时 (ns)"      host_launch_latency  "small_kernel/non_blocking_stream/cuLaunchKernel " "{printf \"%.1f\", \$4/$COUNT}"
row "cuLaunchKernel 指令数 (条)"    host_launch_latency  "non_blocking_stream/cuLaunchKernel/instructions" '{printf "%.1f", $4}'
row "mean_host_latency (ns)"        total_launch_latency "mean_host_latency"   '{print $4}'
row "mean_total_latency (ns)"       total_launch_latency "mean_total_latency"  '{print $4}'
row "device_launch_latency (ns)"    total_launch_latency "device_launch_latency --" '{print $3}'
echo
echo "对照 (本地 Pro5000 48G / driver 615.44 / Xeon w5-3425 / 单 NUMA 裸机):"
echo "  cuLaunchKernel 耗时 ≈ 1360~1390 ns，指令数 ≈ 8334 条"
echo "  mean_host_latency ≈ 1385 ns, mean_total ≈ 3190 ns, device ≈ 1800 ns"
echo "=========================================="
} | tee "$OUT/summary.txt"

echo
echo "=============================================="
echo " 完成。请把整个目录打包回传："
echo "   tar czf result_perfbugs_${TAG}.tar.gz -C $HERE result_perfbugs_${TAG}"
echo "=============================================="
