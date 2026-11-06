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
#  kernel launch 的「指令数 / 周期数 / 耗时」—— 两点差分法
#
#  为什么要这个脚本：perf_bugs 需要 driver >= 570，在老驱动上跑不了。
#  本脚本只用 PMU + 自带的小探针，【任何驱动版本都能跑】，
#  且不需要 root、不需要改驱动。
#
#  原理：用 freq_monitor 包住探针，分别跑 T1 和 T2 次发射，
#        差分消掉 CUDA 初始化、建 context、预热等固定开销：
#            每次 launch 的指令数 = (指令_T2 - 指令_T1) / (T2 - T1)
#
#  ⚠️ 本脚本的指令数【不能】与 perf_bugs 的指令数直接比较
#     （本脚本走 CUDA runtime API，perf_bugs 直接调 driver API，
#       实测本机差约 792 条）。跨机器比较请都用本脚本。
#
#  用法:  bash run_launch_pmu.sh <标签>
#         GPU=0 REPEAT=3 bash run_launch_pmu.sh my-machine
# =============================================================
set -u
cd "$(dirname "$0")"
HERE=$(pwd)
TAG="${1:-$(hostname | tr -cd 'A-Za-z0-9_.-')}"
GPU="${GPU:-0}"; REPEAT="${REPEAT:-3}"; PREWARM="${PREWARM:-4}"
T1="${T1:-200000}"; T2="${T2:-2000000}"; BATCH="${BATCH:-1024}"
# =1 时从 LD_LIBRARY_PATH 剔除 .../cuda/compat/lib，强制使用宿主机注入的 libcuda。
# ★ 测试老驱动时务必打开 ★：宿主机驱动一旦比容器里的 cuda-compat 旧，
#   toolkit 会把 compat 重新启用，于是实际跑的是 compat 的新 UMD 而不是你要测的那个。
NOCOMPAT="${NOCOMPAT:-0}"
if [ "$NOCOMPAT" = 1 ]; then
    export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
        | grep -v 'cuda/compat' | paste -sd: -)
fi
OUT="$HERE/result_launchpmu_${TAG}"; mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1

echo "=============================================="
echo " launch PMU 差分测量 — 标签: $TAG   $(date -Is)"
echo "=============================================="

# ---------- 编译 ----------
command -v nvcc >/dev/null || { echo "❌ 需要 nvcc 来编译探针"; exit 1; }
command -v cc   >/dev/null || { echo "❌ 需要 cc 来编译 freq_monitor"; exit 1; }
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed -n "$((GPU+1))p" | tr -d '.')
[ -n "$ARCH" ] || { echo "❌ 读不到 compute_cap"; exit 1; }
echo "[1/4] 编译 (sm_$ARCH)"
nvcc -O3 -std=c++14 -arch=sm_${ARCH} "$HERE/bench/launch_pmu_probe.cu" -o "$OUT/probe" || exit 1
cc -O2 -o "$OUT/freq_monitor" "$HERE/bench/freq_monitor.c" || exit 1

# ---------- 环境 ----------
{ echo "### 基本"; hostname; uname -r
  echo; echo "### GPU/驱动"; nvidia-smi --query-gpu=index,name,driver_version --format=csv
  echo; echo "### libcuda"; ldd "$OUT/probe" | grep -i cuda
  LC=$(ldd "$OUT/probe" | sed -n 's/.*libcuda\.so\.1 => \(\S*\).*/\1/p'); [ -n "$LC" ] && { ls -l "$LC"; readlink -f "$LC"; }
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-(未设置)}"
  echo "NOCOMPAT=$NOCOMPAT"
  echo "--- 系统中所有的 libcuda（探针静态链 cudart，ldd 看不到，故列目录）---"
  for dd in /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/local/cuda/compat/lib \
            /usr/local/cuda/compat/lib.real /usr/local/nvidia/lib64; do
      ls -l "$dd"/libcuda.so.* 2>/dev/null; done | sort -u

  echo; echo "### CPU"; lscpu 2>/dev/null | grep -Ei 'model name|^CPU\(s\)|Socket|NUMA|MHz'
  echo; echo "### 虚拟化"; V=$(systemd-detect-virt 2>/dev/null); echo "systemd-detect-virt: ${V:-(不可用)}"
  grep -qm1 ' hypervisor' /proc/cpuinfo && echo "hypervisor flag: 有(虚拟机)" || echo "hypervisor flag: 无(裸金属)"
  echo; echo "### perf_event_paranoid"; cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null
  echo "uid=$(id -u)"; grep -E '^Cap(Prm|Eff)' /proc/self/status 2>/dev/null
} > "$OUT/env.txt" 2>&1
grep -E 'Model name|^[0-9]+,' "$OUT/env.txt" | head -3 | sed 's/^/      /'

# ---------- 选核 + 预热 ----------
CORE=$(bash "$HERE/pick_core.sh" "$GPU"); [ -z "$CORE" ] && CORE=0
echo "[2/4] 绑核 $CORE；预热 ${PREWARM}s"
taskset -c "$CORE" timeout "$PREWARM" bash -c 'while :; do :; done' 2>/dev/null

# ---------- 测量 ----------
run_at() {   # $1=T $2=序号 -> 输出 "指令 周期 墙上秒"
    # 每轮都重新预热：两个 T 的运行时长不同，若核的频率状态不一致，差分会失真
    taskset -c "$CORE" timeout 2 bash -c 'while :; do :; done' 2>/dev/null
    CUDA_VISIBLE_DEVICES="$GPU" taskset -c "$CORE" "$OUT/freq_monitor" -i 1000 -- \
        taskset -c "$CORE" "$OUT/probe" "$1" --batch "$BATCH" > "$OUT/T$1_run$2.txt" 2>&1
    awk '/总指令数/{i=$3} /总周期数/{c=$3} /^墙上时间/{w=$3} END{print i, c, w}' "$OUT/T$1_run$2.txt"
}
echo "[3/4] 测量：T1=$T1 与 T2=$T2，各 $REPEAT 轮（batch=$BATCH）"
for T in "$T1" "$T2"; do
    : > "$OUT/agg_$T.txt"
    for r in $(seq 1 "$REPEAT"); do
        R=$(run_at "$T" "$r")
        echo "$R" >> "$OUT/agg_$T.txt"
        echo "      T=$T run$r: 指令=$(echo $R|cut -d' ' -f1) 周期=$(echo $R|cut -d' ' -f2) 墙钟=$(echo $R|cut -d' ' -f3)s"
    done
done

# ---------- 实际加载的 UMD（权威，探针自报）----------
{ echo; echo "### 探针自报的实际加载项（权威）"
  grep -hE 'driver_API_version|已加载 libcuda' "$OUT"/T*_run*.txt 2>/dev/null | sort -u
} >> "$OUT/env.txt"
echo "      实际加载的 UMD:"
grep -hE 'driver_API_version|已加载 libcuda' "$OUT"/T*_run*.txt 2>/dev/null | sort -u | sed 's/^/        /'

# ---------- 差分 ----------
echo "[4/4] 差分"
python3 - "$OUT" "$T1" "$T2" <<'PY' | tee "$OUT/summary.txt"
import sys, statistics as st
out,T1,T2 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
def med(T):
    rows=[l.split() for l in open(f"{out}/agg_{T}.txt") if l.strip()]
    rows=[(float(a),float(b),float(c)) for a,b,c in rows]
    return tuple(st.median([r[k] for r in rows]) for k in range(3))
i1,c1,w1 = med(T1); i2,c2,w2 = med(T2)
dN = T2-T1; di=(i2-i1)/dN; dc=(c2-c1)/dN; dw=(w2-w1)/dN*1e9
print("=========== 每次 kernel launch（两点差分）===========")
print(f"  T1={T1}  指令={i1:.0f}  周期={c1:.0f}  墙钟={w1:.3f}s")
print(f"  T2={T2}  指令={i2:.0f}  周期={c2:.0f}  墙钟={w2:.3f}s")
print()
print(f"  ★ 指令数   : {di:9.1f} 条")
print(f"  ★ 周期数   : {dc:9.1f}")
print(f"  ★ 耗时     : {dw:9.1f} ns")
print(f"  ★ IPC      : {di/dc:9.3f}")
f=dc/dw
print(f"  ★ 隐含频率 : {f:9.3f} GHz   （= 周期数 / 耗时，就是发射循环期间的实际频率）")
# 护栏：隐含频率是三个量算出来的，任何一个解析/测量出错它都会离谱。
# 真实 CPU 不可能在这个区间外，所以它是整组数据最灵敏的自检指标。
if not (0.5 <= f <= 6.0):
    print()
    print(f"  ❌ 隐含频率 {f:.1f} GHz 不合理（正常应在 0.5~6 GHz）。本组数据不可用。")
    print("     多半是某一项解析出错或测量被打断，请把整个目录回传，不要采用上面的数值。")
elif abs(di/dc - 1) > 0 and (di/dc < 0.2 or di/dc > 6):
    print()
    print(f"  ⚠️ IPC {di/dc:.2f} 超出常见范围（0.2~6），请连同原始文件一起回传复核。")
print()
print("  ⚠️ 本指令数不可与 perf_bugs 的指令数直接比较（API 层数不同，本机实测差约 792 条）")
print("=====================================================")
PY
echo
echo "完成。请打包回传: tar czf result_launchpmu_${TAG}.tar.gz -C $HERE result_launchpmu_${TAG}"
