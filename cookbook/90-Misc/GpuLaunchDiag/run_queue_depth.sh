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
#  发射队列深度扫描 —— 回答「这个驱动在深队列下是否会劣化」
#
#  背景：580 分支的发射退化只在【队列深】的时候显现。
#  每 256 次同步一次时它只比 615 慢 ~10%，而连发 10 万次再同步时慢 ~45%。
#  真实推理每次前向连发几百上千个 kernel 且中间不同步，正好落在最糟的工作点。
#
#  本脚本做两件事：
#    [A] 8a 基准  —— 10 万次背靠背 + 末尾一次同步（与历史数据同口径）
#    [B] BATCH 扫描 —— 用 PMU 差分测出每个队列深度下的 指令数/周期/耗时
#       两者应当对上：BATCH=100000 的结果 ≈ 8a
#
#  用法:  bash run_queue_depth.sh <标签>
#         NOCOMPAT=1 bash run_queue_depth.sh <标签>      # 测老驱动时务必加
#         BATCHES="256 8192 100000" bash run_queue_depth.sh <标签>
# =============================================================
set -u
cd "$(dirname "$0")"
HERE=$(pwd)
TAG="${1:-$(hostname | tr -cd 'A-Za-z0-9_.-')}"
GPU="${GPU:-0}"; REPEAT="${REPEAT:-3}"; PREWARM="${PREWARM:-4}"
T1="${T1:-200000}"; T2="${T2:-2000000}"
BATCHES="${BATCHES:-64 256 1024 8192 65536 100000}"
ITERS="${ITERS:-100000}"          # 8a 的发射次数，与历史数据保持一致
NOCOMPAT="${NOCOMPAT:-0}"
if [ "$NOCOMPAT" = 1 ]; then
    export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
        | grep -v 'cuda/compat' | paste -sd: -)
fi
OUT="$HERE/result_queuedepth_${TAG}"; mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1

echo "=============================================="
echo " 队列深度扫描 — 标签: $TAG   $(date -Is)"
echo "=============================================="

command -v nvcc >/dev/null || { echo "❌ 需要 nvcc"; exit 1; }
command -v cc   >/dev/null || { echo "❌ 需要 cc"; exit 1; }
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed -n "$((GPU+1))p" | tr -d '.')
[ -n "$ARCH" ] || { echo "❌ 读不到 compute_cap"; exit 1; }
echo "[1/4] 编译 (sm_$ARCH)"
nvcc -O3 -std=c++14 -arch=sm_${ARCH} "$HERE/bench/launch_pmu_probe.cu"        -o "$OUT/probe" || exit 1
nvcc -O3 -std=c++14 -arch=sm_${ARCH} "$HERE/bench/kernel_launch_bench_split.cu" -o "$OUT/klb" || exit 1
cc -O2 -o "$OUT/freq_monitor" "$HERE/bench/freq_monitor.c" || exit 1

# ---------- 环境 ----------
{ echo "### 基本"; hostname; uname -r; date -Is
  echo; echo "### GPU/驱动(KMD)"; nvidia-smi --query-gpu=index,name,driver_version --format=csv
  echo; echo "### NOCOMPAT=$NOCOMPAT"
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-(未设置)}"
  echo "--- 系统中所有的 libcuda ---"
  for dd in /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/local/cuda/compat/lib \
            /usr/local/cuda/compat/lib.real /usr/local/nvidia/lib64; do
      ls -l "$dd"/libcuda.so.* 2>/dev/null; done | sort -u
  echo; echo "### CPU"; lscpu 2>/dev/null | grep -Ei 'model name|^CPU\(s\)|Socket|NUMA|MHz'
} > "$OUT/env.txt" 2>&1

# ★ 实际加载的 UMD —— 唯一权威来源 ★
# 探针静态链 cudart，libcuda 是运行时 dlopen 的，ldd 看不到。
# 宿主机驱动一旦比容器里的 cuda-compat 旧，compat 会被启用，
# 于是你以为在测老驱动，实际跑的是 compat 的新 UMD。必须核对这一项。
UMD=$(CUDA_VISIBLE_DEVICES="$GPU" "$OUT/probe" 1000 2>&1 | grep -E 'driver_API_version|已加载 libcuda')
{ echo; echo "### ★实际加载的 UMD（权威）★"; echo "$UMD"; } >> "$OUT/env.txt"
echo "[2/4] 实际加载的 UMD:"; echo "$UMD" | sed 's/^/      /'
KMD=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n "$((GPU+1))p")
echo "      KMD(nvidia-smi) = $KMD"
echo "$UMD" | grep -q "libcuda.so.$KMD" || {
    echo "      ⚠️⚠️ 实际加载的 UMD 与 KMD 版本不一致！"
    echo "         多半是 cuda-compat 介入了。若要测 KMD 本身，请加 NOCOMPAT=1 重跑。"
}

CORE=$(bash "$HERE/pick_core.sh" "$GPU"); [ -z "$CORE" ] && CORE=0
echo "[3/4] 绑核 $CORE"
warm() { taskset -c "$CORE" timeout "${1:-2}" bash -c 'while :; do :; done' 2>/dev/null; }
warm "$PREWARM"

# ---------- [A] 8a ----------
echo "--- [A] 8a：${ITERS} 次背靠背 + 末尾一次同步 ---"
: > "$OUT/a8.txt"
for i in $(seq 1 "$REPEAT"); do
    warm 2
    CUDA_VISIBLE_DEVICES="$GPU" taskset -c "$CORE" "$OUT/klb" "$ITERS" 2000 --rounds 3 --tests 8 \
        > "$OUT/a8_run$i.log" 2>&1
    V=$(sed -n 's/^8a.*p50= *\([0-9.][0-9.]*\).*/\1/p' "$OUT/a8_run$i.log" | head -1)
    echo "${V:-NA}" >> "$OUT/a8.txt"
    echo "      run$i: 8a p50 = ${V:-解析失败} ns"
done

# ---------- [B] BATCH 扫描 ----------
echo "--- [B] BATCH 扫描（PMU 两点差分）---"
meas() {  # $1=T $2=BATCH -> "指令 周期 墙上秒"
    warm 2
    CUDA_VISIBLE_DEVICES="$GPU" taskset -c "$CORE" "$OUT/freq_monitor" -i 1000 -- \
        taskset -c "$CORE" "$OUT/probe" "$1" --batch "$2" > "$OUT/B$2_T$1.txt" 2>&1
    awk '/总指令数/{i=$3} /总周期数/{c=$3} /^墙上时间/{w=$3} END{print i,c,w}' "$OUT/B$2_T$1.txt"
}
: > "$OUT/sweep.txt"
printf "      %-9s%10s%10s%9s%8s%8s\n" BATCH instr cycles ns IPC GHz
for B in $BATCHES; do
    R1=$(meas "$T1" "$B"); R2=$(meas "$T2" "$B")
    L=$(echo "$R1 $R2" | awk -v b="$B" -v dn="$((T2-T1))" \
      '{di=($4-$1)/dn; dc=($5-$2)/dn; dw=($6-$3)/dn*1e9; printf "%s %.1f %.1f %.1f %.3f %.3f", b,di,dc,dw,di/dc,dc/dw}')
    echo "$L" >> "$OUT/sweep.txt"
    echo "$L" | awk '{printf "      %-9s%10.1f%10.1f%9.1f%8.3f%8.3f\n",$1,$2,$3,$4,$5,$6}'
done

# ---------- 汇总 ----------
echo "[4/4] 汇总"
python3 - "$OUT" "$ITERS" <<'PY' | tee "$OUT/summary.txt"
import sys, statistics as st
out, iters = sys.argv[1], sys.argv[2]
a8=[float(x) for x in open(f"{out}/a8.txt") if x.strip() and x.strip()!="NA"]
sw=[l.split() for l in open(f"{out}/sweep.txt") if l.strip()]
print("=============== 队列深度扫描汇总 ===============")
if a8: print(f"[A] 8a ({iters} 次背靠背 + 1 次同步)  p50 中位 = {st.median(a8):.1f} ns   ({len(a8)} 轮: {', '.join(f'{v:.1f}' for v in a8)})")
print()
print(f"[B] {'BATCH':>8}{'instr/次':>11}{'cycles/次':>11}{'ns/次':>10}{'IPC':>8}{'GHz':>9}")
rows=[]
for b,i,c,w,ipc,f in sw:
    rows.append((int(b),float(i),float(c),float(w),float(ipc),float(f)))
    print(f"    {b:>8}{float(i):11.1f}{float(c):11.1f}{float(w):10.1f}{float(ipc):8.3f}{float(f):9.3f}")
if len(rows)>=2:
    sh,dp = rows[0], rows[-1]
    print()
    print(f"  浅队列(BATCH={sh[0]}) -> 深队列(BATCH={dp[0]}):")
    print(f"    耗时   {sh[3]:.1f} -> {dp[3]:.1f} ns   ({dp[3]/sh[3]-1:+.1%})")
    print(f"    指令数 {sh[1]:.1f} -> {dp[1]:.1f}      ({dp[1]/sh[1]-1:+.1%})")
    print()
    if a8:
        d=abs(dp[3]-st.median(a8))/st.median(a8)
        print(f"  交叉验证: 最深 BATCH 的耗时 {dp[3]:.1f} ns  vs  8a 的 {st.median(a8):.1f} ns  (差 {d:.1%})")
        print("  " + ("✅ 两套工具一致" if d<0.07 else "⚠️ 两者差异偏大，请检查"))
    print()
    print("  注意：深队列下 CPU 有相当一部分时间在等队列腾空，")
    print("        此时『GHz』列被等待时间稀释，不再代表真实主频；该区间请只看 ns/次。")
    print()
    print("  判读：深队列显著劣化 => 该驱动的背压/提交路径有问题（580 分支的典型特征）；")
    print("        基本持平     => 该驱动在深队列下正常（615 分支的表现）。")
print("================================================")
PY
echo
echo "完成。请打包回传: tar czf result_queuedepth_${TAG}.tar.gz -C $HERE result_queuedepth_${TAG}"
