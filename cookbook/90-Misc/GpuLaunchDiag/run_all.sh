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

# ============================================================
#  Pro5000 / L20 发射开销测试套件 —— 一次跑完
#
#  用法：  ./run_all.sh <标签>
#    例：  ./run_all.sh baremetal-xeon-w5-3425
#          ./run_all.sh kvm-epyc9k84
#
#  标签会进所有文件名和最终 tar 包名，**不同机器/配置必须用不同标签**，
#  否则结果会互相覆盖（两台机器可能挂载同一个目录）。
#
#  全程只读，不修改任何系统配置。有免密 sudo 会多设一次 governor，没有也能跑。
# ============================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
TAG="${1:-}"
[ -z "$TAG" ] && { echo "用法: $0 <标签>   例: $0 kvm-epyc9k84"; exit 1; }
TAG=$(printf '%s' "$TAG" | tr -c 'A-Za-z0-9._-' '-')

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$HERE/out/${TAG}_${STAMP}"
mkdir -p "$OUT"
LOG="$OUT/00_run_all.log"

say() { echo "$@" | tee -a "$LOG"; }
step() { echo | tee -a "$LOG"; say "############ $* ############"; }

# 跳过开关（默认全跑）
SKIP_PERCORE="${SKIP_PERCORE:-0}" # 逐核扫描
SKIP_SLOT="${SKIP_SLOT:-0}"       # 槽位/设备数扫描

say "标签: $TAG"
say "输出: $OUT"
say "开始: $(date -Is)"

# ---------- 1. 环境采集（最先做，即使后面失败也有这份）----------
step "1/6 环境采集"
bash "$HERE/collect_env.sh" > "$OUT/01_env.txt" 2>&1
say "→ 01_env.txt"
grep -E "Model name|current:|CPU hypervisor flag|scaling_driver|^0, " "$OUT/01_env.txt" 2>/dev/null | head -6 | tee -a "$LOG"

# ---------- 2. 编译 ----------
step "2/6 编译发射基准"
if ! command -v nvcc >/dev/null; then
    say "❌ 没有 nvcc，发射基准无法编译。请在带 CUDA toolkit 的环境中运行。"
else
    ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    say "探测到 arch=sm_${ARCH}"
    ( cd "$OUT" && nvcc -O3 -std=c++14 -arch=sm_${ARCH} "$HERE/bench/kernel_launch_bench_split.cu" -o klb 2>&1 | tail -3 ) | tee -a "$LOG"
    ( cd "$OUT" && nvcc -O3 -std=c++14 -arch=sm_${ARCH} "$HERE/bench/launch_roundtrip.cu" -o lrt 2>&1 | tail -3 ) | tee -a "$LOG"
    ( cd "$OUT" && nvcc -O3 -std=c++14 -arch=sm_${ARCH} "$HERE/bench/mmio_probe.cu" -o mmio_probe 2>&1 | tail -3 ) | tee -a "$LOG"
    [ -x "$OUT/klb" ] && say "✓ klb" || say "❌ klb 编译失败"
    [ -x "$OUT/lrt" ] && say "✓ lrt" || say "❌ lrt 编译失败"
    [ -x "$OUT/mmio_probe" ] && say "✓ mmio_probe（KVM MMIO 探针，需配合宿主机脚本，见 README 第 5 节）" \
        || say "❌ mmio_probe 编译失败"
fi

# ---------- 3. 完整发射基准（测项 1~10）----------
step "3/6 完整发射基准（约 3 分钟）"
if [ -x "$OUT/klb" ]; then
    ( cd "$OUT" && BIN=./klb bash "$HERE/measure_launch.sh" "$TAG" ) 2>&1 | tee -a "$LOG"
else
    say "跳过（无可执行文件）"
fi

# ---------- 4. 往返拆分 + 纯 CPU 发射成本 ----------
step "4/6 往返拆分 + 纯 CPU 发射成本（约 2 分钟）"
if [ -x "$OUT/lrt" ]; then
    ( cd "$OUT" && BIN=./lrt OUT="$OUT/roundtrip" bash "$HERE/run_roundtrip.sh" "$TAG" ) 2>&1 | tee -a "$LOG"
else
    say "跳过（无可执行文件）"
fi

# ---------- 5. 逐核扫描 ----------
step "5/6 逐核扫描（约 8 分钟）"
if [ "$SKIP_PERCORE" = "1" ]; then say "按要求跳过（SKIP_PERCORE=1）"
elif [ -x "$OUT/klb" ]; then
    ( cd "$OUT" && BIN=./klb OUT="$OUT/percore" bash "$HERE/percore_scan.sh" ) 2>&1 | tee -a "$LOG"
else say "跳过（无可执行文件）"; fi

# ---------- 6. 槽位 / 可见设备数扫描 ----------
step "6/6 槽位与可见设备数扫描（约 5 分钟）"
if [ "$SKIP_SLOT" = "1" ]; then say "按要求跳过（SKIP_SLOT=1）"
elif [ -x "$OUT/klb" ]; then
    ( cd "$OUT" && BIN=./klb OUT="$OUT/gpuslot" bash "$HERE/gpu_slot_scan.sh" ) 2>&1 | tee -a "$LOG"
else say "跳过（无可执行文件）"; fi

# ---------- 附加：真实模型 QPS（T8，压测器需自备）----------
# 压测器（原 run_qps.sh / bench/qps_bench.cpp）源自客户项目，已从本仓库移除，所以这里不再
# 尝试运行它，只写一份状态文件说明情况 —— 让回传包里"这项为什么没有数据"有据可查，
# 而不是悄无声息地缺一块。方法与四条 CUDA Graph 前提见 诊断手册.md 第十一节。
step "附加 真实模型 QPS（T8，压测器需自备）"
QPS_STATUS="$OUT/08_qps_status.txt"
{
  echo "检查时间: $(date -Is)"
  echo "结果: 未运行 —— 压测器不随本仓库提供"
  echo "原因: 原压测器源自客户项目，不适合公开发布，已移除"
  echo "如何补测: 自行实现压测器；要素与 CUDA Graph 的四条前提见 诊断手册.md 第十一节"
} > "$QPS_STATUS" 2>&1
say "未运行 —— 压测器不随本仓库提供（见 诊断手册.md 第十一节）"
say "    T0~T7 的发射类测试不受影响，结果已保存。"
say "→ 08_qps_status.txt"

# ---------- 打包 ----------
step "打包"
rm -f "$OUT"/klb "$OUT"/lrt 2>/dev/null          # 这两个不入包（体积大且可重编）
# mmio_probe 保留在 $OUT 下，供 KVM 场景下配合宿主机脚本使用（不入 tar 包）
rm -rf "$OUT/qps_build/CMakeFiles" 2>/dev/null
TAR="$HERE/out/testkit_${TAG}_${STAMP}.tar.gz"
# mmio_probe 留在磁盘上供 KVM 场景使用，但不入回传包（回传包只装数据）
tar czf "$TAR" --exclude='mmio_probe' -C "$HERE/out" "$(basename "$OUT")" 2>/dev/null
say "结束: $(date -Is)"
say ""
# 编译失败时上面每一步都会"跳过"，但旧版仍旧打印「全部完成」，
# 容易让人以为拿到了数据。这里显式区分两种结局。
if [ ! -s "$OUT/${TAG}_run1.csv" ]; then
    say "================================================================"
    say "  ❌ 没有产出任何测量数据 —— 发射基准未能编译或未能运行。"
    say ""
    if ! command -v nvcc >/dev/null; then
        say "  原因：本机没有 nvcc。T1/T2/T6 都需要现场编译。"
        say "  可选做法："
        say "    - 在带 CUDA toolkit 的环境（如 NGC 容器）里跑本脚本；或"
        # T5 是唯一不需要 toolkit 的测项，但它依赖 perf_bugs，而公开检出里没有那个二进制。
        # 没有时就别推荐它了，否则是把人指向一条走不通的路。
        if [ -x "$HERE/bin/perf_bugs" ]; then
            say "    - 只跑不需要 toolkit 的 T5：bash $HERE/run_perf_bugs.sh <标签>"
            say "      （perf_bugs 只依赖 libcuda.so.1，不需要任何 toolkit）"
        else
            say "    - 本机没有 bin/perf_bugs（NVIDIA 内部工具，此处不公开，请与作者联系），"
            say "      因此唯一不需要 toolkit 的 T5 也跑不了。装上 nvcc 是这里的必经之路。"
        fi
    else
        say "  请查看上面 2/6 段的编译报错。"
    fi
    say ""
    say "  仍然产出了环境采集，可单独回传："
    say "    $OUT/01_env.txt"
    say "================================================================"
    exit 1
fi
say "================================================================"
say "  全部完成。请回传这一个文件："
say "  $TAR"
say "  大小: $(du -h "$TAR" 2>/dev/null | cut -f1)"
say ""
say "  【若本机是 KVM 虚拟机】还可做一项宿主机侧的验证："
say "    guest 侧:  $OUT/mmio_probe 100000"
say "    host  侧:  sudo bash <套件>/kvm_mmio_host.sh --n 100000"
say "    详见 README 第 5 节"
say "================================================================"
