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

# 逐核扫描 kernel launch 成本。
#
# 背景：kernel launch 的最后一步是往 GPU doorbell 寄存器做 MMIO 写。
# 在多 CCD / 多 NUMA 的大核数 CPU 上，不同核到 IO die（PCIe 所在）的距离不同，
# 这一步的延迟可能因核而异。本脚本量化这个差异。
#
# 用法：
#   ./percore_scan.sh                 # 按步长采样（默认每 8 个逻辑核取一个，再加 0~7）
#   ./percore_scan.sh --all           # 扫描全部允许的核（核多时较慢）
#   ./percore_scan.sh --step 4        # 自定步长
#   STEP/BIN/OUT 也可用环境变量覆盖
set -u
BIN="${BIN:-}"
OUT="${OUT:-./percore_out}"
STEP=8; ALL=0; PASSES=3
while [ $# -gt 0 ]; do
    case "$1" in
        --all) ALL=1;;
        --step) STEP="$2"; shift;;
        --passes) PASSES="$2"; shift;;
        *) echo "未知参数 $1"; exit 1;;
    esac; shift
done
# 准备可执行文件：优先用 BIN 指定的；否则复用 measure.sh 编出来的；都没有就现场编译
if [ -z "$BIN" ]; then
    for cand in ./klb_percore ./klb_pro5000 ./klb_l20 ./klb_cuda129_sm120; do
        [ -x "$cand" ] && BIN="$cand" && break
    done
fi
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    for cand in "$(dirname "$0")/bench/kernel_launch_bench_split.cu" "$(dirname "$0")/kernel_launch_bench_split.cu" "./kernel_launch_bench_split.cu"; do
        [ -f "$cand" ] && SRC="$cand" && break
    done
    SRC="${SRC:-}"
    [ -f "$SRC" ] || SRC=./kernel_launch_bench_split.cu
    [ -f "$SRC" ] || { echo "❌ 既没有现成的可执行文件，也找不到源码 kernel_launch_bench_split.cu"; exit 1; }
    command -v nvcc >/dev/null || { echo "❌ 需要现场编译但没有 nvcc。请先装 CUDA toolkit，或用 BIN=<路径> 指定已编译好的二进制"; exit 1; }
    ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    echo "未找到可执行文件，正在编译（-arch=sm_${ARCH}）…"
    nvcc -O3 -std=c++14 -arch=sm_${ARCH} "$SRC" -o ./klb_percore || { echo "❌ 编译失败"; exit 1; }
    BIN=./klb_percore
fi
echo "使用可执行文件: $BIN"
mkdir -p "$OUT"; rm -f "$OUT"/*

# 允许使用的逻辑核
ALLOWED=$(taskset -pc $$ | sed 's/.*: //')
expand() { echo "$1" | tr ',' '\n' | while read -r r; do
    case "$r" in *-*) seq "${r%-*}" "${r#*-}";; *) echo "$r";; esac; done; }
CPUS=$(expand "$ALLOWED")
NCPU=$(echo "$CPUS" | wc -l)

if [ "$ALL" = 1 ]; then
    SEL="$CPUS"
else
    # 采样：每 STEP 个取一个，另外把前 8 个核全测（0 号核常最忙，值得单独看）
    SEL=$( { echo "$CPUS" | head -8; echo "$CPUS" | awk -v s="$STEP" 'NR%s==1'; } | sort -n -u )
fi

echo "可用逻辑核 $NCPU 个；本次测 $(echo "$SEL" | wc -l) 个"
echo "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
echo
printf "%-6s %-9s %-6s %-9s %s\n" "core" "CCD/die" "NUMA" "min(ns)" "各趟 (ns)"
echo "  ── min 是该核的代表值：抢占只会让结果变慢，不会变快 ──"

for c in $SEL; do
    die=$(cat /sys/devices/system/cpu/cpu$c/topology/die_id 2>/dev/null || echo "-")
    cl=$(cat /sys/devices/system/cpu/cpu$c/topology/cluster_id 2>/dev/null || echo "")
    [ -n "$cl" ] && die="$die/$cl"
    nd=$(ls -d /sys/devices/system/cpu/cpu$c/node* 2>/dev/null | head -1 | sed 's/.*node//' || echo "-")
    vals=""
    for p in $(seq 1 $PASSES); do
        taskset -c "$c" timeout 3 bash -c 'while :; do :; done'
        CUDA_VISIBLE_DEVICES=0 taskset -c "$c" "$BIN" 100000 2000 --rounds 3 --tests 8 \
            > "$OUT/core${c}_p${p}.log" 2>&1
        v=$(grep '^8a ' "$OUT/core${c}_p${p}.log" | sed 's/.*mean=\s*//;s/ ns.*//')
        vals="$vals ${v:-ERR}"
    done
    mn=$(echo $vals | tr ' ' '\n' | grep -v ERR | sort -g | head -1)
    mx=$(echo $vals | tr ' ' '\n' | grep -v ERR | sort -g | tail -1)
    flag=""
    if [ -n "$mn" ] && [ -n "$mx" ]; then
        awk -v a="$mn" -v b="$mx" 'BEGIN{exit !(b/a>1.10)}' && flag="  ⚠ 波动>10%，疑似该核被抢占，建议复测"
    fi
    printf "%-6s %-9s %-6s %-9s %s%s\n" "$c" "$die" "${nd:--}" "${mn:-ERR}" "$vals" "$flag"
    echo "$c,$die,$nd,$mn,$vals" >> "$OUT/_summary.csv"
done
echo done
