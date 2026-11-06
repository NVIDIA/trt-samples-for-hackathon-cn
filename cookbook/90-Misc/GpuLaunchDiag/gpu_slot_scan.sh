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

# 排查：4 张卡的主机上，可见设备数量 / 具体是哪张卡，会不会影响 kernel launch 成本？
#
# 5 个配置，每个跑 3 趟取 min：
#   A  CUDA_VISIBLE_DEVICES=0..3  —— 只暴露 1 张卡，分别试 4 张（看 PCIe 槽位差异）
#   B  不设 CUDA_VISIBLE_DEVICES  —— 4 张卡全可见，用 device 0（看设备数量的影响）
# A 的第一行 与 B 对比 = 同一张卡、只差"进程能看到几张卡"
set -u
BIN="${BIN:-}"
OUT="${OUT:-./gpuslot_out}"; mkdir -p "$OUT"; rm -f "$OUT"/*
CORE="${CORE:-}"
PASSES="${PASSES:-3}"

if [ -z "$BIN" ]; then
    for c in ./klb_percore ./klb_pro5000 ./klb_gpuslot; do [ -x "$c" ] && BIN="$c" && break; done
fi
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    for cand in "$(dirname "$0")/bench/kernel_launch_bench_split.cu" "$(dirname "$0")/kernel_launch_bench_split.cu" "./kernel_launch_bench_split.cu"; do
        [ -f "$cand" ] && SRC="$cand" && break
    done
    SRC="${SRC:-}"
    [ -f "$SRC" ] || { echo "❌ 找不到源码"; exit 1; }
    command -v nvcc >/dev/null || { echo "❌ 需编译但无 nvcc；可用 BIN=<路径> 指定"; exit 1; }
    ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    echo "正在编译（-arch=sm_${ARCH}）…"
    nvcc -O3 -std=c++14 -arch=sm_${ARCH} "$SRC" -o ./klb_gpuslot || exit 1
    BIN=./klb_gpuslot
fi
# 与 measure_launch.sh 保持一致：优先选 GPU 所在 NUMA 节点上的核
[ -z "$CORE" ] && CORE=$(bash "$(dirname "$0")/pick_core.sh" 0)
echo "可执行文件 $BIN  绑核 $CORE"
nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id --format=csv
echo
NOISY=0
printf "%-24s %-9s %s\n" "配置(only_gpuN=只暴露该卡)" "min(ns)" "各趟 (ns)"
echo "  ── 看 min 列：抢占只会让结果变慢，不会变快 ──"

run() { # $1=标签  $2=CUDA_VISIBLE_DEVICES（空=不设）
    vals=""
    for p in $(seq 1 $PASSES); do
        taskset -c "$CORE" timeout 3 bash -c 'while :; do :; done'
        if [ -z "$2" ]; then
            env -u CUDA_VISIBLE_DEVICES taskset -c "$CORE" "$BIN" 100000 2000 --rounds 3 --tests 8 \
                > "$OUT/$1_p${p}.log" 2>&1
        else
            CUDA_VISIBLE_DEVICES="$2" taskset -c "$CORE" "$BIN" 100000 2000 --rounds 3 --tests 8 \
                > "$OUT/$1_p${p}.log" 2>&1
        fi
        v=$(grep '^8a ' "$OUT/$1_p${p}.log" | sed 's/.*mean=\s*//;s/ ns.*//')
        vals="$vals ${v:-ERR}"
    done
    mn=$(echo $vals | tr ' ' '\n' | grep -v ERR | sort -g | head -1)
    mx=$(echo $vals | tr ' ' '\n' | grep -v ERR | sort -g | tail -1)
    flag=""
    if [ -n "$mn" ] && [ -n "$mx" ]; then
        awk -v a="$mn" -v b="$mx" 'BEGIN{exit !(b/a>1.10)}' && flag="  ⚠ 波动>10%" && NOISY=1
    fi
    printf "%-28s %-9s %s%s\n" "$1" "${mn:-ERR}" "$vals" "$flag"
    echo "$1,$mn,$vals" >> "$OUT/_summary.csv"
}

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

# 先空跑一次丢弃：空闲后的第一个 CUDA 进程会有冷启动惩罚（见 progress.md §4.7），
# 否则排在第一位的配置会被系统性地高估。
echo "（预热：先空跑一次并丢弃）"
taskset -c "$CORE" timeout 3 bash -c 'while :; do :; done'
CUDA_VISIBLE_DEVICES=0 taskset -c "$CORE" "$BIN" 100000 2000 --rounds 3 --tests 8 >/dev/null 2>&1
echo

for i in $(seq 0 $((NGPU-1))); do run "only_gpu${i}" "$i"; done
run "all_${NGPU}gpus_dev0" ""
echo
if [ "$NOISY" = 1 ]; then
    echo "⚠️  有配置三趟之间波动超过 10%，说明测量期间机器上有别的负载。"
    echo "    这种数据不能用来比较配置之间的差异，请在机器空闲时重跑。"
else
    echo "✅ 各配置三趟内部都稳定（波动 <10%），数据可用。"
fi
echo done
