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

# 单次 kernel 发射往返的「GPU 时间轴 / 主机侧」拆分
# 用法：./run_roundtrip.sh [平台标签]
set -u
TAG="${1:-$(hostname)}"
OUT="${OUT:-./roundtrip_out}"; mkdir -p "$OUT"; rm -f "$OUT"/*
BIN="${BIN:-}"; CORE="${CORE:-}"; PASSES="${PASSES:-4}"

if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    for cand in "$(dirname "$0")/bench/launch_roundtrip.cu" "$(dirname "$0")/launch_roundtrip.cu" "./launch_roundtrip.cu"; do
        [ -f "$cand" ] && SRC="$cand" && break
    done
    SRC="${SRC:-}"
    [ -f "$SRC" ] || { echo "❌ 找不到源码 launch_roundtrip.cu"; exit 1; }
    command -v nvcc >/dev/null || { echo "❌ 需编译但无 nvcc；可用 BIN=<路径> 指定"; exit 1; }
    ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    echo "正在编译（-arch=sm_${ARCH}）…"
    nvcc -O3 -std=c++14 -arch=sm_${ARCH} "$SRC" -o ./launch_roundtrip || exit 1
    BIN=./launch_roundtrip
fi
# 优先选 GPU 所在 NUMA 节点上的核（双路机上跨 NUMA 多付 11~13%）
[ -z "$CORE" ] && CORE=$(bash "$(dirname "$0")/pick_core.sh" 0)

echo "平台=$TAG  可执行文件=$BIN  绑核=$CORE"
nvidia-smi --query-gpu=index,name,driver_version,persistence_mode --format=csv | head -2
echo

# 空闲后的第一个 CUDA 进程有冷启动惩罚，先空跑一次丢弃
echo "（预热：空跑一次并丢弃）"
taskset -c "$CORE" timeout 4 bash -c 'while :; do :; done'
CUDA_VISIBLE_DEVICES=0 taskset -c "$CORE" "$BIN" 10000 2000 >/dev/null 2>&1

NOISY=0
NLIST="1 16 64 256 1024"
printf "\n%-5s %-10s %-10s %-10s %-7s" "趟次" "wall" "event" "host" "主机侧"
for n in $NLIST; do printf " %9s" "issue@$n"; done; printf "\n"
printf "%-5s %-10s %-10s %-10s %-7s" "" "p50" "p50" "p50" "占比"
for n in $NLIST; do printf " %9s" "(ns)"; done; printf "\n"
for p in $(seq 1 $PASSES); do
    taskset -c "$CORE" timeout 4 bash -c 'while :; do :; done'
    CUDA_VISIBLE_DEVICES=0 taskset -c "$CORE" "$BIN" 10000 2000 > "$OUT/${TAG}_p${p}.log" 2>&1
    p50of() { sed -n "s/^$1 .*p50= *\\([0-9.]*\\).*/\\1/p" "$OUT/${TAG}_p${p}.log"; }
    w=$(p50of wall); e=$(p50of event); h=$(p50of host)
    pct=$(awk -v a="$h" -v b="$w" 'BEGIN{if(b>0)printf "%.0f%%", 100*a/b; else print "-"}')
    printf "%-5s %-10s %-10s %-10s %-7s" "$p" "$w" "$e" "$h" "$pct"
    row="$p,$w,$e,$h"
    for n in $NLIST; do
        v=$(awk -v n="$n" '$1==n{print $2}' "$OUT/${TAG}_p${p}.log")
        printf " %9s" "${v:--}"; row="$row,$v"
    done
    printf "\n"
    echo "$row" >> "$OUT/_summary.csv"
done

# 稳定性自检：host 的 p50 在各趟之间波动是否超过 10%
echo "   （CSV 表头：趟次,wall,event,host,$(echo $NLIST | tr ' ' ',' | sed 's/\([0-9]*\)/issue@\1/g')）"
mn=$(cut -d, -f4 "$OUT/_summary.csv" | sort -g | head -1)
mx=$(cut -d, -f4 "$OUT/_summary.csv" | sort -g | tail -1)
awk -v a="$mn" -v b="$mx" 'BEGIN{exit !(b/a>1.10)}' && NOISY=1
echo
if [ "$NOISY" = 1 ]; then
    echo "⚠️  host 在各趟之间波动超过 10%（$mn ~ $mx），说明测量期间机器上有别的负载。"
    echo "    请在机器空闲时重跑。"
else
    echo "✅ 各趟稳定（波动 <10%），数据可用。取 host 的最小值作为代表：$mn ns"
fi
echo
echo "── 纯 CPU 发射成本的 N 扫描（最后一趟）──"
sed -n '/N  *issue/,/^$/p' "$OUT/${TAG}_p${PASSES}.log"
echo "若 issue 一列随 N 增大而明显上升 → 已撞到队列背压，该列不再代表 CPU 成本"
echo
echo "原始输出在 $OUT/"
echo done
