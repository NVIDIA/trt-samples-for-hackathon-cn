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

# 选一个用于绑定的 CPU 核，并打印选择理由。
#
# 规则（按优先级）：
#   1. 若能确定 GPU 所在的 NUMA 节点 → 只在该节点的核里选
#      （双路机器上跨 NUMA 会多付 11~13% 的 launch 成本，实测见 progress.md §7.4.28）
#   2. 在候选集里避开 0 号核（通常在处理中断）
#   3. 取候选集靠后的一个核（避开常被系统占用的低编号核）
#
# 用法：  CORE=$(bash pick_core.sh [GPU索引])        # 只输出核号
#         bash pick_core.sh --explain [GPU索引]      # 输出核号 + 理由到 stderr
set -u
EXPLAIN=0
[ "${1:-}" = "--explain" ] && { EXPLAIN=1; shift; }
GPUIDX="${1:-0}"
note() { [ "$EXPLAIN" = 1 ] && echo "  [pick_core] $*" >&2; }

expand() { tr ',' '\n' | while read -r r; do
    case "$r" in *-*) seq "${r%-*}" "${r#*-}";; '') :;; *) echo "$r";; esac; done; }

ALLOWED=$(taskset -pc $$ 2>/dev/null | sed 's/.*: //')
ALL=$(printf '%s\n' "$ALLOWED" | expand | sort -n -u)
[ -z "$ALL" ] && { echo 0; exit 0; }

# ---- 找 GPU 的 NUMA 节点与本地核 ----
BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | sed -n "$((GPUIDX+1))p" \
      | tr 'A-Z' 'a-z' | tr -d '[:space:]' | sed 's/^0*//')
LOCAL=""; NODE=""
if [ -n "$BUS" ]; then
    for d in /sys/bus/pci/devices/*"$BUS"*; do
        [ -r "$d/numa_node" ] || continue
        NODE=$(cat "$d/numa_node")
        [ -r "$d/local_cpulist" ] && LOCAL=$(cat "$d/local_cpulist")
        break
    done
fi

CAND="$ALL"
if [ -n "$NODE" ] && [ "$NODE" != "-1" ] && [ -n "$LOCAL" ]; then
    LC=$(printf '%s\n' "$LOCAL" | expand | sort -n -u)
    INTER=$(comm -12 <(echo "$ALL") <(echo "$LC") 2>/dev/null)
    if [ -n "$INTER" ]; then
        CAND="$INTER"
        note "GPU$GPUIDX 在 NUMA node $NODE，本地核 $LOCAL"
        note "与进程可用核取交集后候选 $(echo "$CAND" | wc -l) 个"
    else
        note "⚠️ GPU 本地核与进程可用核无交集，退回使用全部可用核"
    fi
else
    note "⚠️ 无法确定 GPU 的 NUMA 节点（numa_node=${NODE:-未知}），退回使用全部可用核"
    note "   单 NUMA 机器上这是正常的"
fi

# 避开 0 号核；取候选集靠后 1/4 处的一个核（既避开低编号，又不取最边缘）
C2=$(echo "$CAND" | grep -v '^0$'); [ -z "$C2" ] && C2="$CAND"
N=$(echo "$C2" | wc -l)
PICK=$(echo "$C2" | sed -n "$(( N - N/4 ))p")
[ -z "$PICK" ] && PICK=$(echo "$C2" | tail -1)
note "选中核 $PICK"
echo "$PICK"
