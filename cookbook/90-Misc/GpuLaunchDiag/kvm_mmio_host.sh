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
#  KVM MMIO exit 计数 —— ★在宿主机上运行★，不是在虚拟机里
#
#  目的：验证 kernel launch 末端的 doorbell MMIO 写是否被 KVM trap。
#        每次 launch 对应一次 doorbell 写；若被 trap，
#        mmio_exits 的增量应约等于发射次数 N。
#
#  用法：
#     sudo ./kvm_mmio_host.sh                  # 交互式：快照 → 等你在 guest 里跑 → 快照
#     sudo ./kvm_mmio_host.sh --watch          # 持续采样，每秒打印增量
#     sudo ./kvm_mmio_host.sh --n 100000       # 告知预期发射次数，自动算比值
#
#  配套的 guest 侧程序：bench/mmio_probe.cu
#     在虚拟机里编译并运行：  ./mmio_probe 100000
# ============================================================
set -u
N_EXPECT=""; MODE=interactive; VMDIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --watch) MODE=watch;;
        --n) N_EXPECT="$2"; shift;;
        --vmdir) VMDIR="$2"; shift;;
        --help) sed -n '2,20p' "$0"; exit 0;;
        *) echo "未知参数 $1"; exit 1;;
    esac; shift
done

KVM=/sys/kernel/debug/kvm

# ---------- 前置检查 ----------
if [ "$(id -u)" != 0 ]; then
    echo "❌ 需要 root（debugfs 只有 root 可读）。请用 sudo 运行。"; exit 1
fi
if [ ! -d /sys/kernel/debug/kvm ]; then
    if ! mountpoint -q /sys/kernel/debug 2>/dev/null; then
        echo "debugfs 未挂载，尝试挂载…"
        mount -t debugfs none /sys/kernel/debug 2>/dev/null \
            || { echo "❌ 挂载 debugfs 失败"; exit 1; }
    fi
fi
if [ ! -d "$KVM" ]; then
    echo "❌ 找不到 $KVM"
    echo "   可能原因：这台机器不是 KVM 宿主机 / kvm 模块未加载 / 没有正在运行的虚拟机"
    echo "   确认：lsmod | grep kvm"
    exit 1
fi

# ---------- 选择统计源 ----------
# 新内核：每个虚拟机一个子目录 <pid>-<n>/，内含 per-vcpu 计数，更精确
# 老内核：$KVM 下直接是全局计数文件（所有虚拟机共享，需保证宿主机上只有一个 VM 在忙）
pick_dir() {
    [ -n "$VMDIR" ] && { echo "$VMDIR"; return; }
    local dirs; dirs=$(find "$KVM" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort)
    local cnt; cnt=$(echo "$dirs" | grep -c . || true)
    if [ "$cnt" = 0 ]; then echo "$KVM"; return; fi
    if [ "$cnt" = 1 ]; then echo "$dirs"; return; fi
    echo "检测到多个虚拟机，请用 --vmdir 指定其一：" >&2
    for d in $dirs; do
        local pid; pid=$(basename "$d" | cut -d- -f1)
        echo "  $d   (qemu pid=$pid, cmd=$(ps -p "$pid" -o comm= 2>/dev/null))" >&2
    done
    exit 1
}
DIR=$(pick_dir)
echo "统计目录: $DIR"
[ "$DIR" = "$KVM" ] && echo "⚠️  用的是全局计数（所有虚拟机共享）。请确保宿主机上只有被测虚拟机在活动。"

# 关心的计数项（按存在与否自动挑选）
STATS="mmio_exits io_exits exits halt_exits irq_exits signal_exits"
read_all() {
    local out=""
    for s in $STATS; do
        local f v=""
        for f in "$DIR/$s" "$DIR"/vcpu*/"$s"; do
            [ -r "$f" ] || continue
            v=$(( ${v:-0} + $(cat "$f" 2>/dev/null || echo 0) ))
        done
        [ -n "$v" ] && out="$out $s=$v"
    done
    echo "$out"
}
SNAP0=$(read_all)
[ -z "$SNAP0" ] && { echo "❌ 在 $DIR 下没找到任何可读的计数项"; ls "$DIR" 2>/dev/null | head; exit 1; }
echo "可用计数项: $(echo "$SNAP0" | tr ' ' '\n' | grep . | cut -d= -f1 | tr '\n' ' ')"
echo

getv() { echo "$1" | tr ' ' '\n' | grep "^$2=" | cut -d= -f2; }

report() { # $1=before $2=after
    echo
    printf "%-14s %16s %16s %16s\n" "计数项" "之前" "之后" "增量"
    local d_mmio=""
    for s in $STATS; do
        local a b
        a=$(getv "$1" "$s"); b=$(getv "$2" "$s")
        [ -z "$a" ] && continue
        printf "%-14s %16s %16s %16s\n" "$s" "$a" "$b" "$((b-a))"
        [ "$s" = mmio_exits ] && d_mmio=$((b-a))
    done
    echo
    if [ -n "$N_EXPECT" ] && [ -n "$d_mmio" ]; then
        echo "发射次数 N = $N_EXPECT"
        awk -v d="$d_mmio" -v n="$N_EXPECT" 'BEGIN{
            r=d/n; printf "mmio_exits 增量 / N = %.3f\n\n", r;
            if (r>0.8 && r<1.5)
                print "→ ★ 每次 launch 对应一次 MMIO exit：doorbell 写确实被 KVM trap 了。";
            else if (r<0.1)
                print "→ 几乎没有 MMIO exit：doorbell 写没有被 trap，KVM 在这条路径上无辜。";
            else
                print "→ 比值介于两者之间，需结合 exits 总数判断（可能部分 trap，或有其它来源）。";
        }'
    else
        echo "（未提供 --n，无法计算比值。重跑时加上 --n <发射次数> 可自动判读）"
    fi
}

# ---------- 两种模式 ----------
if [ "$MODE" = watch ]; then
    echo "持续采样中，每秒打印增量。请在虚拟机里运行 mmio_probe，看完后 Ctrl-C 退出。"
    echo
    prev=$(read_all)
    while sleep 1; do
        cur=$(read_all)
        a=$(getv "$prev" mmio_exits); b=$(getv "$cur" mmio_exits)
        e1=$(getv "$prev" exits);     e2=$(getv "$cur" exits)
        printf "[%s] mmio_exits +%-12s  exits +%s\n" "$(date +%H:%M:%S)" \
            "$((${b:-0}-${a:-0}))" "$((${e2:-0}-${e1:-0}))"
        prev="$cur"
    done
else
    echo "=== 第 1 次快照 ==="; echo "$SNAP0" | tr ' ' '\n' | grep . | sed 's/^/  /'
    echo
    echo "现在请到【虚拟机】里运行发射程序，例如："
    echo "    ./mmio_probe ${N_EXPECT:-100000}"
    echo
    read -r -p "跑完后按回车继续…" _
    SNAP1=$(read_all)
    report "$SNAP0" "$SNAP1"
fi
