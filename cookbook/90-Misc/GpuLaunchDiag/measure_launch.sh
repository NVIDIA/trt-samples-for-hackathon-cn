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

# 正确的 kernel launch 测量流程（替代原 README 的做法）
# 用法：./measure.sh <平台标签>    例：./measure.sh pro5000 / ./measure.sh l20
set -u
TAG="${1:?需要平台标签，如 pro5000 / l20}"
BIN="${BIN:-}"
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    BIN=./klb_${TAG}
    if [ ! -x "$BIN" ]; then
        SRC="$(dirname "$0")/bench/kernel_launch_bench_split.cu"
        [ -f "$SRC" ] || SRC=./kernel_launch_bench_split.cu
        [ -f "$SRC" ] || { echo "❌ 找不到源码 kernel_launch_bench_split.cu"; exit 1; }
        command -v nvcc >/dev/null || { echo "❌ 需编译但无 nvcc；可用 BIN=<路径> 指定"; exit 1; }
        ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
        nvcc -O3 -std=c++14 -arch=sm_${ARCH} "$SRC" -o "$BIN" || exit 1
    fi
fi
echo "使用可执行文件: $BIN"

# 1) 【关键】CPU 调频是头号干扰源：powersave 下空闲后第一个进程会虚高约 2 倍。
#    优先把 governor 设成 performance；没有 root 的话靠下面每次运行前的 CPU 预热兜底。
#    用 sudo -n：没有免密 sudo 时直接失败返回，不会卡在密码提示上。
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance | sudo -n tee "$g" >/dev/null 2>&1
done || true
echo "governor 现状: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
# 2) persistence mode 开着更好（省进程启动延迟），但实测它不是上面那个假象的成因。
sudo -n nvidia-smi -pm 1 >/dev/null 2>&1 || true
# 3) 不要锁 GPU 频率。上一期已证实锁频在 Blackwell+GDDR7 上会经 DRAM_STRICT 传播拖慢 L2 通路。
# 4) 选绑定的核：优先选 GPU 所在 NUMA 节点上的核。
#    双路机器上跨 NUMA 会多付 11~13% 的 launch 成本（见 README 第 3 节），
#    而"取最后一个核"的老办法在双路机上恰好会选到远端节点。
CORE="${CORE:-$(bash "$(dirname "$0")/pick_core.sh" --explain 0)}"

echo "平台=$TAG 绑核=$CORE"
for i in 1 2 3 4; do
    # 每次跑之前把该核空转 6s 拉到满频，消除调频假象
    taskset -c "$CORE" timeout 6 bash -c 'while :; do :; done'
    taskset -c "$CORE" $BIN 100000 2000 --rounds 3 --csv "${TAG}_run${i}.csv" > "${TAG}_run${i}.log" 2>&1
    echo "  run$i 完成"
done
echo "完成。**分析时丢弃 run1**，对后 3 次取中位数。"
