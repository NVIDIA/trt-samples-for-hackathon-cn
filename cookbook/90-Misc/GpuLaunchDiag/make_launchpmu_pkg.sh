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

# 生成独立的 launch-PMU 测试包（几 KB，全源码，目标机器上编译）。
set -eu
cd "$(dirname "$0")"
OUT="${1:-$PWD/../launchpmu-kit.tar.gz}"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
D="$T/launchpmu-kit"; mkdir -p "$D/bench"
cp bench/launch_pmu_probe.cu bench/freq_monitor.c "$D/bench/"
cp run_launch_pmu.sh pick_core.sh "$D/"
cp docs/launchpmu-README.md "$D/README.md"
bash -n "$D/run_launch_pmu.sh"
tar czf "$OUT" -C "$T" launchpmu-kit
echo "已生成: $OUT ($(du -h "$OUT"|cut -f1))"
