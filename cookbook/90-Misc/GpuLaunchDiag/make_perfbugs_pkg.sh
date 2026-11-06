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

# 生成独立的 perf_bugs 测试包（约 1.6 MB），只含 perf_bugs 相关文件，不含模型数据。
#
# ⚠️ **仅限内部**：本包含 bin/perf_bugs（NVIDIA 内部驱动微基准）。
#    对外分享请改用 make_launchpmu_pkg.sh —— 约 10 KB，纯源码，任何驱动版本都能跑。
set -eu
cd "$(dirname "$0")"

# perf_bugs 不随本仓库发布（见 cookbook/.gitignore），公开检出里打不出这个包。
if [ ! -x bin/perf_bugs ]; then
    echo "跳过：本包依赖 NVIDIA 内部工具 perf_bugs，此处不公开，请与作者联系。"
    echo "  对外分享请用：bash make_launchpmu_pkg.sh"
    exit 0
fi

OUT="${1:-$PWD/../perfbugs-kit.tar.gz}"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
D="$T/perfbugs-kit"; mkdir -p "$D/bin"
cp bin/perf_bugs bin/*_testlist.txt "$D/bin/"
cp run_perf_bugs.sh pick_core.sh "$D/"
cp bench/freq_monitor.c "$D/"
cp docs/perfbugs-README.md "$D/README.md"
bash -n "$D/run_perf_bugs.sh"
tar czf "$OUT" -C "$T" perfbugs-kit
echo "已生成: $OUT  ($(du -h "$OUT" | cut -f1))"
