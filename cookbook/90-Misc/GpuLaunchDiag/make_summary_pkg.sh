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

# 打包整套检测设备（文档 + 全部可运行脚本），用于对内分享。
#
# ⚠️ 本包**在有 bin/perf_bugs 的机器上**会把它一起打进去（NVIDIA 内部工具），那种包仅限内部。
#    对外分享请改用 make_launchpmu_pkg.sh（~10 KB，纯源码，任何驱动版本可跑）。
#    结尾的提示会据实说明本次打出的包里到底有没有那个二进制 —— 不要凭这段注释判断。
set -eu
cd "$(dirname "$0")"
HERE="$(pwd)"

# perf_bugs 不随本仓库发布（见 cookbook/.gitignore）。公开检出里仍然可以打包 —— 文档与其余
# 脚本都在 —— 只是打出来的包不含那个二进制，所以这里只提示，不中止。
if [ ! -x bin/perf_bugs ]; then
    echo "注意：bin/perf_bugs 不存在（NVIDIA 内部工具，此处不公开，请与作者联系）。"
    echo "      本次打出的包不含该工具，收到包的人无法运行 T5，其余测项不受影响。"
fi
NAME="$(basename "$HERE")"
OUT="${1:-$HERE/../${NAME}.tar.gz}"

T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
D="$T/$NAME"; mkdir -p "$D"
# 排除产物目录、模型数据与打包临时物
tar cf - --exclude='./out' --exclude='./result_*' --exclude='__pycache__' \
         --exclude='*.tar.gz' -C "$HERE" . | tar xf - -C "$D"
bash -n "$D/run_all.sh"
tar czf "$OUT" -C "$T" "$NAME"
echo "已生成: $OUT  ($(du -h "$OUT" | cut -f1))"
# 据实报告，不要写死：是否含该二进制取决于打包这台机器上它在不在。
if [ -x "$D/bin/perf_bugs" ]; then
    echo "⚠️ 本包含 bin/perf_bugs（NVIDIA 内部工具），仅限内部分享"
else
    echo "本包不含 bin/perf_bugs，收到的人无法运行 T5；其余测项齐全。"
fi
