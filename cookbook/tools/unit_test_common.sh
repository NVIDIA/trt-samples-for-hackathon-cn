#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared helper for unit_test.sh wrappers.
# Usage in each script:
#   SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
#   COMMON_ROOT="$SCRIPT_DIR"
#   while [ "$COMMON_ROOT" != "/" ] && [ ! -f "$COMMON_ROOT/tools/unit_test_common.sh" ]; do
#       COMMON_ROOT=$(dirname "$COMMON_ROOT")
#   done
#   source "$COMMON_ROOT/tools/unit_test_common.sh"
#   trt_bootstrap_runner "$SCRIPT_DIR"

trt_bootstrap_runner() {
    local start_dir=$1

    COOKBOOK_ROOT="$start_dir"
    while [ "$COOKBOOK_ROOT" != "/" ] && [ ! -f "$COOKBOOK_ROOT/tools/run_examples.py" ]; do
        COOKBOOK_ROOT=$(dirname "$COOKBOOK_ROOT")
    done

    HAS_RUNNER=0
    if [ -f "$COOKBOOK_ROOT/tools/run_examples.py" ]; then
        HAS_RUNNER=1
    fi

    export TRT_COOKBOOK_PATH="$COOKBOOK_ROOT"
}

trt_require_runner() {
    if [ "${HAS_RUNNER:-0}" -ne 1 ]; then
        echo "Can not find tools/run_examples.py from ${SCRIPT_DIR:-$(pwd)}"
        return 2
    fi
}

trt_run_case() {
    local case_dir=$1

    trt_require_runner || return $?

    local rel_case="${case_dir#$COOKBOOK_ROOT/}"
    local args=(--root "$COOKBOOK_ROOT" --case "$rel_case")
    if [ "${TRT_COOKBOOK_CLEAN-}" ]; then
        args+=(--clean)
    fi

    python3 "$COOKBOOK_ROOT/tools/run_examples.py" "${args[@]}"
}

trt_test_subdir() {
    local dir=$1
    local abs_dir
    abs_dir=$(cd -- "$dir" && pwd)

    if [ "${HAS_RUNNER:-0}" -eq 1 ] && [ -f "$abs_dir/unit_test.yaml" ]; then
        trt_run_case "$abs_dir"
    else
        pushd "$dir"
        chmod +x unit_test.sh
        ./unit_test.sh
        popd
    fi

    echo "Finish $dir"
}
