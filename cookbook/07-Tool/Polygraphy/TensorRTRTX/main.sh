#!/bin/bash
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

set -euo pipefail

rm -rf ./*.log ./*.plan ./*.cache ./*.lock

export MODEL_TRAINED=${TRT_COOKBOOK_PATH}/00-Data/model/model-trained.onnx

if ! python3 -c "import tensorrt_rtx" 2> /dev/null; then
    echo "tensorrt_rtx is not installed (pip install tensorrt_rtx), skipping."
    echo "Finish"
    exit 0
fi

# 01-Which TensorRT does polygraphy load? One environment variable decides.
echo "=== 01 module selection ==="
for value in 0 1; do
    POLYGRAPHY_USE_TENSORRT_RTX=${value} polygraphy run "${MODEL_TRAINED}" --trt -vv \
        > "result-01-rtx${value}.log" 2>&1
    printf "  POLYGRAPHY_USE_TENSORRT_RTX=%s -> %s\n" "${value}" \
        "$(grep -m1 'Loaded Module: tensorrt' "result-01-rtx${value}.log" | sed 's/^\s*//')"
done

# 02-The variable is read at import time. Setting it later is a no-op, and the precision flags
#    behave differently through the API than they do through the CLI. See rtx_api.py.
echo
echo "=== 02 Python API, without and with the variable ==="
python3 rtx_api.py                              2>&1 | tee    result-02.log
POLYGRAPHY_USE_TENSORRT_RTX=1 python3 rtx_api.py 2>&1 | tee -a result-02.log

# 03-The CLI does not refuse the same flags, it drops them.
echo
echo "=== 03 --fp16 through the CLI, under TensorRT-RTX ==="
POLYGRAPHY_USE_TENSORRT_RTX=1 polygraphy run "${MODEL_TRAINED}" --trt --fp16 -vv \
    > result-03.log 2>&1
printf "  requested --fp16, engine built with: %s\n" "$(grep -m1 'Flags  ' result-03.log | sed 's/.*| //')"
printf "  and the command still %s\n" "$(grep -oE 'PASSED|FAILED' result-03.log | tail -1)"

# 04-Cross-backend accuracy. Same ONNX, same harness, only the backend changes.
echo
echo "=== 04 accuracy against onnxruntime, both backends ==="
set +e
for value in 0 1; do
    for tolerance in 1e-5 1e-4; do
        POLYGRAPHY_USE_TENSORRT_RTX=${value} polygraphy run "${MODEL_TRAINED}" --trt --onnxrt \
            --atol "${tolerance}" --rtol "${tolerance}" > "result-04-rtx${value}-${tolerance}.log" 2>&1
        printf "  %-13s tol %s -> %s\n" \
            "$([ "${value}" = 1 ] && echo tensorrt_rtx || echo tensorrt)" "${tolerance}" \
            "$(grep -oE 'Pass Rate: [0-9.]+%' "result-04-rtx${value}-${tolerance}.log" | tail -1)"
    done
    printf "  %-13s %s\n" "" \
        "$(grep -oE 'max_absdiff=[^ ]+ .n=1., max_reldiff=[^ ]+' "result-04-rtx${value}-1e-5.log" | head -1)"
done
set -e

# 05-Ahead-of-time targeting. Only `convert` exposes it, and it validates the target by name.
echo
echo "=== 05 --compute-capabilities (polygraphy convert only) ==="
set +e
POLYGRAPHY_USE_TENSORRT_RTX=1 polygraphy run "${MODEL_TRAINED}" --trt --compute-capabilities 8.9 \
    > result-05-run.log 2>&1
printf "  run     --compute-capabilities 8.9  -> %s\n" "$(grep -m1 '\[E\]' result-05-run.log)"
for capability in 10.0 8.9 12.0; do
    POLYGRAPHY_USE_TENSORRT_RTX=1 polygraphy convert "${MODEL_TRAINED}" --convert-to trt \
        --compute-capabilities "${capability}" -o "sm-${capability}.plan" \
        > "result-05-${capability}.log" 2>&1
    if [ -f "sm-${capability}.plan" ]; then
        printf "  convert --compute-capabilities %-4s -> %s bytes\n" "${capability}" \
            "$(stat -c%s "sm-${capability}.plan")"
    else
        printf "  convert --compute-capabilities %-4s -> %s\n" "${capability}" \
            "$(grep -m1 '\[!\]' "result-05-${capability}.log" | sed 's/^\s*//')"
    fi
done
set -e
printf "  plans for the two accepted targets are %s\n" \
    "$(if cmp -s sm-8.9.plan sm-12.0.plan; then echo IDENTICAL; else echo different; fi)"

# 06-Four concurrent builds sharing one timing cache. polygraphy guards it with a LockFile.
echo
echo "=== 06 concurrent builds, one shared timing cache ==="
for i in 1 2 3 4; do
    POLYGRAPHY_USE_TENSORRT_RTX=1 polygraphy convert "${MODEL_TRAINED}" --convert-to trt \
        --save-timing-cache shared.cache -o "concurrent-${i}.plan" \
        > "result-06-${i}.log" 2>&1 &
done
wait
printf "  4 builds finished, failures: %s\n" "$(grep -lE '\[E\]|Traceback' result-06-*.log | wc -l)"
printf "  shared.cache %s bytes, lock file present: %s\n" \
    "$(stat -c%s shared.cache)" "$(test -f shared.cache.lock && echo yes || echo no)"

echo
echo "Finish"
