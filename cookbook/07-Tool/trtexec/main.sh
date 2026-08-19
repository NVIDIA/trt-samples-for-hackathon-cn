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

set -xeuo pipefail

rm -rf *.json *.lock *.log *.onnx *.raw *.TimingCache *.trt *.trt.iter*

# 00-Create ONNX graphs with Onnx Graphsurgeon
export MODEL_TRAINED=${TRT_COOKBOOK_PATH}/00-Data/model/model-trained.onnx
export MODEL_ADDSCALAR=${TRT_COOKBOOK_PATH}/00-Data/model/model-addscalar.onnx

# 01-Run trtexec from ONNX file without any more option
trtexec \
    --onnx=$MODEL_TRAINED \
    > result-01.log 2>&1

# 02-Parse ONNX file, build and save TensorRT engine with regular options (see Help.txt to get more information)
# + For the shape option, use "x" to separate dimensions and use "," to separate the tensors (which is different from polygraphy)
# + e.g. "--optShapes=x:16x320x256,tensorY:8x4"
# + Input tensors of zero dimension should not appear in the shape options.
trtexec \
    --onnx=$MODEL_TRAINED \
    --saveEngine=model-trained.trt \
    --timingCacheFile=model-trained.TimingCache \
    --minShapes=x:1x1x28x28 \
    --optShapes=x:4x1x28x28 \
    --maxShapes=x:16x1x28x28 \
    --noTF32 \
    --memPoolSize=workspace:1024MiB \
    --builderOptimizationLevel=5 \
    --maxAuxStreams=4 \
    --skipInference \
    --verbose \
    > result-02.log 2>&1

# 03-Load TensorRT engine built above and do inference
trtexec \
    --loadEngine=model-trained.trt \
    --shapes=x:4x1x28x28 \
    --noDataTransfers \
    --useSpinWait \
    --useCudaGraph \
    --verbose \
    > result-03.log 2>&1

# 04-Print information of the TensorRT engine
# + `--profilingVerbosity=detailed` must be added during buildtime
# + output of "--dumpLayerInfo" locates in result*.log file, output of "--exportLayerInfo" locates in specified file
trtexec \
    --onnx=$MODEL_TRAINED \
    --skipInference \
    --profilingVerbosity=detailed \
    --dumpLayerInfo \
    --exportLayerInfo="./model-trained-exportLayerInfo.log" \
    > result-04.log 2>&1

# 05-Print information of profiling
# + output of "--dumpProfile" locates in result*.log file, output of "--exportProfile" locates in specified file
trtexec \
    --loadEngine=./model-trained.trt \
    --dumpProfile \
    --exportTimes="./model-trained-exportTimes.json" \
    --exportProfile="./model-trained-exportProfile.json" \
    > result-05.log 2>&1

# 06-Save data of input/output
# + output of "--dumpOutput" locates in result*.log file, output of "--dumpRawBindingsToFile" locates in *.raw files
trtexec \
    --loadEngine=./model-trained.trt \
    --dumpOutput \
    --dumpRawBindingsToFile \
    > result-06.log 2>&1

# 07-Run TensorRT engine with loading input data
trtexec \
    --loadEngine=./model-trained.trt \
    --loadInputs=x:x.input.1.1.28.28.fp32.raw \
    --dumpOutput \
    > result-07.log 2>&1

# 07.1-Import / Export data as raw format
python3 -c "import numpy as np; data=np.arange(60, dtype=np.float32).reshape([3,4,5]);data.tofile('x.raw')"
python3 -c "import numpy as np; data=np.fromfile('x.raw', dtype=np.float32);print(data)"

# 08-Build and run TensorRT engine with plugins
pushd ${TRT_COOKBOOK_PATH}/05-Plugin/BasicExample-V2-deprecated
make build
popd
cp ${TRT_COOKBOOK_PATH}/05-Plugin/BasicExample-V2-deprecated/AddScalarPlugin.so .

trtexec \
    --onnx=$MODEL_ADDSCALAR \
    --plugins=./AddScalarPlugin.so \
    > result-08.log 2>&1

# 09-Build a second engine at a different optimization level to get a reference profile for A/B
trtexec \
    --onnx=$MODEL_TRAINED \
    --saveEngine=model-trained-reference.trt \
    --builderOptimizationLevel=1 \
    --skipInference \
    > result-09.log 2>&1

trtexec \
    --loadEngine=./model-trained-reference.trt \
    --exportProfile="./model-trained-reference-exportProfile.json" \
    > result-10.log 2>&1

# 10-Consume the exported JSON (see parse_export_json.py)
# + `--exportProfile` -> per-layer time ranking + A/B comparison between the two engines above
# + `--exportTimes`   -> per-iteration H2D / compute / D2H trace + percentile latency
python3 parse_export_json.py \
    --profile=./model-trained-exportProfile.json \
    --times=./model-trained-exportTimes.json \
    --reference=./model-trained-reference-exportProfile.json \
    --threshold=5 \
    > result-11.log 2>&1

# 11-Accuracy checking against reference outputs
# Notice
# + trtexec compares against golden outputs itself, so a simple numerical check needs no Polygraphy.
# + "--accuracyThreshold" with a positive value is MANDATORY as soon as "--loadRefOutputs" or
#   "--refPair" is used, otherwise trtexec stops with an error and prints its whole help text.
# + "--refPair=N" groups one "--loadInputs" with one "--loadRefOutputs", and it needs **at least two
#   pairs**: with a single "--refPair=0" trtexec fails with
#   "When using --refPair, you need at least two pairs of I/O.". For one pair, just omit it.
# + Algorithms: l0 (fraction of elements outside atol/rtol), l1 (MAE), l2 (MSE), lInf (max abs error),
#   cos (1 - cosine similarity). Lower is better, 0.0 is a perfect match.
# The two inputs are generated unconditionally: step 17 below loads `xa.raw` whether or not this
# trtexec has `--accuracyAlgorithm`. Creating them inside the guard made step 17 depend on a file
# that a build without that flag never writes, which is exactly how this example broke on
# TensorRT 11.1 (trtexec here has no `--accuracyAlgorithm`).
python3 -c "
import numpy as np
np.random.default_rng(1).random([1, 1, 28, 28]).astype(np.float32).tofile('xa.raw')
np.random.default_rng(2).random([1, 1, 28, 28]).astype(np.float32).tofile('xb.raw')
"

if trtexec --help 2>&1 | grep -q -- "--accuracyAlgorithm"; then
    # Two (input, reference output) pairs, generated by running the engine itself
    for NAME in a b; do
        trtexec \
            --loadEngine=./model-trained.trt \
            --loadInputs=x:x${NAME}.raw \
            --dumpRawBindingsToFile \
            > /dev/null 2>&1
        cp y.output.1.10.fp32.raw y${NAME}.reference.raw
    done

    # 11A-One pair: no "--refPair" at all
    trtexec \
        --loadEngine=./model-trained.trt \
        --loadInputs=x:xa.raw \
        --loadRefOutputs=y:ya.reference.raw \
        --accuracyAlgorithm=cos \
        --accuracyThreshold=1e-3 \
        > result-12.log 2>&1

    # 11B-Two pairs, which is the case "--refPair" exists for
    trtexec \
        --loadEngine=./model-trained.trt \
        --refPair=0 --loadInputs=x:xa.raw --loadRefOutputs=y:ya.reference.raw \
        --refPair=1 --loadInputs=x:xb.raw --loadRefOutputs=y:yb.reference.raw \
        --accuracyAlgorithm=cos \
        --accuracyThreshold=1e-3 \
        >> result-12.log 2>&1

    # 11C-A check that is supposed to fail: feed input a but the reference output of input b.
    # Without this, a passing check proves nothing about whether the check works at all.
    # trtexec exits non-zero here, hence the "|| true".
    trtexec \
        --loadEngine=./model-trained.trt \
        --loadInputs=x:xa.raw \
        --loadRefOutputs=y:yb.reference.raw \
        --accuracyAlgorithm=cos \
        --accuracyThreshold=1e-3 \
        >> result-12.log 2>&1 || true

    # 11D-The same mismatch under every algorithm. The numbers are NOT comparable to each other,
    # so a threshold tuned for one algorithm is meaningless for another.
    for ALGORITHM in l0 l1 l2 lInf cos; do
        trtexec \
            --loadEngine=./model-trained.trt \
            --loadInputs=x:xa.raw \
            --loadRefOutputs=y:yb.reference.raw \
            --accuracyAlgorithm=$ALGORITHM \
            --accuracyThreshold=1e9 \
            2>&1 | grep "Accuracy loss for tensor" >> result-12.log
    done
else
    echo "Skip accuracy checking, this trtexec has no --accuracyAlgorithm" > result-12.log
fi

# 12-Global performance tuner: search over a build-route expression
# Notice
# + The knobs are the **internal compiler knobs** listed by "--helpBuildRoute" (209 of them here,
#   tuner_version 2.19.45), such as "-conv_lowering" or "-kgen:tiling". They are NOT the ordinary
#   trtexec build options: "-builderOptimizationLevel=[1|3|5]" is rejected with
#   "Failed to parse --tuneBuildRoutes expression: Unknown knob: -builderOptimizationLevel".
# + "-knob=[a|b|c]" declares a variable knob, "-knob=fixed" pins one; knobs are space-separated.
# + "--tuningSearch": fast = baseline + one variation per knob (2 knobs of 2 and 3 values -> 4 routes);
#   full = Cartesian product (-> 6 routes); mixed = fast scan then exhaustive over the knobs that helped.
# + Every iteration forks a child trtexec with "--setBuildRoute=<route>", so any result is
#   reproducible on its own, and "--dryRun" lists the routes without building anything.
if trtexec --help 2>&1 | grep -q -- "--tuneBuildRoutes"; then
    # The knob database, which is the only place the legal knob names come from
    trtexec --helpBuildRoute > result-13.log 2>&1

    export TUNING_EXPRESSION="-conv_lowering=[on|off] -kgen:tiling=[0|1|2]"

    # 12A-How many routes each search algorithm enumerates, without building
    trtexec \
        --onnx=$MODEL_TRAINED \
        --tuneBuildRoutes="$TUNING_EXPRESSION" \
        --tuningSearch=fast \
        --dryRun \
        2>&1 | grep -E "Expanded to|would be tried|^\[.*\] \[I\] \[[0-9]+\]:" > result-14.log

    trtexec \
        --onnx=$MODEL_TRAINED \
        --tuneBuildRoutes="$TUNING_EXPRESSION" \
        --tuningSearch=full \
        --dryRun \
        2>&1 | grep -E "Expanded to|would be tried|^\[.*\] \[I\] \[[0-9]+\]:" >> result-14.log

    # 12B-Really run the search (~1 minute for these 4 routes on MNIST)
    trtexec \
        --onnx=$MODEL_TRAINED \
        --tuneBuildRoutes="$TUNING_EXPRESSION" \
        --tuningSearch=fast \
        --tuningCacheFile=./model-trained.TuningCache.json \
        --tuningTimeOut=300 \
        --saveEngine=model-trained-tuned.trt \
        >> result-14.log 2>&1

    # 12C-Read the tuning cache back. It is JSON-lines: the first line is the metadata (including the
    # complete default build route, i.e. the value of all 209 knobs), then one line per iteration.
    python3 -c "
import json
with open('model-trained.TuningCache.json') as f:
    lines = [json.loads(line) for line in f if line.strip()]
meta, iterations = lines[0], lines[1:]
print(f\"tuner_version={meta['tuner_version']}, search={meta['searching_algorithm']}, expression={meta['tuning_expr']}\")
print(f\"default build route has {len(meta['default_build_route'].split())} knobs\")
best = min(iterations, key=lambda x: x['gpu_time'])
for row in iterations:
    mark = ' <- best' if row is best else ''
    print(f\"  iter {row['iter']}: gpu_time={row['gpu_time'] * 1000:.4f} us  route='{row['build_route']}'{mark}\")
spread = (max(r['gpu_time'] for r in iterations) / best['gpu_time'] - 1) * 100
print(f'spread between best and worst route: {spread:.2f}%')
" >> result-14.log 2>&1
else
    echo "Skip build-route tuning, this trtexec has no --tuneBuildRoutes" > result-13.log
    cp result-13.log result-14.log
fi

# 13-Strongly typed network, and the precision flags that no longer exist
# Notice
# + Upstream `samples/trtexec/README.md` (Example 6) presents "--stronglyTyped" as the way to opt in
#   to `kSTRONGLY_TYPED`. On TensorRT 11 it is a **no-op**: the default build already reports
#   "Precision: Strongly Typed" (grep result-02.log, which passes no such flag).
# + The flags that example warns against combining it with are gone entirely: "--fp16", "--int8",
#   "--best", "--precisionConstraints" and "--layerPrecisions" are rejected as "Unknown option".
#   That is a friendlier failure than Polygraphy's, whose "--fp16" still parses and then throws
#   from inside `CreateConfig` (see ../Polygraphy/README.md).
trtexec \
    --onnx=$MODEL_TRAINED \
    --stronglyTyped \
    --skipInference \
    --saveEngine=model-trained-stronglyTyped.trt \
    > result-15.log 2>&1

for OPTION in --fp16 --int8 --best --precisionConstraints=obey; do
    echo "--- trtexec $OPTION" >> result-15.log
    trtexec --onnx=$MODEL_TRAINED --skipInference $OPTION 2>&1 | grep -E "Unknown option" >> result-15.log || true
done

# 14-Throughput with several inference streams (upstream README Example 5)
# Notice
# + "--infStreams=N" runs N execution contexts concurrently. "--streams=N" is still accepted as the
#   older spelling that the upstream README uses.
# + Latency and throughput move in opposite directions here, and trtexec itself warns that the
#   latency numbers stop being meaningful once the streams overlap.
for N_STREAM in 1 2 4; do
    echo "--- infStreams=$N_STREAM" >> result-16.log
    trtexec \
        --loadEngine=./model-trained.trt \
        --infStreams=$N_STREAM \
        --iterations=200 \
        2>&1 | grep -E "Throughput:|GPU Compute Time:" >> result-16.log
done

# 15-Weight-stripped engine
# Notice
# + "--stripWeights" keeps the kernels and the schedule but drops the weights, here 13,250,084 B
#   -> 157,708 B (84x smaller). "--stripAllWeights" is the alias for "--refit --stripWeights".
# + **The stripped engine still loads and runs, reports PASSED, and outputs all zeros.** Nothing
#   warns about the missing weights, so a stripped plan that reaches production is silent.
# + Refitting it back from the CLI does not work on this build: "--refitFromOnnx" produces no log
#   line and the output stays zero, and "--dumpRefit" prints nothing. The working path is the
#   Python one, measured in ../../04-Feature/WeightStripping/ (same model, output restored exactly).
trtexec \
    --onnx=$MODEL_TRAINED \
    --stripWeights \
    --skipInference \
    --saveEngine=model-trained-stripped.trt \
    > result-17.log 2>&1

ls -l model-trained.trt model-trained-stripped.trt >> result-17.log 2>&1

echo "--- output of the full engine" >> result-17.log
trtexec --loadEngine=./model-trained.trt --loadInputs=x:xa.raw --dumpOutput --iterations=10 2>&1 \
    | grep -A1 "y: (1x10)" >> result-17.log

echo "--- output of the stripped engine (all zeros, still PASSED)" >> result-17.log
trtexec --loadEngine=./model-trained-stripped.trt --loadInputs=x:xa.raw --dumpOutput --iterations=10 2>&1 \
    | grep -A1 "y: (1x10)" >> result-17.log

echo "--- output after --refitFromOnnx (unchanged on TensorRT 11.1, see the note above)" >> result-17.log
trtexec --onnx=$MODEL_TRAINED --loadEngine=./model-trained-stripped.trt --refitFromOnnx --dumpRefit \
    --loadInputs=x:xa.raw --dumpOutput --iterations=10 2>&1 \
    | grep -A1 "y: (1x10)" >> result-17.log

echo "Finish"
