#/bin/bash

set -e
set -x
rm -rf replays/ *.json *.log *.onnx
#clear

# 01-Find the first failed subgraph
polygraphy debug reduce \
    $TRT_COOKBOOK_PATH/00-Data/model/model-unknown.onnx \
    --output reduced.onnx \
    --model-input-shapes 'x:[1,1,28,28]' \
    --check polygraphy run --trt \
    > 01-result.log 2>&1

# 02-Find flaky tactic
# + This example no longer works reliably
# + https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/debug/01_debugging_flaky_trt_tactics#debugging-flaky-tensorrt-tactics)
polygraphy run \
    $TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx \
    --onnxrt \
    --save-outputs model-trained.json \
    > 02-result.log 2>&1

polygraphy debug build \
    $TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx \
    --fp16 \
    --artifacts-dir replays \
    --save-tactics replay.json \
    --artifacts replay.json \
    --until=10 \
    --check polygraphy run polygraphy_debug.engine --trt --load-outputs model-trained.json
    >> 02-result.log 2>&1

# 03-Reduce Failing ONNX Models
# + Just shows the process of search failed node, but the model used in this example is no problem
polygraphy run \
    $TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx \
    --onnxrt \
    --onnx-outputs mark all \
    --save-inputs model-trained-input.json \
    --save-outputs model-trained-output.json \
    > 03-result.log 2>&1

polygraphy data to-input \
    model-trained-input.json \
    model-trained-output.json \
    -o model-trained-io.json
    >> 03-result.log 2>&1

polygraphy debug reduce \
    $TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx \
    --mode bisect \
    -o model-trained-reduce.onnx \
    --load-inputs model-trained-io.json \
    --check polygraphy run polygraphy_debug.onnx --trt --load-inputs model-trained-io.json --load-outputs model-trained-output.json
    >> 03-result.log 2>&1

# 04-Reduce Failing ONNX Models
# + Continue of the example above, and pretend we have a bad node of type "Gemm"
polygraphy debug reduce \
    $TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx \
    --mode bisect \
    --fail-regex "Op: Gemm" \
    -o model-trained-reduce-gemm.onnx \
    --check polygraphy inspect model polygraphy_debug.onnx --show layers \
    > 04-result.log 2>&1

echo "Finish"
