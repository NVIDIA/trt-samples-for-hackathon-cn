#/bin/bash

set -e
set -x
rm -rf *.json *.lock *.log *.onnx *.so *.TimingCache *.trt polygraphy_run.py
#clear

# 00-Get ONNX model
export MODEL_TRAINED=$TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx
export MODEL_INVALID=$TRT_COOKBOOK_PATH/00-Data/model/model-invalid.onnx
export MODEL_ADDSCALAR=$TRT_COOKBOOK_PATH/00-Data/model/model-addscalar.onnx

# 01-Run polygraphy from ONNX file in onnxruntime without any more option
polygraphy run \
    $MODEL_TRAINED \
    --onnxrt \
    > result-01.log 2>&1

# 02-Parse ONNX file, build, save and run TensorRT engine with regular options (see Help.txt to get more parameters)
# + For the shape option, use "," to separate dimensions and use " " to separate the tensors (which is different from `trtexec`)
# + e.g. "--trt-min-shapes 'x:[16,320,256]' 'y:[8,4]' 'z:[]'"
# + Timing cache can be reused with `--load-timing-cache` during rebuild
# + More than one combination of `--trt-*-shapes` can be used for multiple optimization-profile
polygraphy run \
    $MODEL_TRAINED \
    --trt \
    --save-engine model-trained.trt \
    --save-timing-cache model-trained.TimingCache \
    --save-tactics model-trained-tactics.json \
    --trt-min-shapes 'x:[1,1,28,28]' \
    --trt-opt-shapes 'x:[4,1,28,28]' \
    --trt-max-shapes 'x:[16,1,28,28]' \
    --fp16 \
    --memory-pool-limit workspace:1G \
    --builder-optimization-level 5 \
    --max-aux-streams 4 \
    --input-shapes   'x:[4,1,28,28]' \
    --verbose \
    > result-02.log 2>&1

# 03-Load and run TensorRT engine
polygraphy run \
    model-trained.trt \
    --model-type engine \
    --trt \
    --input-shapes 'x:[4,1,28,28]' \
    --verbose \
    > result-03.log 2>&1

# 04-Compare the output of every layer between Onnxruntime and TensorRT
polygraphy run \
    $MODEL_TRAINED \
    --onnxrt --trt \
    --onnx-outputs mark all \
    --trt-outputs mark all \
    --trt-min-shapes 'x:[1,1,28,28]' \
    --trt-opt-shapes 'x:[4,1,28,28]' \
    --trt-max-shapes 'x:[16,1,28,28]' \
    --input-shapes   'x:[4,1,28,28]' \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    > result-04.log 2>&1

# 05-Compare the output of certain layer(s) between Onnxruntime and TensorRT
# Notice:
# + Use " " to separate names of the tensors need to be compared
polygraphy run \
    $MODEL_TRAINED \
    --onnxrt --trt \
    --onnx-outputs /MaxPool_output_0 /MaxPool_1_output_0 \
    --trt-outputs /MaxPool_output_0 /MaxPool_1_output_0 \
    --trt-min-shapes 'x:[1,1,28,28]' \
    --trt-opt-shapes 'x:[4,1,28,28]' \
    --trt-max-shapes 'x:[16,1,28,28]' \
    --input-shapes   'x:[4,1,28,28]' \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    > result-05.log 2>&1

# 06-Generate a script to do the same work as part 02, afterward we can use this script to actually run it
polygraphy run \
    $MODEL_TRAINED \
    --trt \
    --save-engine model-trained.trt \
    --save-timing-cache model-trained.TimingCache \
    --save-tactics model-trained-tactics.json \
    --trt-min-shapes 'x:[1,1,28,28]' \
    --trt-opt-shapes 'x:[4,1,28,28]' \
    --trt-max-shapes 'x:[16,1,28,28]' \
    --fp16 \
    --pool-limit workspace:1G \
    --builder-optimization-level 5 \
    --max-aux-streams 4 \
    --input-shapes   'x:[4,1,28,28]' \
    --silent \
    --gen-script=./polygraphy_run.py \
    > result-06.log 2>&1

python3 polygraphy_run.py >> result-06.log 2>&1

# 07-Build and run TensorRT engine with plugins
pushd $TRT_COOKBOOK_PATH/05-Plugin/BasicExample
make clean
make
popd
cp $TRT_COOKBOOK_PATH/05-Plugin/BasicExample/AddScalarPlugin.so .

polygraphy run \
    $MODEL_ADDSCALAR \
    --trt \
    --plugins ./AddScalarPlugin.so \
    > result-07.log 2>&1

# 08A-Validate the output of a ONNX file
polygraphy run \
    $MODEL_TRAINED \
    --onnxrt \
    --validate \
    --fail-fast \
    --verbose \
    > result-08.log 2>&1

# 08B-Validate the output of a invalid ONNX file
polygraphy run \
    $MODEL_INVALID \
    --onnxrt \
    --validate \
    --fail-fast \
    --verbose \
    >> result-08.log 2>&1 || true

# 09-Save and load input/output data for comparison
polygraphy run \
    model-trained.trt \
    --model-type engine \
    --trt \
    --input-shapes 'x:[4,1,28,28]' \
    --save-inputs "model-trained-input.json" \
    --save-outputs "model-trained-output.json" \
    --verbose \
    > result-09.log 2>&1

polygraphy run \
    model-trained.trt \
    --model-type engine \
    --trt \
    --input-shapes 'x:[4,1,28,28]' \
    --load-inputs "model-trained-input.json" \
    --load-outputs "model-trained-output.json" \
    --verbose \
    >> result-09.log 2>&1

echo "Finish"
