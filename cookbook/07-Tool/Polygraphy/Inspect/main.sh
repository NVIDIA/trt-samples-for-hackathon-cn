#/bin/bash

set -e
set -x
rm -rf *.json *.log *.onnx *.raw *.trt bad/ good/ polygraphy_capability_dumps/
#clear

# 00-Get engines
export MODEL_TRAINED=$TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx
export MODEL_TRAINED_SPARITY=$TRT_COOKBOOK_PATH/00-Data/model/model-trained-sparsity.onnx
export MODEL_UNKNOWN=$TRT_COOKBOOK_PATH/00-Data/model/model-unknown.onnx

polygraphy run \
    $MODEL_TRAINED \
    --trt \
    --save-engine ./model-trained.trt \
    --save-tactics ./model-trained-tactics.json \
    --trt-min-shapes 'x:[1,1,28,28]' \
    --trt-opt-shapes 'x:[4,1,28,28]' \
    --trt-max-shapes 'x:[16,1,28,28]' \
    --input-shapes   'x:[4,1,28,28]' \
    --save-inputs model-trained-inputs.raw \
    --save-outputs model-trained-outputs.raw \
    --silent

polygraphy run \
    $MODEL_TRAINED \
    --trt \
    --save-engine ./model-trained-FP16.trt \
    --save-tactics ./model-trained-FP16-tactics.json \
    --fp16 \
    --trt-min-shapes 'x:[1,1,28,28]' \
    --trt-opt-shapes 'x:[4,1,28,28]' \
    --trt-max-shapes 'x:[16,1,28,28]' \
    --input-shapes   'x:[4,1,28,28]' \
    --silent

# 01-Export information of the ONNX file
polygraphy inspect model \
    $MODEL_TRAINED \
    --model-type=onnx \
    --shape-inference \
    --show layers attrs weights \
    --list-unbounded-dds \
    --verbose \
    > result-01.log 2>&1

# 02-Export information of the TensorRT network
polygraphy inspect model \
    $MODEL_TRAINED \
    --model-type=onnx \
    --convert-to=trt \
    --shape-inference \
    --show layers attrs weights \
    --list-unbounded-dds \
    --verbose \
    > result-02.log 2>&1

# 03-Export information of the TensorRT engine
polygraphy inspect model \
    model-trained.trt \
    --model-type=engine \
    --shape-inference \
    --show layers attrs weights \
    --list-unbounded-dds \
    --verbose \
    > result-03.log 2>&1

# 04-Export information of input / output data
polygraphy inspect data \
    model-trained-inputs.raw \
    --all \
    --show-values \
    --histogram \
    --num-items 5 \
    --line-width 100 \
    > result-04.log 2>&1

polygraphy inspect data \
    model-trained-outputs.raw \
    --all \
    --show-values \
    --histogram \
    --num-items 5 \
    --line-width 100 \
    >> result-04.log 2>&1

# 05-Export information of tactics, json -> txt
polygraphy inspect tactics model-trained-tactics.json \
    > result-05.log

# 06-Judge whether a ONNX file is supported by TensorRT natively
# Notice:
# `$MODEL_UNKNOWN` is not fully supportede by TensorRT
# So the output directory "polygraphy_capability_dumps" is crerated, which contains information of the subgraphs supported / unsupported by TensorRT
polygraphy inspect capability \
    $MODEL_TRAINED \
    > result-06-A.log 2>&1

polygraphy inspect capability \
    $MODEL_UNKNOWN \
    > result-06-B.log 2>&1

# 07-Filter potentially bad TensorRT tactics
# Here we assume the output of model-trained.trt is correct, and the output of another model-trained-FP16.trt is incorrect
# So we put them into two different directories (more tactic files in each directories is acceptable) to filter the tactics which may cause the error
mkdir good bad
cp model-trained-tactics.json good/
cp model-trained-FP16-tactics.json bad/

polygraphy inspect diff-tactics \
    --good ./good \
    --bad ./bad \
    > result-07.log 2>&1

# 08-Check whether sparsity is supported by the model
polygraphy inspect sparsity \
    $MODEL_TRAINED_SPARITY \
    > result-08-A.log 2>&1

polygraphy inspect sparsity \
    $MODEL_TRAINED \
    > result-08-B.log 2>&1

echo "Finish"
