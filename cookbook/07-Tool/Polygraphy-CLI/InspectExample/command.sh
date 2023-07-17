#/bin/bash

set -e
set -x

clear
rm -rf ./*.onnx ./*.plan ./*.log ./*.raw ./*.tactic ./polygraphy_capability_dumps/ ./good ./bad

# 00-Create ONNX graphs with Onnx Graphsurgeon and corresponding TensorRT engine
python3 getOnnxModel.py

polygraphy run modelA.onnx \
    --trt \
    --save-engine ./model-00.plan \
    --save-tactics ./model-00.tactic \
    --trt-min-shapes 'tensorX:[1,1,28,28]' \
    --trt-opt-shapes 'tensorX:[4,1,28,28]' \
    --trt-max-shapes 'tensorX:[16,1,28,28]' \
    --input-shapes   'tensorX:[4,1,28,28]' \
    --save-inputs model-00-inputs.raw \
    --save-outputs model-00-outputs.raw \
    --silent

# 01-Export information of the ONNX file
polygraphy inspect model modelA.onnx \
    --model-type=onnx \
    --shape-inference \
    --show layers attrs weights \
    --list-unbounded-dds \
    --verbose \
    > result-01.log

# 02-Export information of the ONNX file as format of TensorRT INetwork
polygraphy inspect model modelA.onnx \
    --model-type=onnx \
    --convert-to=trt \
    --shape-inference \
    --show layers attrs weights \
    --list-unbounded-dds \
    --verbose \
    > result-02.log

# 03-Export information of the TensorRT engine（TensorRT>=8.2）
polygraphy inspect model model-00.plan \
    --model-type=engine \
    --shape-inference \
    --show layers attrs weights \
    --list-unbounded-dds \
    --verbose \
    > result-03.log

# 04-Export information of input / output data
polygraphy inspect data model-00-inputs.raw \
    --all \
    --show-values \
    --histogram \
    --num-items 5 \
    --line-width 100 \
    > result-04.log

polygraphy inspect data model-00-outputs.raw \
    --all \
    --show-values \
    --histogram \
    --num-items 5 \
    --line-width 100 \
    >> result-04.log

# 05-Export information of tactics, json -> txt
polygraphy inspect tactics model-00.tactic \
    > result-05.log

# 06-Judge whether a ONNX file is supported by TensorRT natively
# Notice:
# + The modelB is not fully supportede by TensorRT, so the output directory "polygraphy_capability_dumps" is crerated, which contains information of the subgraphs supported / unsupported by TensorRT natively
polygraphy inspect capability modelA.onnx \
    > result-06-A.log

polygraphy inspect capability modelB.onnx \
    > result-06-B.log 2>&1

# 07-Filter potentially bad TensorRT tactics
# Here we assume the output of model-00.plan is correct, and the output of another model-00-FP16.plan is incorrect, so we put them into two different directories (more tactic files in each directories is acceptable) to filter the tactics which may cause the error
polygraphy run modelA.onnx \
    --trt \
    --save-engine ./model-00-FP16.plan \
    --save-tactics ./model-00-FP16.tactic \
    --fp16 \
    --trt-min-shapes 'tensorX:[1,1,28,28]' \
    --trt-opt-shapes 'tensorX:[4,1,28,28]' \
    --trt-max-shapes 'tensorX:[16,1,28,28]' \
    --input-shapes   'tensorX:[4,1,28,28]' \
    --silent
mkdir good bad
cp model-00.tactic good/
cp model-00-FP16.tactic bad/

polygraphy inspect diff-tactics \
    --good ./good \
    --bad ./bad \
    > result-07.log
