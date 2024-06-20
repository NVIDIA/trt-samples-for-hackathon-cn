#/bin/bash

set -e
set -x
rm -rf *.json *.log *.onnx
#clear

#00-Get ONNX model
cp $TRT_COOKBOOK_PATH/00-Data/model/model-unknown.onnx .

# 01-Use Reduce tool to find the first failed subgraph
polygraphy debug reduce model-unknown.onnx \
    --output reduced.onnx \
    --model-input-shapes 'x:[1,1,28,28]' \
    --check polygraphy run --trt \
    > 01-result.log 2>&1
