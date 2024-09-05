#/bin/bash

set -e
set -x
rm -rf .json *.log
#clear

export MODEL_UNKNOWN=$TRT_COOKBOOK_PATH/00-Data/model/model-unknown.onnx

# 01-Check the model
polygraphy check lint \
    $MODEL_UNKNOWN \
    -o model-unknown.json \
    > result-01.log 2>&1 || true

echo "Finish"
