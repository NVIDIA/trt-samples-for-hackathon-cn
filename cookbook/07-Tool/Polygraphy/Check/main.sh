#/bin/bash

set -e
set -x
rm -rf .json *.log
#clear

# 01-Check the model
polygraphy check lint \
    $TRT_COOKBOOK_PATH/00-Data/model/model-unknown.onnx \
    -o model-unknown.json \
    > result-01.log 2>&1 || true

echo "Finish"
