#/bin/bash

set -e
set -x
rm -rf *.log *.onnx
#clear

# 00-Get ONNX model
cp $TRT_COOKBOOK_PATH/00-Data/model/model-unknown.onnx .

# 01-Check the model
polygraphy check lint model-unknown.onnx \
    > result-01.log 2>&1 || true

echo "Finish"
