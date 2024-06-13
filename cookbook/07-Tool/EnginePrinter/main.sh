#/bin/bash

set -e
set -x
rm -rf *.json *.onnx
#clear

cp $TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx .

trtexec \
    --onnx=model-trained-po.onnx \
    --profilingVerbosity=detailed \
    --exportLayerInfo=engine_layer.json \
    --skipInference

python3 main.py
