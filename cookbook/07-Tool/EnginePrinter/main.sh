#/bin/bash

set -e
set -x
rm -rf *.json model-*.onnx
#clear

trtexec \
    --onnx=$TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx \
    --profilingVerbosity=detailed \
    --exportLayerInfo=model-trained.json \
    --skipInference

trtexec \
    --onnx=$TRT_COOKBOOK_PATH/00-Data/model/model-wenet.onnx \
    --profilingVerbosity=detailed \
    --exportLayerInfo=model-wenet.json \
    --skipInference

python3 main.py
