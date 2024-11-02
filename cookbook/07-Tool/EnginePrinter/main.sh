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
    --onnx=$TRT_COOKBOOK_PATH/00-Data/model/model-large.onnx \
    --profilingVerbosity=detailed \
    --exportLayerInfo=model-large.json \
    --skipInference

python3 main.py > log-main.py.log

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.log
fi
