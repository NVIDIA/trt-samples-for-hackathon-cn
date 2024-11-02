#!/bin/bash

set -e
set -x
#clear

# Build a engine with 2 optimization profiles (just for this example, 1 is enough in normal use case).
trtexec \
    --onnx=$TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx \
    --saveEngine=model.trt \
    --profile=0 \
        --minShapes=x:1x1x28x28 \
        --optShapes=x:4x1x28x28 \
        --maxShapes=x:16x1x28x28 \
    --profile=1 \
        --minShapes=x:8x1x28x28 \
        --optShapes=x:32x1x28x28 \
        --maxShapes=x:64x1x28x28 \
    --fp16 \
    --noTF32 \
    --memPoolSize=workspace:1024MiB \
    --builderOptimizationLevel=0 \
    --skipInference \
    --verbose

python3 main.py -i model.trt > log-main.py.log

make test

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.trt *.log
fi

echo "Finish `basename $(pwd)`"
