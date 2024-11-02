#!/bin/bash

set -e
set -x
rm -rf *.log *.onnx *.nsys-rep *.qdrep *.qdrep-nsys *.trt
#clear

export MODEL_TRAINED=$TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx

nsys profile \
    --force-overwrite=true \
    -o BuildAndRun \
    trtexec \
        --onnx=$MODEL_TRAINED \
        --saveEngine=model-trained.trt \
    > result-01.log 2>&1

nsys profile \
    --force-overwrite=true \
    -o LoadAndRun \
    trtexec \
        --loadEngine=model-trained.trt \
    > result-02.log 2>&1

echo "Finish"
