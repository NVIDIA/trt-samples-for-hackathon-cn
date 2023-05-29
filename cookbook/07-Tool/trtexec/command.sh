#/bin/bash

set -e set -x

clear
rm -rf ./*.onnx ./*.plan ./result-*.log

# 01-Create a ONNX graph with Onnx Graphsurgeon
python3 getOnnxModel.py

# 02-Build TensorRT engine from ONNX file using FP16 mode, just build and save engine
trtexec \
    --onnx=model.onnx \
    --minShapes=tensor-0:1x1x28x28 \
    --optShapes=tensor-0:4x1x28x28 \
    --maxShapes=tensor-0:16x1x28x28 \
    --memPoolSize=workspace:1024MiB \
    --saveEngine=model-FP32.plan \
    --skipInference \
    --fp16 \
    --verbose \
    > result-02.log 2>&1

# Notie:
# + The format of shape of input tensor is different from polygrapy, use "x" to separate dimensions and use "," to separate the input tensors
#       For example: "--optShapes=tensor-0:16x320x256,tensor-1:16x320,tensor-2:16"
# + Use "-buildOnly" rather than "--skipInference" if using TensorRT<=8.5
# + Inference and performance test with random data of smallest input shape will be done if not using "--skipInference"

# 03-Load TensorRT engine built above to do inference
trtexec \
    --loadEngine=./model-FP32.plan \
    --shapes=tensor-0:4x1x28x28 \
    --verbose \
    > result-03.log 2>&1

# 04-Print information of the TensorRT engine built above (since TensorRT 8.4）
trtexec \
    --loadEngine=./model-FP32.plan \
    --shapes=tensor-0:4x1x28x28 \
    --verbose \
    --dumpLayerInfo \
    --exportLayerInfo="./modelInformation.log" \
    > result-04.log 2>&1
