#/bin/bash

set -e
set -x
rm -rf *.log *.onnx *.trt modify_config.py modify_network.py modify_onnx.py
#clear

export MODEL_TRAINED=$TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx

# 01-Create a script to modify the network
polygraphy template trt-network \
    $MODEL_TRAINED \
    --output modify_network.py

# Once we finish the edition, we can use convert mode to build the TensorRT engine
polygraphy convert \
    modify_network.py \
    --convert-to trt \
    --output "./model-trained.trt" \
    --model-type=trt-network-script \
    > result-01.log 2>&1

#02-Create a script to modify the config in TensorRT. TODO: how to use it?
polygraphy template trt-config \
    $MODEL_TRAINED \
    --output modify_config.py

#03-Create a script to modify ONNX file
polygraphy template onnx-gs \
    $MODEL_TRAINED \
    --output modify_onnx.py

echo "Finish"
