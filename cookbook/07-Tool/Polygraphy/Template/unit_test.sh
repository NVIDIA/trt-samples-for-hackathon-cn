#!/bin/bash

set -e
set -x
#clear

chmod +x main.sh
./main.sh

polygraphy template             --help > Help-template.txt
polygraphy template trt-network --help > Help-template-trt-network.txt
polygraphy template trt-config  --help > Help-template-trt-config.txt
polygraphy template onnx-gs     --help > Help-template-onnx-gs.txt

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.log *.onnx *.trt modify_config.py modify_network.py modify_onnx.py
fi

echo "Finish `basename $(pwd)`"
