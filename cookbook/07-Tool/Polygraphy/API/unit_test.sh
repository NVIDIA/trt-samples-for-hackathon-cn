#!/bin/bash

set -e
set -x
#clear

chmod +x main.sh
./main.sh

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.lock *.log *.onnx *.TimingCache *.trt
fi

echo "Finish `basename $(pwd)`"
