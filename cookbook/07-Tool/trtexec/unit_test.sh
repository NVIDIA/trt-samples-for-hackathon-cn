#!/bin/bash

set -e
set -x
#clear

chmod +x main.sh
./main.sh

trtexec --help > Help.txt

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.json *.lock *.log *.onnx *.raw *.TimingCache *.trt
fi

echo "Finish `basename $(pwd)`"
