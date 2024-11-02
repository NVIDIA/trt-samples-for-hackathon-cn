#!/bin/bash

set -e
set -x
#clear

chmod +x main.sh
./main.sh

polygraphy convert --help > Help-convert.txt

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.json *.lock *.log *.onnx *.so *.TimingCache *.trt
fi

echo "Finish `basename $(pwd)`"
