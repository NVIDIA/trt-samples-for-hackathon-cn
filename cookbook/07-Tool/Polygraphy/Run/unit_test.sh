#!/bin/bash

set -e
set -x
#clear

chmod +x main.sh
./main.sh

polygraphy run --help > Help-run.txt

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.json *.lock *.log *.onnx *.so *.TimingCache *.trt polygraphy_run.py
fi

echo "Finish `basename $(pwd)`"
