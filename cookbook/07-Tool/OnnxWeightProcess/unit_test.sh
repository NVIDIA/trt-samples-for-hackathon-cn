#!/bin/bash

set -e
set -x
#clear

python main.py

if [ $TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST ]; then
    rm -rf *.onnx *.onnx.weight
fi

echo "Finish `basename $(pwd)`"
