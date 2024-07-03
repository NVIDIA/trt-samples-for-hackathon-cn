#!/bin/bash

set -e
set -x
#clear

chmod +x main.sh
./main.sh

if [ $TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST ]; then
    rm -rf *.json model-*.onnx
fi

echo "Finish `basename $(pwd)`"
