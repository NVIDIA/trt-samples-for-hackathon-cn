#!/bin/bash

set -e
set -x
#clear

chmod +x main.sh
./main.sh

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.log *.onnx *.weight
fi

echo "Finish `basename $(pwd)`"
