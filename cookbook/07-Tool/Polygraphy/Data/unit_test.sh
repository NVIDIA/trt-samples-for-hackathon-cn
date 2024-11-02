#!/bin/bash

set -e
set -x
#clear

chmod +x main.sh
./main.sh

polygraphy data             --help > Help-data.txt
polygraphy data to-input    --help > Help-data-to-input.txt

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.log *.onnx *.raw
fi

echo "Finish `basename $(pwd)`"
