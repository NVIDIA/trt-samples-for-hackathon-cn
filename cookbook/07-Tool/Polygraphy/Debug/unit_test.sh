#!/bin/bash

set -e
set -x
#clear

chmod +x main.sh
./main.sh

polygraphy debug            --help > Help-debug.txt
polygraphy debug build      --help > Help-debug-build.txt
polygraphy debug precision  --help > Help-debug-precision.txt
polygraphy debug reduce     --help > Help-debug-reduce.txt
polygraphy debug repeat     --help > Help-debug-repeat.txt

if [ $TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST ]; then
    rm -rf replays/ *.json *.log *.onnx
fi

echo "Finish `basename $(pwd)`"
