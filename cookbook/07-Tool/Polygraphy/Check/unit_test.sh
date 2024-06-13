#!/bin/bash

set -e
set -x
#clear

chmod +x main.sh
./main.sh

polygraphy check        --help > Help-check.txt
polygraphy check lint   --help > Help-check-lint.txt

if [ $TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST ]; then
    rm -rf *.log *.onnx
fi

echo "Finish `basename $(pwd)`"
