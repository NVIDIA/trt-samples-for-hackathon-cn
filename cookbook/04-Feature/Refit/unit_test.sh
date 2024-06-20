#!/bin/bash

set -e
set -x
#clear

python3 main.py

if [ $TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST ]; then
    rm -rf *.trt
fi

echo "Finish `basename $(pwd)`"
