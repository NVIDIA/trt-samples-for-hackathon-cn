#!/bin/bash

set -e
set -x
#clear

cp $TRT_COOKBOOK_PATH/00-Data/model/model-labeled.onnx .
python3 main.py

if [ $TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST ]; then
    rm -rf *.onnx
fi

echo "Finish `basename $(pwd)`"
