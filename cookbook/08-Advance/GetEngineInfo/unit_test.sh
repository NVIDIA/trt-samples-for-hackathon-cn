#!/bin/bash

set -e
set -x
#clear

pushd $TRT_COOKBOOK_PATH/01-SimpleDemo/TensorRT-10.0
python3 main_numpy.py
popd

cp $TRT_COOKBOOK_PATH/01-SimpleDemo/TensorRT-10.0/model.trt .

make
./main.exe model.trt

if [ $TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST ]; then
    rm -rf *.trt
fi

echo "Finish `basename $(pwd)`"
