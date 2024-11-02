#!/bin/bash

set -e
set -x
#clear

pushd $TRT_COOKBOOK_PATH/01-SimpleDemo/TensorRT-10
python3 main_numpy.py
popd

cp $TRT_COOKBOOK_PATH/01-SimpleDemo/TensorRT-10/model.trt .

python3 main.py > log-main.py.log

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.trt *.log
fi

echo "Finish `basename $(pwd)`"
