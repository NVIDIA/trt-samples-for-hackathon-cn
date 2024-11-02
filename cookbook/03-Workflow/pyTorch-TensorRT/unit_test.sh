#!/bin/bash

set -e
set -x
#clear

python3 main.py > log-main.py.log

pushd C++
make test
popd

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.trt* *.Int8Cache C++/*.d C++/*.o C++/*.exe C++/*.trt C++/*.Int8Cache *.log
fi

echo "Finish `basename $(pwd)`"
