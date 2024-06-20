#!/bin/bash

set -e
set -x
#clear

python3 main.py

pushd C++
make -j
./main.exe
popd

if [ $TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST ]; then
    rm -rf *.trt *.trt-* *.Int8Cache C++/*.d *.Int8Cache C++/*.exe *.Int8Cache C++/*.o *.Int8Cache C++/*.Int8Cache *.Int8Cache C++/*.trt
fi

echo "Finish `basename $(pwd)`"
