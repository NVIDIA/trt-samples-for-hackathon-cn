#!/bin/bash

set -e
set -x
#clear

python3 main_numpy.py           > log-main_numpy.py.log
python3 main_pytorch.py         > log-main_pytorch.py.log
python3 main_cookbook_flavor.py > log-main_cookbook_flavor.py.log

make test

if [ $TRT_COOKBOOK_CLEAN ]; then
    make clean
    rm -rf *.log
fi

echo "Finish `basename $(pwd)`"
