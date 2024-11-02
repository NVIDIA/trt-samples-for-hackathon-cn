#!/bin/bash

set -e
set -x
#clear

python3 main.py > log-main.py.log
nsys profile -o Multi-Stream -f true python3 main.py

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.nsys-rep *.log
fi

echo "Finish `basename $(pwd)`"
