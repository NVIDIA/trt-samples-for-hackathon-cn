#!/bin/bash

set -e
set -x
#clear

python3 main.py
nsys profile -o Multi-Stream -f true python3 main.py

if [ $TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST ]; then
    rm -rf *.nsys-rep
fi

echo "Finish `basename $(pwd)`"
