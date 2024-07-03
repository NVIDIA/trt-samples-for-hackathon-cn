#!/bin/bash

set -e
set -x
#clear

python3 main.py > result-Python.log

if [ $TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST ]; then
    rm *.log
fi

echo "Finish `basename $(pwd)`"
