#!/bin/bash

set -e
set -x
#clear

for file in *.py;
do
    if [ $file -a add_scalar_numba.py ]; then
        continue
    fi

    python3 $file
done

if [ $TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST ]; then
    rm -rf *.trt
fi

echo "Finish `basename $(pwd)`"
