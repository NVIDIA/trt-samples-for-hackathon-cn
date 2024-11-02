#!/bin/bash

set -e
set -x
#clear

SKIP_LIST=\
"""
add_scalar_numba.py
"""

for file in *.py;
do
    if echo $SKIP_LIST | grep -q $file; then
        continue
    fi

    python3 $file > log-$file.log
done

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.trt *.log
fi

echo "Finish `basename $(pwd)`"
