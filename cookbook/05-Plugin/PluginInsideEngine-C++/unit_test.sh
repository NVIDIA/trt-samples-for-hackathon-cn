#!/bin/bash

set -e
set -x
#clear

make test

if [ $TRT_COOKBOOK_CLEAN ]; then
    make clean
    rm -rf *.log
fi

echo "Finish `basename $(pwd)`"
