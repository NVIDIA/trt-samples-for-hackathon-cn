#!/bin/bash

set -e
set -x
#clear

function test ()
{
    pushd $1
    chmod +x unit_test.sh
    ./unit_test.sh
    popd
    echo "Finish $1"
}

test TensorRT-10

python3 build-README.py

echo "Finish `basename $(pwd)`"
