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

python3 main.py

for dir in */;
do
    test $dir
done

echo "Finish `basename $(pwd)`"
