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

python3 main.py > log-main.py.log

for dir in */;
do
    test $dir
done

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.log
fi

echo "Finish `basename $(pwd)`"
