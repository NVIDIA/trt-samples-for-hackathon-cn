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

SKIP_LIST=\
"""
APIs/
APIs-V2-deprecated/
"""

for dir in */;
do
    if echo $SKIP_LIST | grep -q $dir; then
        continue
    fi

    test $dir
done

python3 build-README.py

echo "Finish `basename $(pwd)`"
