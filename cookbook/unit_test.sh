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

export TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST=
for dir in */;
do
    if [ $dir = "99-Todo/" ] || [ $dir = "include/" ]; then
        continue
    fi

    test $dir
done

export TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST=1
for dir in */;
do
    if [ $dir = "99-Todo/" ] || [ $dir = "include/" ]; then
        continue
    fi

    test $dir
done

export TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST=

echo "Finish ALL tests"
