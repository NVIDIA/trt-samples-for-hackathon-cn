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
    skip="""
    if [ $dir = "00-Data/" ] || \
        [ $dir = "01-SimpleDemo/" ] || \
        [ $dir = "02-API/" ] || \
        [ $dir = "03-Workflow/" ] || \
        [ $dir = "04-Feature/" ] || \
        [ $dir = "05-Plugin/" ] || \
        [ $dir = "06-DLFrameworkTRT" ] ||
        [ $dir = "07-Tool" ] ||
        [ $dir = "08-Advance" ] \
        ; then
        continue
    fi
    """
    test $dir
done
skip="""
export TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST=1
for dir in */;
do
    if [ $dir = "99-Todo/" ] || [ $dir = "include/" ]; then
        continue
    fi

    test $dir
done

export TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST=
"""
echo "Finish ALL tests"
