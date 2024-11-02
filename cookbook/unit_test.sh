#!/bin/bash

set -e
set -x
#clear

export TRT_COOKBOOK_PATH=$(pwd)

function test ()
{
    pushd $1
    chmod +x unit_test.sh
    ./unit_test.sh
    popd
    echo "Finish $1"
}

EXCLUDE_LIST=\
"""
09-TRT-LLM/
99-Todo/
build/
dist/
include/
tensorrt_cookbook/
tensorrt_cookbook.egg-info/
"""

SKIP_LIST=\
"""
00-Data/
01-SimpleDemo/
02-API/
"""

fff=\
"""
03-Workflow/
04-Feature/
05-Plugin/
06-DLFrameworkTRT/
07-Tool/
08-Advance/
09-TRTLLM/
98-Uncategorized/
"""

for dir in */;
do
    if echo $EXCLUDE_LIST | grep -q $dir; then
        continue
    fi
    # Skip when the tests fail at somewhere in the half way
    if echo $SKIP_LIST | grep -q $dir; then
        continue
    fi

    test $dir
done

echo "Finish ALL tests"
