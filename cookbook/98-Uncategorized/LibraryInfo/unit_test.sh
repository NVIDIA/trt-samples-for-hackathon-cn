#!/bin/bash

set -e
set -x
#clear

python3 main.py > log-main.py.log

if [ ! $TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST ]; then
  rm -rf *.log
fi

echo "Finish `basename $(pwd)`"
