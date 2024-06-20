#!/bin/bash

set -e
set -x
#clear

make test

if [ $TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST ]; then
  make clean
fi

echo "Finish `basename $(pwd)`"
