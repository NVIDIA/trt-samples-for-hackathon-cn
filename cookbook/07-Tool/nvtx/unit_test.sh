#!/bin/bash

set -e
set -x
#clear

make test

if [ $TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST ]; then
  rm -rf *.d *.o *.exe *.nsys-rep *.qdrep *.qdrep-nsys
fi

echo "Finish `basename $(pwd)`"
