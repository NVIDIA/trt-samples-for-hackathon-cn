#!/bin/bash

set -e
set -x
#clear

make test

if [ $TRT_COOKBOOK_CLEAN ]; then
  rm -rf *.d *.o *.exe *.nsys-rep *.qdrep *.qdrep-nsys
fi

echo "Finish `basename $(pwd)`"
