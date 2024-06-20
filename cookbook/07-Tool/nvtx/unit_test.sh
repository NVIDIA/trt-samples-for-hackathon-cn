#!/bin/bash

set -e
set -x
#clear

python3 main.py

make
./main.exe

if [ $TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST ]; then
  rm -rf *.d *.o *.exe *.nsys-rep *.qdrep *.qdrep-nsys
fi

echo "Finish `basename $(pwd)`"
