#!/bin/bash

set -e
set -x
#clear

python3 main_numpy.py
python3 main_pytorch.py
python3 main_cookbook_flavor.py

make
./main.exe

if [ $TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST ]; then
  rm -rf *.d *.exe *.o *.trt
fi

echo "Finish `basename $(pwd)`"
