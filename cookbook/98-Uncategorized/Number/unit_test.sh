#!/bin/bash

set -xeuo pipefail

python3 build_number_md.py > log-build_number_md.py.log
python3 build_number_picture.py > log-build_number_picture.py.log

if [ ! $TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST ]; then
  rm -rf *.log
fi

echo "Finish `basename $(pwd)`"
