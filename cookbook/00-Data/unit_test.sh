#!/bin/bash

set -e
set -x
#clear

python3 extract_MNIST.py
python3 get_model.py

# Do not remove files after unit tests
#if [ $TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST ]; then
#    rm -rf data/*/*.jpg. models/*.npz models/*.onnx models/*.weight
#fi

echo "Finish `basename $(pwd)`"
