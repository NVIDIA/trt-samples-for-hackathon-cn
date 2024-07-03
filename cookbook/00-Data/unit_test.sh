#!/bin/bash

set -e
set -x
#clear

python3 extract_MNIST.py
python3 get_model.py

# Do not remove files after unit tests
#if [ $TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST ]; then
#    rm -rf data/test data/train data/*.npy data/*.npz models/*.onnx models/*.weight model/*npz models/*.pth
#fi

echo "Finish `basename $(pwd)`"
