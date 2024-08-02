#!/bin/bash

set -e
set -x
#clear

python3 main.py

echo "Finish `basename $(pwd)`"
