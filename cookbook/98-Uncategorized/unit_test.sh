#!/bin/bash

set -e
set -x
#clear

python3 build_number_md.py
python3 get_device_info.py
python3 get_library_info.py

echo "Finish `basename $(pwd)`"
