#!/bin/bash

set -e
set -x
#clear

python3 build_data_type_md.py -s 1 -e 4 -m 3 -b 0 -S False
python3 build_data_type_md.py -s 1 -e 5 -m 2 -b 0
python3 build_data_type_md.py -s 1 -e 5 -m 10 -b 0
python3 build_data_type_md.py -s 1 -e 8 -m 7 -b 0
python3 build_data_type_md.py -s 1 -e 8 -m 10 -b 0
python3 build_data_type_md.py -s 1 -e 8 -m 23 -b 0
python3 build_data_type_md.py -s 1 -e 11 -m 52 -b 0
#python3 build_data_type_md.py -s 1 -e 15 -m 112 -b 0
#python3 build_data_type_md.py -s 1 -e 19 -m 236 -b 0

python3 get_device_info.py
python3 get_library_info.py

echo "Finish `basename $(pwd)`"
