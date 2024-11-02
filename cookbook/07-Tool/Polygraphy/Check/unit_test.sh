#!/bin/bash

set -e
set -x
#clear

chmod +x main.sh
./main.sh

polygraphy check        --help > Help-check.txt
polygraphy check lint   --help > Help-check-lint.txt

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.json *.log
fi

echo "Finish `basename $(pwd)`"
