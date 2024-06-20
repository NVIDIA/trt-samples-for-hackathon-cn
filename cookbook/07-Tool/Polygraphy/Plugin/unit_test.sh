#!/bin/bash

set -e
set -x
#clear

chmod +x main.sh
./main.sh

polygraphy plugin           --help > Help-plugin.txt
polygraphy plugin list      --help > Help-plugin-list.txt
polygraphy plugin match     --help > Help-plugin-match.txt
polygraphy plugin replace   --help > Help-plugin-replace.txt

if [ $TRT_COOKBOOK_CLEAN_AFTER_UNIT_TEST ]; then
    rm -rf *.log *.onnx *.so *.yaml
fi

echo "Finish `basename $(pwd)`"
