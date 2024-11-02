#!/bin/bash

set -e
set -x
#clear

chmod +x main.sh
./main.sh

polygraphy surgeon          --help > Help-surgeon.txt
polygraphy surgeon extract  --help > Help-surgeon-extract.txt
polygraphy surgeon insert   --help > Help-surgeon-insert.txt
polygraphy surgeon prune    --help > Help-surgeon-prune.txt
polygraphy surgeon sanitize --help > Help-surgeon-sanitize.txt

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.log *.onnx
fi

echo "Finish `basename $(pwd)`"
