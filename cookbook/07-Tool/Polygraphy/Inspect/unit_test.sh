#!/bin/bash

set -e
set -x
#clear

chmod +x main.sh
./main.sh

polygraphy inspect              --help > Help-inspect.txt
polygraphy inspect model        --help > Help-inspect-model.txt
polygraphy inspect data         --help > Help-inspect-data.txt
polygraphy inspect tactics      --help > Help-inspect-tactics.txt
polygraphy inspect capability   --help > Help-inspect-capability.txt
polygraphy inspect diff-tactics --help > Help-inspect-diff-tactics.txt
polygraphy inspect sparsity     --help > Help-inspect-sparsity.txt

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.json *.log *.onnx *.raw *.trt bad/ good/ polygraphy_capability_dumps/
fi

echo "Finish `basename $(pwd)`"
