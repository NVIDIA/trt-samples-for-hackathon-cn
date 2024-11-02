#!/bin/bash

set -e
set -x
#clear

chmod +x main.sh
./main.sh

nsys                --help > Help.txt
nsys analyze        --help > Help-analyze.txt
nsys cancel         --help > Help-cancel.txt
nsys export         --help > Help-export.txt
nsys profile        --help > Help-profile.txt
nsys launch         --help > Help-launch.txt
nsys stop           --help > Help-stop.txt
nsys service        --help > Help-service.txt
nsys stats          --help > Help-stats.txt
nsys shutdown       --help > Help-shutdown.txt
nsys sessions list  --help > Help-sessions-list.txt
nsys recipe         --help > Help-recipe.txt
nsys nvprof         --help > Help-nvprof.txt

if [ $TRT_COOKBOOK_CLEAN ]; then
    rm -rf *.log *.onnx *.nsys-rep *.qdrep *.qdrep-nsys *.trt
fi

echo "Finish `basename $(pwd)`"
