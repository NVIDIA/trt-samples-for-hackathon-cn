#/bin/bash

set -e
set -x
rm -rf *.lock *.log *.onnx *.TimingCache *.trt
#clear

#01-Parse ONNX file, build and run engine in TensorRT with polygraphy APIs
python3 main.py > result-01.log 2>&1

#02-Combinate workflow of onnx-graphsurgeon and polygraphy
python3 gs_workflow.py > result-02.log 2>&1

echo "Finish"
