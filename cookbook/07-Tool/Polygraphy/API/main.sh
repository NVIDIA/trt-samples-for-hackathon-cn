#/bin/bash

set -e
set -x
rm -rf *.lock *.log *.onnx *.TimingCache *.trt
#clear

# 00-Get ONNX model
cp $TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx .
cp $TRT_COOKBOOK_PATH/00-Data/model/model-redundant.onnx .

#01-Parse ONNX file, build and run engine in TensorRT with polygraphy APIs
python3 main.py > result-01.log 2>&1

#02-Combinate workflow of onnx-graphsurgeon and polygraphy
python3 gs_workflow.py

echo "Finish"
