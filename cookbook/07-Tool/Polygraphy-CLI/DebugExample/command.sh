#/bin/bash

set -e
set -x

clear
rm -rf ./*.onnx ./*.plan ./*.cache ./*.lock ./*.json ./*.log ./*.o ./*.d ./*.so

# 00-Create ONNX graphs with Onnx Graphsurgeon
python3 getOnnxModel.py

# 01-Use Reduce tool to find the first failed subgraph
polygraphy debug reduce modelB.onnx \
    --output reduced.onnx \
    --model-input-shapes 'tensorX:[1,1,28,28]' \
    --check polygraphy run --trt
