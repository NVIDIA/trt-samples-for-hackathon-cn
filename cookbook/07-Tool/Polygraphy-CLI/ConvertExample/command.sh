#/bin/bash

set -e
set -x

clear
rm -rf ./*.onnx ./*.plan ./*.cache ./*.lock ./*.json ./*.log ./*.o ./*.d ./*.so ./polygraphyRun.py

# 00-Create ONNX graphs with Onnx Graphsurgeon
python3 getOnnxModel.py

# 01-Build TensorRT engine from ONNX file without any more option
polygraphy convert modelA.onnx \
    --output ./model-01.plan \
    > result-01.log 2>&1

# 02-Parse ONNX file, build and save TensorRT engine with more options (see Help.txt to get more information)
# Notie:
# + For the shape option, use "," to separate dimensions and use " " to separate the tensors (which is different from trtexec)
# + For example options of a model with 3 input tensors named "tensorX" and "tensorY" and "tensorZ" should be like "--trt-min-shapes 'tensorX:[16,320,256]' 'tensorY:[8,4]' tensorZ:[]"
polygraphy convert modelA.onnx \
    --output ./model-02.plan \
    --save-timing-cache model-02.cache \
    --save-tactics model-02-tactics.json \
    --trt-min-shapes 'tensorX:[1,1,28,28]' \
    --trt-opt-shapes 'tensorX:[4,1,28,28]' \
    --trt-max-shapes 'tensorX:[16,1,28,28]' \
    --fp16 \
    --pool-limit workspace:1G \
    --builder-optimization-level 5 \
    --max-aux-streams 4 \
    --verbose \
    > result-02.log 2>&1

# 03-Convert a TensorRT INetwork into a ONNX-like file for visualization in Netron
polygraphy convert modelA.onnx \
    --output ./model-03.onnx \
    --convert-to onnx-like-trt-network \
    > result-03.log
