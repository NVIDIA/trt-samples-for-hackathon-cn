#/bin/bash

set -e
set -x

clear
rm -rf ./*.onnx ./*.plan ./*.cache ./*.lock ./*.json ./*.log ./*.o ./*.d ./*.so ./polygraphyRun.py

# 00-Create ONNX graph and TensorRT engine
python3 getOnnxModel.py
polygraphy run modelA.onnx \
    --onnxrt \
    --save-inputs model-00-inputs.raw \
    --save-outputs model-00-outputs.raw

# 01-Combine input and output data into a raw file
polygraphy data to-input model-00-inputs.raw model-00-outputs.raw \
    --output model-00-io.raw \
    > result-01.log
