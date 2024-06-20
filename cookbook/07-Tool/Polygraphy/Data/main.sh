#/bin/bash

set -e
set -x
rm -rf *.log *.onnx *.raw
#clear

# 00-Get ONNX model
cp $TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx .

# 01-Save input / output data
polygraphy run model-trained.onnx \
    --onnxrt \
    --save-inputs model-trained-inputs.raw \
    --save-outputs model-trained-outputs.raw \
    > result-01.log 2>&1

# 02-Combine input and output data into a raw file
polygraphy data to-input model-trained-inputs.raw model-trained-outputs.raw \
    --output model-trained-io.raw \
    > result-02.log 2>&1
