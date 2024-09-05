#/bin/bash

set -e
set -x
rm -rf *.log *.onnx *.so *.yaml
#clear

# 00-Get ONNX model
export MODEL_ADDSCALAR=$TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx

pushd $TRT_COOKBOOK_PATH/05-Plugin/BasicExample
make clean
make
popd
cp $TRT_COOKBOOK_PATH/05-Plugin/BasicExample/AddScalarPlugin.so .

#01-?
polygraphy plugin list $MODEL_ADDSCALAR \
    --plugin-dir . \
    > reuslt-01.log 2>&1

#02-?
polygraphy plugin match $MODEL_ADDSCALAR \
    --plugin-dir . \
    > result-02.log 2>&1

#03-?
polygraphy plugin replace $MODEL_ADDSCALAR \
    --plugin-dir . \
    -o model-custom-op-RE.onnx \
    > result-03.log 2>&1

echo "Finish"
