#/bin/bash

set -e
set -x

clear
rm -rf ./*.onnx ./*.plan ./*.cache ./*.lock ./*.json ./*.log ./*.o ./*.d ./*.so ./polygraphyRun.py

# 00-Create ONNX graphs with Onnx Graphsurgeon
python3 getOnnxModel.py

# 01-Run polygraphy from ONNX file in onnxruntime without any more option
polygraphy run modelA.onnx \
    --onnxrt \
    > result-01.log 2>&1

# 02-Parse ONNX file, build and save TensorRT engine with more options (see Help.txt to get more information)
# Notie:
# + For the shape option, use "," to separate dimensions and use " " to separate the tensors (which is different from trtexec)
# + For example options of a model with 3 input tensors named "tensorX" and "tensorY" and "tensorZ" should be like "--trt-min-shapes 'tensorX:[16,320,256]' 'tensorY:[8,4]' tensorZ:[]"
polygraphy run modelA.onnx \
    --trt \
    --save-engine model-02.plan \
    --save-timing-cache model-02.cache \
    --save-tactics model-02-tactics.json \
    --trt-min-shapes 'tensorX:[1,1,28,28]' \
    --trt-opt-shapes 'tensorX:[4,1,28,28]' \
    --trt-max-shapes 'tensorX:[16,1,28,28]' \
    --fp16 \
    --pool-limit workspace:1G \
    --builder-optimization-level 5 \
    --max-aux-streams 4 \
    --input-shapes   'tensorX:[4,1,28,28]' \
    --verbose \
    > result-02.log 2>&1

# 03-Run TensorRT engine
polygraphy run model-02.plan \
    --trt \
    --input-shapes 'tensorX:[4,1,28,28]' \
    --verbose \
    > result-03.log 2>&1

# 04-Compare the output of each layer between Onnxruntime and TensorRT
polygraphy run modelA.onnx \
    --onnxrt --trt \
    --save-engine=model-04.plan \
    --onnx-outputs mark all \
    --trt-outputs mark all \
    --trt-min-shapes 'tensorX:[1,1,28,28]' \
    --trt-opt-shapes 'tensorX:[4,1,28,28]' \
    --trt-max-shapes 'tensorX:[16,1,28,28]' \
    --input-shapes   'tensorX:[4,1,28,28]' \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    > result-04.log 2>&1

# 05-Compare the output of certain layer(s) between Onnxruntime and TensorRT
# Notice:
# + Use " " to separate names of the tensors need to be compared
polygraphy run modelA.onnx \
    --onnxrt --trt \
    --save-engine=model-04.plan \
    --onnx-outputs A-V-2-MaxPool A-V-5-MaxPool \
    --trt-outputs A-V-2-MaxPool A-V-5-MaxPool \
    --trt-min-shapes 'tensorX:[1,1,28,28]' \
    --trt-opt-shapes 'tensorX:[4,1,28,28]' \
    --trt-max-shapes 'tensorX:[16,1,28,28]' \
    --input-shapes   'tensorX:[4,1,28,28]' \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    > result-05.log 2>&1

# 06-Generate a script to do the same work as part 02, afterward we can use command "python polygraphyRun.py" to actually run it
polygraphy run modelA.onnx \
    --trt \
    --save-engine model-02.plan \
    --save-timing-cache model-02.cache \
    --save-tactics model-02-tactics.json \
    --trt-min-shapes 'tensorX:[1,1,28,28]' \
    --trt-opt-shapes 'tensorX:[4,1,28,28]' \
    --trt-max-shapes 'tensorX:[16,1,28,28]' \
    --fp16 \
    --pool-limit workspace:1G \
    --builder-optimization-level 5 \
    --max-aux-streams 4 \
    --input-shapes   'tensorX:[4,1,28,28]' \
    --silent \
    --gen-script=./polygraphyRun.py \
    > result-06.log 2>&1

# 07-Build and run TensorRT engine with plugins
make
polygraphy run modelB.onnx \
    --trt \
    --plugins ./AddScalarPlugin.so \
    > result-07.log 2>&1

# 08-Validate the output of ONNX files
polygraphy run modelA.onnx \
    --onnxrt \
    --validate \
    --fail-fast \
    --verbose \
    > result-08.log

polygraphy run modelC.onnx \
    --onnxrt \
    --validate \
    --fail-fast \
    --verbose \
    >> result-08.log 2>&1

# 09-Save and load input/output data
polygraphy run model-02.plan \
    --trt \
    --input-shapes 'tensorX:[4,1,28,28]' \
    --save-inputs "input.json" \
    --save-outputs "output.json" \
    --verbose

polygraphy run model-02.plan \
    --trt \
    --input-shapes 'tensorX:[4,1,28,28]' \
    --load-inputs "input.json" \
    --load-outputs "output.json" \
    --verboses
