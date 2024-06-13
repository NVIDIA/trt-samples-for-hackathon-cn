#/bin/bash

set -e
set -x
rm -rf *.json *.lock *.log *.onnx *.raw *.TimingCache *.trt
#clear

# 00-Create ONNX graphs with Onnx Graphsurgeon
cp $TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx $TRT_COOKBOOK_PATH/00-Data/model/model-custom-op.onnx .

# 01-Run trtexec from ONNX file without any more option
trtexec \
    --onnx=model-trained.onnx \
    > result-01.log 2>&1

# 02-Parse ONNX file, build and save TensorRT engine with regular options (see Help.txt to get more information)
# + For the shape option, use "x" to separate dimensions and use "," to separate the tensors (which is different from polygraphy)
# + e.g. "--optShapes=x:16x320x256,tensorY:8x4"
# + Input tensors of zero dimension should not appear in the shape options.
trtexec \
    --onnx=model-trained.onnx \
    --saveEngine=model-trained.trt \
    --timingCacheFile=model-trained.TimingCache \
    --minShapes=x:1x1x28x28 \
    --optShapes=x:4x1x28x28 \
    --maxShapes=x:16x1x28x28 \
    --fp16 \
    --noTF32 \
    --memPoolSize=workspace:1024MiB \
    --builderOptimizationLevel=5 \
    --maxAuxStreams=4 \
    --skipInference \
    --verbose \
    > result-02.log 2>&1

# 03-Load TensorRT engine built above and do inference
trtexec \
    --loadEngine=model-trained.trt \
    --shapes=x:4x1x28x28 \
    --noDataTransfers \
    --useSpinWait \
    --useCudaGraph \
    --verbose \
    > result-03.log 2>&1

# 04-Print information of the TensorRT engine
# Notice
# + `--profilingVerbosity=detailed` must be added during buildtime
# + output of "--dumpLayerInfo" locates in result*.log file, output of "--exportLayerInfo" locates in specified file
trtexec \
    --onnx=model-trained.onnx \
    --skipInference \
    --profilingVerbosity=detailed \
    --dumpLayerInfo \
    --exportLayerInfo="./model-trained-exportLayerInfo.log" \
    > result-04.log 2>&1

# 05-Print information of profiling
# Notice
# + output of "--dumpProfile" locates in result*.log file, output of "--exportProfile" locates in specified file
trtexec \
    --loadEngine=./model-trained.trt \
    --dumpProfile \
    --exportTimes="./model-trained-exportTimes.json" \
    --exportProfile="./model-trained-exportProfile.json" \
    > result-05.log 2>&1

# 06-Save data of input/output
# Notice
# + output of "--dumpOutput" locates in result*.log file, output of "--dumpRawBindingsToFile" locates in *.raw files
trtexec \
    --loadEngine=./model-trained.trt \
    --dumpOutput \
    --dumpRawBindingsToFile \
    > result-06.log 2>&1

# 07-Run TensorRT engine with loading input data
trtexec \
    --loadEngine=./model-trained.trt \
    --loadInputs=x:x.input.1.1.28.28.fp32.raw \
    --dumpOutput \
    > result-07.log 2>&1

# 08-Build and run TensorRT engine with plugins
pushd $TRT_COOKBOOK_PATH/05-Plugin/BasicExample
make clean
make
popd
cp $TRT_COOKBOOK_PATH/05-Plugin/BasicExample/AddScalarPlugin.so .

trtexec \
    --onnx=model-custom-op.onnx \
    --plugins=./AddScalarPlugin.so \
    > result-08.log 2>&1
