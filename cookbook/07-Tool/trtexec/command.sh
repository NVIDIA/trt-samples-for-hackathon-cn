#/bin/bash

set -e
set -x

clear
rm -rf ./*.onnx ./*.plan ./*.cache ./*.lock ./*.json ./*.raw ./*.log ./*.o ./*.d ./*.so

# 00-Create ONNX graphs with Onnx Graphsurgeon
python3 getOnnxModel.py

# 01-Run trtexec from ONNX file without any more option
trtexec \
    --onnx=modelA.onnx \
    > result-01.log 2>&1

# 02-Parse ONNX file, build and save TensorRT engine with more options (see HelpInformation.txt to get more information)
# Notie:
# + For the shape option, use "x" to separate dimensions and use "," to separate the tensors (which is different from polygraphy)
# + For example options of a model with 2 input tensors named "tensorX" and "tensorY" should be like "--optShapes=tensorX:16x320x256,tensorY:8x4"
# + Input tensors of zero dimension should not be appeared in the shape options
# + Use "--buildOnly" rather than "--skipInference" (TensorRT<=8.5)
trtexec \
    --onnx=modelA.onnx \
    --saveEngine=model-02.plan \
    --timingCacheFile=model-02.cache \
    --minShapes=tensorX:1x1x28x28 \
    --optShapes=tensorX:4x1x28x28 \
    --maxShapes=tensorX:16x1x28x28 \
    --fp16 \
    --noTF32 \
    --memPoolSize=workspace:1024MiB \
    --builderOptimizationLevel=5 \
    --maxAuxStreams=4 \
    --skipInference \
    --verbose \
    > result-02.log 2>&1

# 03-Load TensorRT engine built above and do inference
trtexec model-02.plan \
    --trt \
    --shapes=tensorX:4x1x28x28 \
    --noDataTransfers \
    --useSpinWait \
    --useCudaGraph \
    --verbose \
    > result-03.log 2>&1

# 04-Print information of the TensorRT engine (TensorRT>=8.4）
# Notice
# + Start from ONNX file because option "--profilingVerbosity=detailed" should be added during buildtime
# + output of "--dumpLayerInfo" locates in result*.log file, output of "--exportLayerInfo" locates in specified file
trtexec \
    --onnx=modelA.onnx \
    --skipInference \
    --profilingVerbosity=detailed \
    --dumpLayerInfo \
    --exportLayerInfo="./model-02-exportLayerInfo.log" \
    > result-04.log 2>&1

# 05-Print information of profiling
# Notice
# + output of "--dumpProfile" locates in result*.log file, output of "--exportProfile" locates in specified file
trtexec \
    --loadEngine=./model-02.plan \
    --dumpProfile \
    --exportTimes="./model-02-exportTimes.json" \
    --exportProfile="./model-02-exportProfile.json" \
    > result-05.log 2>&1

# 06-Save data of input/output
# Notice
# + output of "--dumpOutput" locates in result*.log file, output of "--dumpRawBindingsToFile" locates in *.raw files
trtexec \
    --loadEngine=./model-02.plan \
    --dumpOutput \
    --dumpRawBindingsToFile \
    > result-06.log 2>&1

# 07-Run TensorRT engine with loading input data
trtexec \
    --loadEngine=./model-02.plan \
    --loadInputs=tensorX:tensorX.input.1.1.28.28.Float.raw \
    --dumpOutput \
    > result-07.log 2>&1

# 08-Build and run TensorRT engine with plugins
make
trtexec \
    --onnx=modelB.onnx \
    --plugins=./AddScalarPlugin.so \
    > result-07.log 2>&1
