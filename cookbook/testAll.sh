#!/bin/bash
set -e
set -x

clear
tf=`pip list |grep "tensorflow-gpu"`
tfVersion=${tf#* }
tfMajorVersion=${tfVersion%%\.*}
pt=`pip list|grep "^torch\ "`
pd=`pip list|grep "^paddlepaddle-gpu\ "`

echo $tfMajorVersion
echo $pt
echo $pd

# 00 ---------------------------------------------------------------------------
echo "[00-MNISTData] Start"

cd 00-MNISTData
python3 extractMnistData.py
cd ..

echo "[00-MNISTData] Finish"

# 01 ---------------------------------------------------------------------------
echo "[01-SimpleDemo] Start"

cd 01-SimpleDemo

#cd TensorRT6
#make test > result.log
#cd ..

#cd TensorRT7
#make test > result.log
#cd ..

#cd TensorRT8.0
#make test > result.log
#cd ..

#cd TensorRT8.4
#make test > result.log
#cd ..

cd TensorRT8.5
make test > result.log
cd ..

cd ..
echo "[01-SimpleDemo] Finish"

# 02 ---------------------------------------------------------------------------
echo "[02-API] Start"

cd 02-API

cd Builder
python3 main.py > result.log
cd ..

cd BuilderConfig
python3 main.py > result.log
cd ..

cd CudaEngine
python3 main.py > result.log
cd ..

cd ExecutionContext
python3 main.py > result.log
cd ..

#cd Int8-PTQ
#python3 main.py > result.log
#cd ..

cd Int8-QDQ
python3 main.py > result.log
cd ..

cd Layer
python3 main.py > result.log
python3 testAllLayer.py
cd ..

cd Network
python3 main.py > result.log
cd ..

cd PrintNetwork
python3 main.py > result.log
cd ..

cd Tensor
python3 main.py > result.log
cd ..

cd ..
echo "[02-API] Finish"

# 03 ---------------------------------------------------------------------------
echo "[03-APIModel] Start"
cd 03-APIModel

if [[ $pd ]]; then
cd MNISTExample-Paddlepaddle
python3 main.py > result.log
cd ..
fi

if [[ $pt ]]; then
cd MNISTExample-pyTorch
python3 main.py > result.log
cd C++
make test 
cd ../..
fi

if [ $tfMajorVersion = "1" ]; then
cd MNISTExample-TensorFlow1
python3 main.py > result.log
cd ..
fi

if [ $tfMajorVersion = "2" ]; then
cd MNISTExample-TensorFlow2
python3 main.py > result.log
cd ..
fi

if [ $tfMajorVersion = "1" ]; then
cd TensorFlow1
python3 Convolution.py > result-Convolution.log
python3 FullyConnected.py > result-FullyConnected.log
python3 RNN-LSTM.py > result-RNN-LSTM.log
cd ..
fi

cd ..
echo "[03-APIModel] Finish"

# 04 ---------------------------------------------------------------------------
echo "[04-Parser] Start"
cd 04-Parser

if [[ $pd ]]; then
cd Paddlepaddle-ONNX-TensorRT
python3 main.py > result.log
cd ..
fi

if [[ $pt ]]; then
cd pyTorch-ONNX-TensorRT
python3 main.py > result.log
cd ..
fi 

if [[ $pt ]]; then
cd pyTorch-ONNX-TensorRT-QAT
python main.py > result.log
cd ..
fi 

#if [ $tfMajorVersion = "1" ]; then
#cd TensorFlow1-Caffe-TensorRT
#python buildModelInTensorFlow.py > result.log
#python runModelInTensorRT.py >> result.log
#cd ..
#fi 

if [ $tfMajorVersion = "1" ]; then
cd TensorFlow1-ONNX-TensorRT
python3 main-NCHW.py > result-NCHW.log
python3 main-NHWC.py > result-NHWC.log
python3 main-NHWC-C2.py > result-NHWC-C2.log
cd ..
fi 

if [ $tfMajorVersion = "1" ]; then
cd TensorFlow1-ONNX-TensorRT-QAT
python3 main.py > result.log
cd ..
fi

if [ $tfMajorVersion = "1" ]; then
cd TensorFlow1-UFF-TensorRT
python3 main.py > result.log
cd ..
fi

if [ $tfMajorVersion = "2" ]; then
cd TensorFlow2-ONNX-TensorRT
python main-NCHW.py > result-NCHW.log
python main-NHWC.py > result-NHWC.log
python main-NHWC-C2.py > result-NHWC-C2.log
cd ..
fi

#if [ $tfMajorVersion = "2" ]; then
#cd TensorFlow2-ONNX-TensorRT-QAT
#python main.py > result.log
#cd ..
#fi

cd ..
echo "[04-Parser] Finish"

# 05 ---------------------------------------------------------------------------
echo "[05-Plugin] Start"
cd 05-Plugin

#cd loadNpz # some problem
#make test > result.log
#cd ..

cd MultipleVersion
make test > result.log
cd ..

cd PluginProcess
make test > result.log
cd ..

#cd PluginRepository
# ???
#cd ..

cd useCuBLAS
make test > result.log
cd ..

cd useFP16
make test > result.log
cd ..

cd useINT8-PTQ
make test > result.log
cd ..

cd useINT8-QDQ
make test > result.log
cd ..

cd usePluginV2DynamicExt
make test > result.log
cd ..

cd usePluginV2Ext
make test > result.log
cd ..

cd usePluginV2IOExt
make test > result.log
cd ..

cd ..
echo "[05-Plugin] Finish"

# 06 ---------------------------------------------------------------------------
echo "[06-PluginAndParser] Start"
cd 06-PluginAndParser

if [[ $pt ]]; then
cd pyTorch-FailConvertNonZero
python3 main.py > result.log 2>&1
cd ..
fi

if [[ $pt ]]; then
cd pyTorch-LayerNorm
make test > result.log
cd ..
fi

if [ $tfMajorVersion = "1" ]; then
cd TensorFlow1-AddScalar
python3 main.py > result.log
cd ..
fi

#if [ $tfMajorVersion = "1" ]; then
#cd TensorFlow1-LayerNorm
#python3 main.py > result.log
#cd ..
#fi

if [ $tfMajorVersion = "2" ]; then
cd TensorFlow2-AddScalar
make test > result.log
cd ..
fi

#if [ $tfMajorVersion = "2" ]; then
#cd TensorFlow2-LayerNorm
#python3 main.py > result.log
#cd ..
#fi

cd ..
echo "[06-PluginAndParser] Finish"

# 07 ---------------------------------------------------------------------------
echo "[07-FrameworkTRT] Start"
cd 07-FrameworkTRT

if [ $tfMajorVersion = "1" ]; then
cd TensorFlow1-TFTRT
python3 main.py > result.log
cd ..
fi

#if [ $tfMajorVersion = "2" ]; then
#cd TensorFlow2-TFTRT
#python3 main.py > result.log
#cd ..
#fi

if [[ $pt ]]; then
cd Torch-TensorRT
python3 main.py > result.log
cd ..
fi

cd ..
echo "[07-FrameworkTRT] Finish"

# 08 ---------------------------------------------------------------------------
echo "[08-Tool] Start"
cd 08-Tool

cd NsightSystems
chmod +x ./command.sh
./command.sh > result.log
cd ..

cd OnnxGraphSurgeon
chmod +x ./command.sh
./command.sh > result.log
cd ..

cd OnnxRuntime
python3 main.py > result.log
cd ..

cd Polygraphy

    cd convertExample
    chmod +x command.sh
    ./command.sh > result.log
    cd ..

    #cd dataExample
    #chmod +x command.sh
    #./command.sh > result.log 2>&1
    #cd ..

    cd debugExample
    chmod +x command.sh
    ./command.sh > result.log 2>&1
    cd ..

    cd inspectExample
    chmod +x command.sh
    ./command.sh > result.log
    cd ..

    cd runExample
    chmod +x command.sh
    ./command.sh > result.log
    cd ..

    cd surgeonExample
    chmod +x command.sh
    ./command.sh > result.log
    cd ..

    cd templateExample
    chmod +x command.sh
    ./command.sh > result.log
    cd ..

    cd ..

# Examples of trex need several manual steps

cd trtexec
chmod +x ./command.sh
./command.sh > result.log
cd ..

cd ..
echo "[08-Tool] Finish"

# 09 ---------------------------------------------------------------------------
echo "[09-Advance] Start"
cd 09-Advance

cd AlgorithmSelector
python3 main.py > result.log
cd ..

#cd CreateExecutionContextWithoutDeviceMemory
#python3 main.py > result.log
#cd ..

cd CudaGraph
make test > result.log
cd ..

#cd EmptyTensor
#python3 main.py > result.log
#cd ..

cd EngineInspector
python3 main.py > result.log
cd ..

cd ErrorRecoder
python3 main-buildtime.py > result-buildtime.log
python3 main-runtime.py > result-runtime.log
cd ..

cd GPUAllocator
python3 main.py > result.log
cd ..

cd LabeledDimension
python3 main.py > result.log
cd ..

cd Logger
python3 main.py > result.log
cd ..

cd MultiContext
python3 MultiContext.py > result-MultiContext.log
python3 MultiContextV2.py > result-MultiContextV2.log
python3 MultiContext+CudaGraph.py > result-MultiContext+CudaGraph.log
cd ..

cd MultiOptimizationProfile
python3 main.py > result.log
cd ..

cd MultiStream
python3 main.py > result.log
cd ..

cd nvtx
make test > result.log
cd ..

cd Profiler
python3 main.py > result.log
cd ..

cd ProfilingVerbosity
python3 main.py > result.log
cd ..

cd Refit
python3 Refit-set_weights.py > result-Refit-set_weights.log
python3 Refit-set_named_weights.py > result-Refit-set_named_weights.log
python3 Refit-OnnxByParser.py > result-OnnxByParser.log
python3 Refit-OnnxByWeight.py > result-OnnxByWeight.log
cd ..

#cd Safety # only for QNX
#make test > result.log
#cd ..

cd Sparsity
python3 main.py > result.log
cd ..

cd StreamAndAsync
make test > result.log
cd ..

cd StrictType
python3 main.py > result.log
cd ..

cd TacticSource
python3 main.py > result.log
cd ..

cd TimingCache
python3 main.py >result.log
cd ..

cd ..
echo "[09-Advance] Finish"

# 10 ---------------------------------------------------------------------------
echo "[10-BestPractice] Start"
cd 10-BestPractice

cd AdjustReduceLayer
python3 main.py > resut.log 2>&1
cd ..

cd AlignSize
python3 main.py >result.log 2>&1
cd ..

cd ComputationInAdvance
python3 main.py > result.log
cd ..

cd Convert3DMMTo2DMM
python3 main.py > result.log 2>&1
cd ..

cd ConvertTranposeMultiplicationToConvolution
python3 main.py > result.log
cd ..

cd EliminateSqueezeUnsqueezeTranspose
python3 main.py > result.log
cd ..

cd IncreaseBatchSize
python3 main.py > result.log
cd ..

cd ..
echo "[10-BestPractice] Finish"

# 11 ---------------------------------------------------------------------------
echo "[11-ProblemSolving] Start"
cd 11-ProblemSolving

#cd ParameterCheckFailed
#make test > result.log 2>&1
#cd ..

cd SliceNodeWithBoolIO
python3 main.py > result.log 2>&1
cd ..

cd WeightsAreNotPermittedSinceTheyAreOfTypeInt32
python3 main.py > result.log
cd ..

cd WeightsHasCountXButYWasExpected
python3 main.py > result.log
cd ..

cd ..
echo "[11-ProblemSolving] Finish"

# 51 ---------------------------------------------------------------------------
echo "[51-Uncategorized] Start"
cd 51-Uncategorized

chmod +x getTensorRTVersion.sh
./getTensorRTVersion.sh

cd ..
echo "[51-Uncategorized] Finish"

echo "All test finish"
