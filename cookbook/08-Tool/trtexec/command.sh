clear

rm -rf ./*.onnx ./*.plan ./result-*.log

# 01-Create a ONNX graph with Onnx Graphsurgeon
python3 getOnnxModel.py

# 02-Build TensorRT engine using the ONNX file above
trtexec \
    --onnx=model.onnx \
    --minShapes=tensor-0:1x1x28x28 \
    --optShapes=tensor-0:4x1x28x28 \
    --maxShapes=tensor-0:16x1x28x28 \
    --memPoolSize=workspace:1024MiB \
    --saveEngine=model-FP32.plan \
    --shapes=tensor-0:4x1x28x28 \
    --verbose \
    > result-FP32.log 2>&1

# Notie: the format of parameters is different from polygrapy
# use "x" to separate dimensions and use "," to separate the input tensors, for example:
# --minShapes=tensor-0:16x320x256,tensor-1:16x320,tensor-2:16

# 02-Build TensorRT engine using the ONNX file above using FP16 mode
trtexec \
    --onnx=model.onnx \
    --minShapes=tensor-0:1x1x28x28 \
    --optShapes=tensor-0:4x1x28x28 \
    --maxShapes=tensor-0:16x1x28x28 \
    --memPoolSize=workspace:1024MiB \
    --saveEngine=model-FP16.plan \
    --shapes=tensor-0:4x1x28x28 \
    --verbose \
    --fp16 \
    > result-FP16.log

# 04-Load TensorRT engine built above to do inference
trtexec \
    --loadEngine=./model-FP32.plan \
    --shapes=tensor-0:4x1x28x28 \
    --verbose \
    > result-loadAndInference.log
    
# 05-Print information of the TensorRT engine built above (since TensorRT 8.4）
trtexec \
    --loadEngine=./model-FP32.plan \
    --shapes=tensor-0:4x1x28x28 \
    --verbose \
    --dumpLayerInfo \
    --exportLayerInfo="./modelInformation.log" \
    > result-PrintInformation.log
