clear

rm ./*.pb ./*.onnx ./*.trt ./result-*.txt

python3 getOnnxModel.py

trtexec \
    --onnx=model.onnx \
    --minShapes=x:0:1x1x28x28 \
    --optShapes=x:0:4x1x28x28 \
    --maxShapes=x:0:16x1x28x28 \
    --workspace=1024 \
    --saveEngine=model-FP32.trt \
    --shapes=x:0:4x1x28x28 \
    --verbose \
    > result-FP32.txt

trtexec \
    --onnx=model.onnx \
    --minShapes=x:0:1x1x28x28 \
    --optShapes=x:0:4x1x28x28 \
    --maxShapes=x:0:16x1x28x28 \
    --workspace=1024 \
    --saveEngine=model-FP16.trt \
    --shapes=x:0:4x1x28x28 \
    --verbose \
    --fp16 \
    > result-FP16.txt

