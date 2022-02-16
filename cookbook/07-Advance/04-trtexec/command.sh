clear

trtexec \
    --onnx=model-NCHW-backup.onnx \
    --minShapes=x:0:1x1x28x28 \
    --optShapes=x:0:4x1x28x28 \
    --maxShapes=x:0:16x1x28x28 \
    --workspace=1024 \
    --saveEngine=model.trt \
    --shapes=x:0:4x1x28x28 \
    --verbose \
    > output-fp32.txt

trtexec \
    --onnx=model-NCHW-backup.onnx \
    --minShapes=x:0:1x1x28x28 \
    --optShapes=x:0:4x1x28x28 \
    --maxShapes=x:0:16x1x28x28 \
    --workspace=1024 \
    --saveEngine=model.trt \
    --shapes=x:0:4x1x28x28 \
    --verbose \
    --fp16 \
    > output-fp16.txt    

