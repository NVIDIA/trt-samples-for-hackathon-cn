clear

rm ./*.pb ./*.onnx ./*.plan ./result-*.txt

# 从 TensorFlow 创建一个 .onnx 用来做 trtexec 的输入文件
python3 getOnnxModel.py

# 用上面的 .onnx 构建一个 TensorRT 引擎并作推理
trtexec \
    --onnx=model.onnx \
    --minShapes=x:0:1x1x28x28 \
    --optShapes=x:0:4x1x28x28 \
    --maxShapes=x:0:16x1x28x28 \
    --workspace=1024 \
    --saveEngine=model-FP32.plan \
    --shapes=x:0:4x1x28x28 \
    --verbose \
    > result-FP32.txt

# 用上面的 .onnx 构建一个 TensorRT 引擎并作推理，使用 FP16 模式
trtexec \
    --onnx=model.onnx \
    --minShapes=x:0:1x1x28x28 \
    --optShapes=x:0:4x1x28x28 \
    --maxShapes=x:0:16x1x28x28 \
    --workspace=1024 \
    --saveEngine=model-FP16.plan \
    --shapes=x:0:4x1x28x28 \
    --verbose \
    --fp16 \
    > result-FP16.txt

# 读取上面构建的 result-FP32.plan 并作推理
trtexec \
    --loadEngine=./model-FP32.plan \
    --shapes=x:0:4x1x28x28 \
    --verbose \
    > result-load-FP32.txt
