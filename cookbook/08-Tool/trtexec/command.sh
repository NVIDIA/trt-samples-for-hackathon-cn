clear

rm -rf ./*.onnx ./*.plan ./result-*.log

# 01-从 TensorFlow 创建一个 .onnx 用来做 trtexec 的输入文件
python3 getOnnxModel.py

# 02-用上面的 .onnx 构建一个 TensorRT 引擎并作推理
trtexec \
    --onnx=model.onnx \
    --minShapes=tensor-0:1x1x28x28 \
    --optShapes=tensor-0:4x1x28x28 \
    --maxShapes=tensor-0:16x1x28x28 \
    --memPoolSize=workspace:1024MiB \
    --saveEngine=model-FP32.plan \
    --shapes=tensor-0:4x1x28x28 \
    --verbose \
    > result-FP32.log

# 注意参数名和格式跟 polygrapy 不一样，多个形状之间用逗号分隔，如：
# --minShapes=tensor-0:16x320x256,tensor-1:16x320,tensor-2:16

# 03-用上面的 .onnx 构建一个 TensorRT 引擎并作推理，使用 FP16 模式
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

# 04-读取上面构建的 result-FP32.plan 并作推理
trtexec \
    --loadEngine=./model-FP32.plan \
    --shapes=tensor-0:4x1x28x28 \
    --verbose \
    > result-loadAndInference.log
    
# 05-读取上面构建的 result-FP32.plan 打印引擎的详细信息（since TRT8.4）
trtexec \
    --loadEngine=./model-FP32.plan \
    --shapes=tensor-0:4x1x28x28 \
    --verbose \
    --dumpLayerInfo \
    --exportLayerInfo="./modelInformation.log" \
    > result-PrintInformation.log
