clear

rm ./*.pb ./*.onnx ./*.plan ./result-*.txt

# 从 TensorFlow 创建一个 .onnx 用来做 polygraphy 的输入文件
python getShapeOperateOnnxModel.py

# 01 用上面的 .onnx 构建一个 TensorRT 引擎，使用 FP16精度，同时在 onnxruntime 和 TensorRT 中运行，对比结果
polygraphy surgeon sanitize model.onnx \
    --fold-constant \
    -o model-foldConstant.onnx \
    > result-surgeon.txt 

