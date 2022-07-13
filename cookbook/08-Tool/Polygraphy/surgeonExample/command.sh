clear

rm ./*.pb ./*.onnx ./*.plan ./result-*.txt

# 从 TensorFlow 创建一个 .onnx 用来做 polygraphy 的输入文件
python3 getShapeOperateOnnxModel.py

# 01 对上面的 .onnx 进行图优化
polygraphy surgeon sanitize model.onnx \
    --fold-constant \
    -o model-foldConstant.onnx \
    > result-surgeon.txt 

