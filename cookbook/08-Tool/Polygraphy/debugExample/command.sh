clear

rm ./*.pb ./*.onnx ./*.plan ./result-*.txt

# 从 TensorFlow 创建一个 .onnx 用来做 polygraphy 的输入文件
python getOnnxModel-NonZero.py

# 01 检查 convert 过程中出现的错误
polygraphy debug reduce model-NonZero.onnx \
    --output="./reduced.onnx" \
    --check \
        polygraphy convert model-NonZero.onnx \
            -output="./model.plan"
    > result-debug.txt 2>&1


