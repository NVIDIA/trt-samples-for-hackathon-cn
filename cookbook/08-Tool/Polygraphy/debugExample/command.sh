clear

rm -rf ./*.onnx ./*.plan ./result.log ./*.txt

# 创建 .onnx 模型
python3 getOnnxModel-NonZero.py

# 01 检查 convert 过程中出现的错误
polygraphy debug reduce model-NonZero.onnx \
    --output="./reduced.onnx" \
    --check \
        polygraphy convert model-NonZero.onnx \
            -output="./model.plan"
