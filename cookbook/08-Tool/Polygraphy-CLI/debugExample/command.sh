clear

rm -rf ./*.onnx ./*.plan ./result.log ./*.txt

# 01-Create a ONNX graph with Onnx Graphsurgeon
python3 getOnnxModel-UnknowNode.py

# 02-Check error during converting model into TtensorRT
polygraphy debug reduce model-UnknowNode.onnx \
    --output="./reduced.onnx" \
    --check \
        polygraphy convert model-NonZero.onnx \
            -output="./model.plan"
