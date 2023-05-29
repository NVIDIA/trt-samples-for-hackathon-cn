clear

rm -rf ./*.onnx ./*.plan ./result-*.log

# 01-Create a ONNX graph with Onnx Graphsurgeon
python3 getShapeOperateOnnxModel.py

# 02-Simplify the graph using polygraphy
polygraphy surgeon sanitize model.onnx \
    --fold-constant \
    -o model-foldConstant.onnx \
    > result-surgeon.log
