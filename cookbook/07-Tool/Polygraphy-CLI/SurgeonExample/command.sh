clear

rm -rf ./*.onnx ./*.plan ./*.log

# 00-Create a ONNX graph with Onnx Graphsurgeon
python3 getOnnxModel.py

# 01-Simplify the graph using polygraphy
polygraphy surgeon sanitize modelA.onnx \
    --fold-constant \
    -o modelA-FoldConstant.onnx \
    > result-01.log
