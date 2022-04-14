clear

rm ./*.pb ./*.onnx ./*.plan ./result-*.txt

# 从 TensorFlow 创建一个 .onnx 用来做 polygraphy 的输入文件
python getOnnxModel.py

# 导出上面 .onnx 的详细信息
polygraphy inspect model model.onnx \
    --mode=full \
    > result-inspectOnnxModel.txt

# 用上面 .onnx 生成一个 .plan 及其相应的 tactics 用于后续分析
polygraphy run model.onnx \
    --trt \
    --workspace 1000000000 \
    --save-engine="./model.plan" \
    --save-tactics="./model.tactic" \
    --save-inputs="./model-input.txt" \
    --save-outputs="./model-output.txt" \
    --silent \
    --trt-min-shapes 'x:0:[1,1,28,28]' \
    --trt-opt-shapes 'x:0:[4,1,28,28]' \
    --trt-max-shapes 'x:0:[16,1,28,28]' \
    --input-shapes   'x:0:[4,1,28,28]'
    
# 导出上面 .plan 的详细信息（要求 TensorRT >= 8.2）
polygraphy inspect model model.plan \
    --mode=full \
    > result-inspectPlanModel.txt

# 导出上面 .tactic 的信息，就是 json 转 text
polygraphy inspect tactics model.tactic \
    > result-inspectPlanTactic.txt
    
# 导出上面 .plan 输入输出数据信息
polygraphy inspect data model-input.txt \
    > result-inspectPlanInputData.txt

polygraphy inspect data model-output.txt \
    > result-inspectPlanOutputData.txt
    
# 确认 TensorRT 是否完全原生支持该 .onnx
polygraphy inspect capability model.onnx
# 输出信息：[I] Graph is fully supported by TensorRT; Will not generate subgraphs.

# 生成一个含有 TensorRT 不原生支持的 .onnx，再次用 inspect capability 来确认
python getOnnxModel-NonZero.py

polygraphy inspect capability model-NonZero.onnx > result-NonZero.txt
# 产生目录 .results，包含网络分析信息和支持的子图(.onnx)、不支持的子图(.onnx)

