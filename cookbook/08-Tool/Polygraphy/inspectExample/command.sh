clear

rm -rf ./*.onnx ./*.plan ./result*.log ./polygraphy_capability_dumps/

# 创建 .onnx 模型
python3 getOnnxModel.py

# 导出上面 .onnx 的详细信息
polygraphy inspect model model.onnx \
    --mode=full \
    > result-inspectOnnxModel.log

# 用上面 .onnx 生成一个 .plan 及其相应的 tactics 用于后续分析
polygraphy run model.onnx \
    --trt \
    --pool-limit workspace:1000000000 \
    --save-engine="./model.plan" \
    --save-tactics="./model.tactic" \
    --save-inputs="./model-input.log" \
    --save-outputs="./model-output.log" \
    --silent \
    --trt-min-shapes 'tensor-0:[1,1,28,28]' \
    --trt-opt-shapes 'tensor-0:[4,1,28,28]' \
    --trt-max-shapes 'tensor-0:[16,1,28,28]' \
    --input-shapes   'tensor-0:[4,1,28,28]'
    
# 导出上面 .plan 的详细信息（since TensorRT 8.2）
polygraphy inspect model model.plan \
    --mode=full \
    > result-inspectPlanModel.log

# 导出上面 .tactic 的信息，就是 json 转 text
polygraphy inspect tactics model.tactic \
    > result-inspectPlanTactic.log
    
# 导出上面 .plan 输入输出数据信息
polygraphy inspect data model-input.log \
    > result-inspectPlanInputData.log

polygraphy inspect data model-output.log \
    > result-inspectPlanOutputData.log
    
# 确认 TensorRT 是否完全原生支持该 .onnx
polygraphy inspect capability model.onnx > result-inspectCapability.log

# 生成一个含有 TensorRT 不原生支持的 .onnx，再次用 inspect capability 来确认
python3 getOnnxModel-NonZero.py

polygraphy inspect capability model-NonZero.onnx > result-NonZero.log
# 产生目录 .results，包含网络分析信息和支持的子图(.onnx)、不支持的子图(.onnx)

