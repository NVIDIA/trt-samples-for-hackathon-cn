clear

rm -rf ./*.onnx ./*.plan ./result-*.log ./*.tactic

# 从 TensorFlow 创建一个 .onnx 用来做 polygraphy 的输入文件
python3 getOnnxModel.py

# 01 用上面的 .onnx 构建一个 TensorRT 引擎，使用 FP16精度，同时在 onnxruntime 和 TensorRT 中运行，对比结果
polygraphy convert model.onnx \
    --pool-limit workspace:1000000000 \
    --output "./model-FP16.plan" \
    --fp16 \
    --verbose \
    --trt-min-shapes 'tensor-0:[1,1,28,28]' \
    --trt-opt-shapes 'tensor-0:[4,1,28,28]' \
    --trt-max-shapes 'tensor-0:[16,1,28,28]' \
    --input-shapes   'tensor-0:[4,1,28,28]' \
    > result-run-FP16.log
    
# 02 用上面的 .onnx 构建一个 TensorRT 引擎，输出所有层的计算结果作对比
polygraphy convert model.onnx \
    --pool-limit workspace:1000000000 \
    --output "./model-FP32-MarkAll.plan" \
    --verbose \
    --trt-outputs mark all \
    --trt-min-shapes 'tensor-0:[1,1,28,28]' \
    --trt-opt-shapes 'tensor-0:[4,1,28,28]' \
    --trt-max-shapes 'tensor-0:[16,1,28,28]' \
    --input-shapes   'tensor-0:[4,1,28,28]'
    > result-run-FP32-MarkAll.log

# 03 用上面的 .onnx 构建一个 TensorRT 引擎，并保存 tactic 信息
polygraphy convert model.onnx \
    --pool-limit workspace:1000000000 \
    --output "./model-FP32-tactic01.plan" \
    --quiet \
    --silent \
    --save-tactics "./model-FP32.tacitc" \
    --trt-min-shapes 'tensor-0:[1,1,28,28]' \
    --trt-opt-shapes 'tensor-0:[4,1,28,28]' \
    --trt-max-shapes 'tensor-0:[16,1,28,28]' \
    --input-shapes   'tensor-0:[4,1,28,28]' \
    > result-saveTactic.log

# 03 用上面的 .onnx 构建一个 TensorRT 引擎，并加载已经存在的 tactic 重建一模一样的引擎
polygraphy convert model.onnx \
    --pool-limit workspace:1000000000 \
    --output "./model-FP32-tactic02.plan" \
    --quiet \
    --silent \
    --load-tactics "./model-FP32.tacitc" \
    --trt-min-shapes 'tensor-0:[1,1,28,28]' \
    --trt-opt-shapes 'tensor-0:[4,1,28,28]' \
    --trt-max-shapes 'tensor-0:[16,1,28,28]' \
    --input-shapes   'tensor-0:[4,1,28,28]' \
    > result-loadTactic.log
