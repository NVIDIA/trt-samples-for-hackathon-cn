clear

rm ./*.pb ./*.onnx ./*.plan ./result-*.txt

# 从 TensorFlow 创建一个 .onnx 用来做 polygraphy 的输入文件
python getOnnxModel.py

# 01 用上面的 .onnx 构建一个 TensorRT 引擎，同时在 onnxruntime 和 TensorRT 中运行，对比结果
polygraphy run model.onnx \
    --trt --onnxrt \
    --workspace 1000000000 \
    --save-engine=model-FP32.plan \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'x:0:[1,1,28,28]' \
    --trt-opt-shapes 'x:0:[4,1,28,28]' \
    --trt-max-shapes 'x:0:[16,1,28,28]' \
    --input-shapes   'x:0:[4,1,28,28]' \
    > result-01.txt

# 注意参数名和格式跟 trtexec 不一样，多个形状之间用空格分隔，如：
# --trt-max-shapes 'input0:[16,320,256]' 'input1:[16，320]' 'input2:[16]'
    
# 02 同上，但是使用 FP16 精度，并输出所有层的计算结果作对比
polygraphy run model.onnx \
    --trt --onnxrt \
    --workspace 1000000000 \
    --save-engine=model-FP32.plan \
    --atol 1e-3 --rtol 1e-3 \
    --fp16 \
    --verbose \
    --onnx-outputs mark all \
    --trt-outputs mark all \
    --trt-min-shapes 'x:0:[1,1,28,28]' \
    --trt-opt-shapes 'x:0:[4,1,28,28]' \
    --trt-max-shapes 'x:0:[16,1,28,28]' \
    --input-shapes   'x:0:[4,1,28,28]'
    > result-02.txt
    
    
    
    
