clear

rm -rf ./*.onnx ./*.plan ./result-*.log ./*.tactic

# 01-Create a ONNX graph with Onnx Graphsurgeon
python3 getOnnxModel.py

# 02-Build TensorRT engine using the ONNX file above using FP16 mode, and compare the output of the network between Onnxruntime and TensorRT
polygraphy convert model.onnx \
    --workspace 1000000000 \
    --output "./model-FP16.plan" \
    --fp16 \
    --verbose \
    --trt-min-shapes 'tensor-0:[1,1,28,28]' \
    --trt-opt-shapes 'tensor-0:[4,1,28,28]' \
    --trt-max-shapes 'tensor-0:[16,1,28,28]' \
    --input-shapes   'tensor-0:[4,1,28,28]' \
    > result-convert-FP16.log 2>&1
    
# 03-Build TensorRT engine using the ONNX file above, and compare the output of each layer between Onnxruntime and TensorRT
polygraphy convert model.onnx \
    --workspace 1000000000 \
    --output "./model-FP32-MarkAll.plan" \
    --verbose \
    --trt-outputs mark all \
    --trt-min-shapes 'tensor-0:[1,1,28,28]' \
    --trt-opt-shapes 'tensor-0:[4,1,28,28]' \
    --trt-max-shapes 'tensor-0:[16,1,28,28]' \
    --input-shapes   'tensor-0:[4,1,28,28]'
    > result-convert-FP32-MarkAll.log 2>&1

# 04-Build TensorRT engine using the ONNX file above, and save the tactics
polygraphy convert model.onnx \
    --workspace 1000000000 \
    --output "./model-FP32-tactic01.plan" \
    --quiet \
    --silent \
    --save-tactics "./model-FP32.tacitc" \
    --trt-min-shapes 'tensor-0:[1,1,28,28]' \
    --trt-opt-shapes 'tensor-0:[4,1,28,28]' \
    --trt-max-shapes 'tensor-0:[16,1,28,28]' \
    --input-shapes   'tensor-0:[4,1,28,28]' \
    > result-convert-saveTactic.log 2>&1

# 05-Build TensorRT engine reusing the tactic file above
polygraphy convert model.onnx \
    --workspace 1000000000 \
    --output "./model-FP32-tactic02.plan" \
    --quiet \
    --silent \
    --load-tactics "./model-FP32.tacitc" \
    --trt-min-shapes 'tensor-0:[1,1,28,28]' \
    --trt-opt-shapes 'tensor-0:[4,1,28,28]' \
    --trt-max-shapes 'tensor-0:[16,1,28,28]' \
    --input-shapes   'tensor-0:[4,1,28,28]' \
    > result-convert-loadTactic.log 2>&1
