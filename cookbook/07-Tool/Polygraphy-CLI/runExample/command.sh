clear

rm -rf ./*.onnx ./*.plan ./result-*.log

# 01-Create a ONNX graph with Onnx Graphsurgeon
python3 getOnnxModel.py

# 02-Build TensorRT engine using the ONNX file above using FP16 mode, and compare the output of the network between Onnxruntime and TensorRT
polygraphy run model.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --save-engine=model-FP16.plan \
    --atol 1e-3 --rtol 1e-3 \
    --fp16 \
    --verbose \
    --trt-min-shapes 'tensor-0:[1,1,28,28]' \
    --trt-opt-shapes 'tensor-0:[4,1,28,28]' \
    --trt-max-shapes 'tensor-0:[16,1,28,28]' \
    --input-shapes   'tensor-0:[4,1,28,28]' \
    > result-run-FP16.log 2>&1

# Notie: the format of parameters is different from polygrapy
# use "," to separate dimensions and use " " to separate the input tensors, for example:
# --trt-min-shapes 'input0:[16,320,256]' 'input1:[16，320]' 'input2:[16]'

# 03-Build TensorRT engine using the ONNX file above, and compare the output of each layer between Onnxruntime and TensorRT

polygraphy run model.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --save-engine=model-FP32-MarkAll.plan \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --onnx-outputs mark all \
    --trt-outputs mark all \
    --trt-min-shapes 'tensor-0:[1,1,28,28]' \
    --trt-opt-shapes 'tensor-0:[4,1,28,28]' \
    --trt-max-shapes 'tensor-0:[16,1,28,28]' \
    --input-shapes   'tensor-0:[4,1,28,28]'
    > result-run-FP32-MarkAll.log 2>&1

# 04-Build TensorRT engine using the ONNX file above, and save the tactics, data of input / output, and the script created by polygraphy
polygraphy run model.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --save-engine=model-FP32.plan \
    --save-tactics="./model.tactic" \
    --save-inputs="./model-input.log" \
    --save-outputs="./model-output.log" \
    --atol 1e-3 --rtol 1e-3 \
    --silent \
    --gen-script="./polygraphyRun.py" \
    --trt-min-shapes 'tensor-0:[1,1,28,28]' \
    --trt-opt-shapes 'tensor-0:[4,1,28,28]' \
    --trt-max-shapes 'tensor-0:[16,1,28,28]' \
    --input-shapes   'tensor-0:[4,1,28,28]' \
    > result-run-FP32-Save.log 2>&1
