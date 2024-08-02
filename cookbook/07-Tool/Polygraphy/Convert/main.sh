#/bin/bash

set -e
set -x
rm -rf *.json *.lock *.log *.onnx *.TimingCache *.trt
#clear

# 01-Parse ONNX file, build and save TensorRT engine without any more option
polygraphy convert \
    $TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx \
    --convert-to trt \
    --output ./model-trained-0.trt \
    > result-01.log 2>&1

# 02-Parse ONNX file, build and save TensorRT engine with more regular options (see Help.txt to get more parameters)
# + For the shape option, use "," to separate dimensions and use " " to separate the tensors (which is different from `trtexec`)
# + e.g. "--trt-min-shapes 'x:[16,320,256]' 'y:[8,4]' 'z:[]'"
# + Timing cache can be reused with `--load-timing-cache` during rebuild
# + More than one combination of `--trt-*-shapes` can be used for multiple optimization-profile
polygraphy convert \
    $TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx \
    --convert-to trt \
    --output ./model-trained.trt \
    --save-timing-cache model-trained.TimingCache \
    --save-tactics model-trained-tactics.json \
    --trt-min-shapes 'x:[1,1,28,28]' \
    --trt-opt-shapes 'x:[4,1,28,28]' \
    --trt-max-shapes 'x:[16,1,28,28]' \
    --fp16 \
    --memory-pool-limit workspace:1G \
    --builder-optimization-level 3 \
    --max-aux-streams 4 \
    --verbose \
    > result-02.log 2>&1

# 03-Convert a TensorRT network into a ONNX-like file for visualization in Netron
# Here is a error to convert model-trained.onnx:
# + ValueError: Could not infer attribute `reshape_dims` type from empty iterator), so we use another model
polygraphy convert \
    $TRT_COOKBOOK_PATH/00-Data/model/model-half-mnist.onnx \
    --convert-to onnx-like-trt-network \
    --output model-half-mnist-network.onnx \
    > result-03.log 2>&1

# 02-Parse ONNX file, build and save TensorRT engine in INT8 mode
# + We need to provide a script to load calibration data
# + INT8 cache can be reused with `---calibration-cache` during rebuild
polygraphy convert \
    $TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx \
    --convert-to trt \
    -output model-trained-int8.trt \
    --int8 \
    --data-loader-script data_loader.py \
    --calibration-cache model-trained.Int8Cache \
    > result-03.log 2>&1

echo "Finish"
