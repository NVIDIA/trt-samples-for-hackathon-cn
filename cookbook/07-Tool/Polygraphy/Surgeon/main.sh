#/bin/bash

set -e
set -x
rm -rf *.log *.onnx
#clear

# 00-Simplify the graph using polygraphy (the most common usegae)
# If we provide more information (such as static batch-size), we can see the ONNX is significantly simplified.
cp $TRT_COOKBOOK_PATH/00-Data/model/model-trained.onnx .
cp $TRT_COOKBOOK_PATH/00-Data/model/model-redundant.onnx .

polygraphy surgeon sanitize model-redundant.onnx \
    --cleanup \
    --fold-constant \
    --toposort \
    -o model-redundant-FC-DynamicBatch.onnx \
    > result-00.log

polygraphy surgeon sanitize model-redundant.onnx \
    --cleanup \
    --fold-constant \
    --toposort \
    --override-input-shapes inputT0:[7,2,3,4] \
    -o model-redundant-FC-StaticBatch.onnx \
    > result-01.log

# 02-Extract a subgraph from ONNX
polygraphy surgeon extract model-redundant.onnx \
    --inputs "inputT0:[nBS,2,3,4]:float32" \
    --outputs "RedundantModel-V-6-Concat-0:auto" \
    -o model-redundant-EX.onnx \
    > result-02.log

# 03-Insert a node into ONNX
polygraphy surgeon insert model-redundant.onnx \
    --name "MyNewNode" \
    --op "NewNode" \
    --inputs "RedundantModel-V-1-ReduceProd-0" \
    --outputs "RedundantModel-V-1-ReduceProd-0" \
    --attrs arg_int=31193 arg_float=3.14 arg_str=wili arg_list=[0,1,2] \
    -o model-redundant-IN.onnx \
    > result-03.log

# 04-Prune a ONNX to support sparisty in TensorRT
# In this example, our model is pruned successfully but not adpoted in engine finally due to performance.
polygraphy surgeon prune model-trained.onnx \
    -o model-trained-PR.onnx \
    > result-04.log

cat result-04.log | grep pruning

polygraphy run model-trained-PR.onnx \
    --trt \
    --sparse-weights \
    --verbose \
    | grep Sparsity

echo "Finish"
