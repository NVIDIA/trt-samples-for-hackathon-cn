rm -rf ./*.onnx ./*.log

# Create a ONNX graph with Onnx Graphsurgeon
python3 01-CreateModel.py

# Check the consistency of the model in onnxruntime and TensorRT
polygraphy run model-01.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'tensor0:[1,3,64,64]' \
    --trt-opt-shapes 'tensor0:[4,3,64,64]' \
    --trt-max-shapes 'tensor0:[16,3,64,64]' \
    --input-shapes   'tensor0:[4,3,64,64]' \
    > result-01.log

# Add a new node into the graph
python3 02-AddNode.py

polygraphy run model-02-01.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'tensor0:[1,3,64,64]' \
    --trt-opt-shapes 'tensor0:[4,3,64,64]' \
    --trt-max-shapes 'tensor0:[16,3,64,64]' \
    --input-shapes   'tensor0:[4,3,64,64]' \
    > result-02.log

polygraphy run model-02-02.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'tensor0:[1,3,64,64]' \
    --trt-opt-shapes 'tensor0:[4,3,64,64]' \
    --trt-max-shapes 'tensor0:[16,3,64,64]' \
    --input-shapes   'tensor0:[4,3,64,64]' \
    >> result-02.log

# Remove a node in the graph
python3 03-RemoveNode.py

polygraphy run model-03-01.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'tensor0:[1,3,64,64]' \
    --trt-opt-shapes 'tensor0:[4,3,64,64]' \
    --trt-max-shapes 'tensor0:[16,3,64,64]' \
    --input-shapes   'tensor0:[4,3,64,64]' \
    > result-03.log

polygraphy run model-03-02.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'tensor0:[1,3,64,64]' \
    --trt-opt-shapes 'tensor0:[4,3,64,64]' \
    --trt-max-shapes 'tensor0:[16,3,64,64]' \
    --input-shapes   'tensor0:[4,3,64,64]' \
    >> result-03.log

# Replace a node in the graph
python3 04-ReplaceNode.py

polygraphy run model-04-01.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'tensor0:[1,3,64,64]' \
    --trt-opt-shapes 'tensor0:[4,3,64,64]' \
    --trt-max-shapes 'tensor0:[16,3,64,64]' \
    --input-shapes   'tensor0:[4,3,64,64]' \
    > result-04.log

polygraphy run model-04-02.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'tensor0:[1,3,64,64]' \
    --trt-opt-shapes 'tensor0:[4,3,64,64]' \
    --trt-max-shapes 'tensor0:[16,3,64,64]' \
    --input-shapes   'tensor0:[4,3,64,64]' \
    >> result-04.log

polygraphy run model-04-03.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'tensor0:[1,3,64,64]' \
    --trt-opt-shapes 'tensor0:[4,3,64,64]' \
    --trt-max-shapes 'tensor0:[16,3,64,64]' \
    --input-shapes   'tensor0:[4,3,64,64]' \
    >> result-04.log

# print the information of the graph
python3 05-PrintGraphInformation.py > result-05.log

# Constant-Fold, Cleanup and Toposort of the graph
python3 06-Fold.py > result-06.log

polygraphy run model-06-04.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'tensor0:[1,3,64,64]'\
    --trt-opt-shapes 'tensor0:[4,3,64,64]'\
    --trt-max-shapes 'tensor0:[16,3,64,64]'\
    --input-shapes   'tensor0:[4,3,64,64]'\
    >> result-06.log

# Shape related operation
python3 07-ShapeOperationAndSimplify.py

polygraphy run model-07-01.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'tensor0:[1,3,1,5]' \
    --trt-opt-shapes 'tensor0:[2,3,4,5]' \
    --trt-max-shapes 'tensor0:[4,3,16,5]' \
    --input-shapes   'tensor0:[4,3,64,64]' \
    > result-07.log

polygraphy run model-07-02.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --input-shapes   'tensor0:[2,3,4,5]' \
    >> result-07.log

# Clip the graph into subgraph
python3 08-IsolateSubgraph.py

polygraphy run model-08-01.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'tensor0:[1,3,64,64]' \
    --trt-opt-shapes 'tensor0:[4,3,64,64]' \
    --trt-max-shapes 'tensor0:[16,3,64,64]' \
    --input-shapes   'tensor0:[4,3,64,64]' \
    > result-08.log

polygraphy run model-08-02.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'tensor1:[1,3,64,64]' \
    --trt-opt-shapes 'tensor1:[4,3,64,64]' \
    --trt-max-shapes 'tensor1:[16,3,64,64]' \
    --input-shapes   'tensor1:[4,3,64,64]' \
    >> result-08.log

# Use gs.Graph.register() to create the graph
python3 09-BuildModelWithAPI.py

polygraphy run model-09-01.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --input-shapes 'tensor0:[64,64]' \
    > result-09.log

polygraphy run model-09-02.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --input-shapes 'tensor0:[64,64]' \
    >> result-09.log

