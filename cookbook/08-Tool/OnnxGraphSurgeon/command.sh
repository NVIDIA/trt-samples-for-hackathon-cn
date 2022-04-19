rm ./*.pb
rm ./*.onnx
rm ./*.txt

# 使用 onnx-graphsurgeon 构造张量和节点来创建一个模型
python 01-CreateModel.py

# 检验模型在 onnxruntime 和 TensorRT 中的一致性
polygraphy run model-01.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'x:0:[1,3,64,64]' \
    --trt-opt-shapes 'x:0:[4,3,64,64]' \
    --trt-max-shapes 'x:0:[16,3,64,64]' \
    --input-shapes   'x:0:[4,3,64,64]' \
    > result-01.txt

# 向模型中添加一个节点
python 02-AddNode.py > result-02.txt

polygraphy run model-02-01.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'x:0:[1,3,64,64]' \
    --trt-opt-shapes 'x:0:[4,3,64,64]' \
    --trt-max-shapes 'x:0:[16,3,64,64]' \
    --input-shapes   'x:0:[4,3,64,64]' \
    > result-02.txt

polygraphy run model-02-02.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'x:0:[1,3,64,64]' \
    --trt-opt-shapes 'x:0:[4,3,64,64]' \
    --trt-max-shapes 'x:0:[16,3,64,64]' \
    --input-shapes   'x:0:[4,3,64,64]' \
    >> result-02.txt

# 从模型中删除一个节点
python 03-RemoveNode.py > result-03.txt

polygraphy run model-03-01.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'x:0:[1,3,64,64]' \
    --trt-opt-shapes 'x:0:[4,3,64,64]' \
    --trt-max-shapes 'x:0:[16,3,64,64]' \
    --input-shapes   'x:0:[4,3,64,64]' \
    > result-03.txt

polygraphy run model-03-02.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'x:0:[1,3,64,64]' \
    --trt-opt-shapes 'x:0:[4,3,64,64]' \
    --trt-max-shapes 'x:0:[16,3,64,64]' \
    --input-shapes   'x:0:[4,3,64,64]' \
    >> result-03.txt

# 模型节点替换
python 04-ReplaceNode.py > result-04.txt

polygraphy run model-04-01.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'x:0:[1,3,64,64]' \
    --trt-opt-shapes 'x:0:[4,3,64,64]' \
    --trt-max-shapes 'x:0:[16,3,64,64]' \
    --input-shapes   'x:0:[4,3,64,64]' \
    > result-04.txt

polygraphy run model-04-02.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'x:0:[1,3,64,64]' \
    --trt-opt-shapes 'x:0:[4,3,64,64]' \
    --trt-max-shapes 'x:0:[16,3,64,64]' \
    --input-shapes   'x:0:[4,3,64,64]' \
    >> result-04.txt

polygraphy run model-04-03.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'x:0:[1,3,64,64]' \
    --trt-opt-shapes 'x:0:[4,3,64,64]' \
    --trt-max-shapes 'x:0:[16,3,64,64]' \
    --input-shapes   'x:0:[4,3,64,64]' \
    >> result-04.txt

# 打印计算图信息，包括遍历节点和遍历张量
python 05-PrintGraphInformation.py > result-05.txt

# 使用常量折叠（fold_constants）清理（cleanup）和拓扑排序（toposort）
python 06-Fold.py > result-06.txt

polygraphy run model-06-04.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'tensor0:[1,3,64,64]'\
    --trt-opt-shapes 'tensor0:[4,3,64,64]'\
    --trt-max-shapes 'tensor0:[16,3,64,64]'\
    --input-shapes   'tensor0:[4,3,64,64]'\
    > result-06.txt

# 形状操作
python 07-ShapeOperationAndSimplify.py > result-07.txt

polygraphy run model-07-01.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'x:0:[1,3,64,64]' \
    --trt-opt-shapes 'x:0:[4,3,64,64]' \
    --trt-max-shapes 'x:0:[16,3,64,64]' \
    --input-shapes   'x:0:[4,3,64,64]' \
    > result-07.txt

polygraphy run model-07-02.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'x:0:[1,3,64,64]' \
    --trt-opt-shapes 'x:0:[4,3,64,64]' \
    --trt-max-shapes 'x:0:[16,3,64,64]' \
    --input-shapes   'x:0:[4,3,64,64]' \
    >> result-07.txt

# 分割子图
python 08-IsolateSubgraph.py > result-08.txt

polygraphy run model-08-01.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'x:0:[1,3,64,64]' \
    --trt-opt-shapes 'x:0:[4,3,64,64]' \
    --trt-max-shapes 'x:0:[16,3,64,64]' \
    --input-shapes   'x:0:[4,3,64,64]' \
    > result-08.txt

polygraphy run model-08-02.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'x:0:[1,3,64,64]' \
    --trt-opt-shapes 'x:0:[4,3,64,64]' \
    --trt-max-shapes 'x:0:[16,3,64,64]' \
    --input-shapes   'x:0:[4,3,64,64]' \
    >> result-08.txt

# 使用 gs.Graph.register() 创建模型
python 09-BuildModelWithAPI.py > result-09.txt

polygraphy run model-09-01.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'x:0:[1,3,64,64]' \
    --trt-opt-shapes 'x:0:[4,3,64,64]' \
    --trt-max-shapes 'x:0:[16,3,64,64]' \
    --input-shapes   'x:0:[4,3,64,64]' \
    > result-09.txt

polygraphy run model-09-02.onnx \
    --onnxrt --trt \
    --workspace 1000000000 \
    --atol 1e-3 --rtol 1e-3 \
    --verbose \
    --trt-min-shapes 'x:0:[1,3,64,64]' \
    --trt-opt-shapes 'x:0:[4,3,64,64]' \
    --trt-max-shapes 'x:0:[16,3,64,64]' \
    --input-shapes   'x:0:[4,3,64,64]' \
    >> result-09.txt
