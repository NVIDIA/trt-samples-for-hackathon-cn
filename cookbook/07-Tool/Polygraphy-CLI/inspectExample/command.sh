clear

rm -rf ./*.onnx ./*.plan ./result*.log ./polygraphy_capability_dumps/

# 01-Create a ONNX graph with Onnx Graphsurgeon
python3 getOnnxModel.py

# 02-export detailed information of the ONNX file above
polygraphy inspect model model.onnx \
    --mode=full \
    > result-inspectOnnxModel.log

# 03-Build TensorRT engine using the ONNX file above, and save the tactics
polygraphy run model.onnx \
    --trt \
    --workspace 1000000000 \
    --save-engine="./model.plan" \
    --save-tactics="./model.tactic" \
    --save-inputs="./model-input.log" \
    --save-outputs="./model-output.log" \
    --silent \
    --trt-min-shapes 'tensor-0:[1,1,28,28]' \
    --trt-opt-shapes 'tensor-0:[4,1,28,28]' \
    --trt-max-shapes 'tensor-0:[16,1,28,28]' \
    --input-shapes   'tensor-0:[4,1,28,28]'
    
# 04-export detailed information of the TensorRT engine above（since TensorRT 8.2）
polygraphy inspect model model.plan \
    --mode=full \
    > result-inspectPlanModel.log

# 05-export information of tactics, json -> txt
polygraphy inspect tactics model.tactic \
    > result-inspectPlanTactic.log
    
# 06-export information of input / output data
polygraphy inspect data model-input.log \
    > result-inspectPlanInputData.log

polygraphy inspect data model-output.log \
    > result-inspectPlanOutputData.log
    
# 07-Judge whether a ONNX file is supported by TensorRT natively
polygraphy inspect capability model.onnx > result-inspectCapability.log

# 08-# Create a ONNX graph with Onnx Graphsurgeon which TensorRT does not support natively, and use polygraphy to jusge
python3 getOnnxModel-NonZero.py
polygraphy inspect capability model-NonZero.onnx > result-NonZero.log
# the output directory ".result" contains information of the graph and the subgraphs supported / unsupported by TensorRT natively

