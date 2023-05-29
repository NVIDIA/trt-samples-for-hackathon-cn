clear

rm -rf ./*.onnx ./*.plan ./result-*.log

# 01-Create a ONNX graph with Onnx Graphsurgeon
python3 getOnnxModel.py

# 02-Create a ONNX modifier with ONNX file above, so that we can use TensorRT API to ed it the network
polygraphy template trt-network model.onnx \
    --output modifyNetwork.py

# Once we finish the edition, we can use convert mode to build the TensorRT engine
polygraphy convert modifyNetwork.py \
    --output "./model.plan" \
    --model-type=trt-network-script \
    > result-BuildByTemplate.log
