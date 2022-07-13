clear

rm ./*.pb ./*.onnx ./*.plan ./result-*.txt

# 从 TensorFlow 创建一个 .onnx 用来做 polygraphy 的输入文件
python3 getOnnxModel.py

# 01 用上面的 .onnx 构建一个网络修改器，可以在其中使用 TensorRT 的 API 对网络进行修改
polygraphy template trt-network model.onnx \
    --output modifyNetwork.py

# 假装我们修改完了，继续使用 convert 工具把模型转成 .plan 
polygraphy convert modifyNetwork.py \
    --output "./model.plan" \
    --model-type=trt-network-script


