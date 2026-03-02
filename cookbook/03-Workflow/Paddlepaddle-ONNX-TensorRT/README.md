# Workflow of Paddlepaddle -> ONNX -> TensorRT

+ A workflow of: export trained model from Paddlepaddle to ONNX, parse ONNX in TensorRT, build TensorRT engine and do inference.

+ Need package `paddlepaddle` and `paddle2onnx` which are not included in the docker image.

+ Steps to run.

```bash
python3 main.py
```
