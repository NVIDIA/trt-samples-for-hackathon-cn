# Workflow of Mindspore -> ONNX -> TensorRT

+ A workflow of: train model in Mindspore, export model to ONNX, parse ONNX in TensorRT, build TensorRT engine and do inference.

+ Need package `mindspore` which are not included in the docker image.

+ Steps to run.

```bash
python3 main.py
```
