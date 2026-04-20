# 03-Workflow

+ Common workflow of using TensorRT from DL frameworks.

## Workflow of JAX -> ONNX -> TensorRT

+ A workflow of: train model in JAX, export model to ONNX through `jax2tf` + `tf2onnx`, parse ONNX in TensorRT, build TensorRT engine and do inference.

## Workflow of Mindspore -> ONNX -> TensorRT

+ A workflow of: train model in Mindspore, export model to ONNX, parse ONNX in TensorRT, build TensorRT engine and do inference.

## Workflow of OneFlow -> ONNX -> TensorRT

+ A workflow of: train model in OneFlow, export model to ONNX, parse ONNX in TensorRT, build TensorRT engine and do inference.

## Workflow of Paddlepaddle -> ONNX -> TensorRT

+ A workflow of: export trained model from Paddlepaddle to ONNX, parse ONNX in TensorRT, build TensorRT engine and do inference.

## Workflow of TensorFlow2 -> ONNX -> TensorRT

+ A workflow of: export trained model from TensorFlow2 to ONNX, parse ONNX in TensorRT, build TensorRT engine and do inference.

## Workflow of pyTorch -> ONNX -> TensorRT

+ A workflow of: export trained model from pyTorch to ONNX, parse ONNX in TensorRT, build TensorRT engine and do inference.

## Workflow of pyTorch -> TensorRT

+ A workflow of: rebuild model in TensorRT with exported weights, build TensorRT engine and do inference.
