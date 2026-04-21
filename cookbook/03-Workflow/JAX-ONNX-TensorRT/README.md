# Workflow of JAX -> ONNX -> TensorRT

+ A workflow of: train model in JAX, export model to ONNX through `jax2tf` + `tf2onnx`, parse ONNX in TensorRT, build TensorRT engine and do inference.

+ Need package `jax`, `flax`, `optax`, `tensorflow`, `tf2onnx` which are not included in the docker image.

+ Steps to run.

```bash
python3 main.py
```
