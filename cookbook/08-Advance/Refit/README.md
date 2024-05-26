# Refit

+ Several methods to refit a TensorRT engine.

+ In case `case_dummy_engine`, we build a dummy engine with untrained weights.

+ In case `case_set_weights`, we refit the dummay engine with new weights, it contains two equivalent implementations.

+ In case `case_set_weights_gpu`, we refit the dummay engine with new weights, which GPU is on GPU.

+ In case `case_refit_engine`, we refit the dummy engine with trained ONNX file directly, it contains three equivalent implementations.

## Steps to run

```shell
python3 main.py
```
