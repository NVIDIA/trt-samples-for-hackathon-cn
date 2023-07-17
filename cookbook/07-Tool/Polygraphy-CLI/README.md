# Polygraphy - Client tool

+ CLI tool of polygraphy

+ Deep learning model debugger, including equivalent python APIs and command-line tools.

+ Method of installation:

```shell
pip install polygraph\
```

+ [Document](https://docs.nvidia.com/deeplearning/tensorrt/polygraphy/docs/index.html) and [tutorial video](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31695/).

+ Function:
  + Do inference computation using multiple backends, including TensorRT, onnxruntime, TensorFlow etc.
  + Compare results of computation layer by layer among different backends.
  + Generate TensorRT engine from model file and serialize it as .plan file.
  + Print the detailed information of model.
  + Modify ONNX model, such as extracting subgraph, simplifying computation graph.
  + Analyze the failure of parsing ONNX model into TensorRT, and save the subgrapha that can / cannot be converted to TensorRT.
