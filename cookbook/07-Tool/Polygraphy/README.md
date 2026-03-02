# Polygraphy - Client tool

+ CLI tool of polygraphy (deep learning model debugger).

+ Method of installation

```bash
pip install polygraph\
```

+ Features:
  + Do inference computation using multiple backends, including TensorRT, onnxruntime, TensorFlow etc.
  + Compare results of computation layer by layer among different backends.
  + Generate TensorRT engine from model file and serialize it as .trt file.
  + Print the detailed information of model.
  + Modify ONNX model, such as extracting subgraph, simplifying computation graph.
  + Analyze the failure of parsing ONNX model into TensorRT, and save the subgrapha that can / cannot be converted to TensorRT.
