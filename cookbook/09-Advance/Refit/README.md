#

## Introduction

+ Refit-set_weights.py, using set_ Weights API

+ Refit-set_named_weights.py, using set_named_Weights API (slightly different from set_weights API, single function is basically equivalent)

+ Refit-OnnxByParser.py, use new weights from ONNX files, and use TensorRT Parser for modification

+ Refit-OnnxByWeight.py, use the new weight from the ONNX file, and use the method of saving the weight of the ONNX file and re feeding it to the layer in TensorRT for modification

## Steps to run

```shell
python3 Refit-set_weights.py
python3 Refit-set_named_weights.py
python3 Refit-OnnxByParser.py
python3 Refit-OnnxByWeight.py
```

## Output for reference: ./result-*.log

