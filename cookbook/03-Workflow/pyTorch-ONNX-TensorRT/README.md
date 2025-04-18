# Workflow of pyTorch -> ONNX -> TensorRT

+ A workflow of: export trained model from pyTorch to ONNX, parse ONNX in TensorRT, build TensorRT engine and do inference.

+ Steps to run.

```bash
python3 main.py
```

+ The process of build and export a ONNX model is inside `00-Data/get_model_part1.py`.

+ Here are some independent cases:
  + Normal: use FP32 to work (original ONNX is in FP32 mode).
  + FP16: use FP16 to work.
  + INT8-PTQ: use INT8-PTQ to work, it shows the usage of calibrator as well.
  + INT8-QAT: use INT8-QAT to work, we need a INT8-QAT ONNX as input.

+ Here is a equivalent INT8-PTQ workflow in `C++` directory, the key point is the calibrator, comparing to the example in 01-SimpleDemo

```bash
cd C++
make clean && make
./main.exe
```
