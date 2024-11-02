# Workflow of pyTorch -> TensorRT

+ Steps to run.

```bash
python3 main.py
```

+ Here is a workflow of:
  + export weights of a trained model from pyTorch to NPZ file.
  + Build TensorRT engine with the weights and do inference

+ We need to run `00-Data/get_model.py` firstly to get related weights files as input.

+ Here are some independent cases:
  + Normal: use FP32 to work (original ONNX is in FP32 mode).
  + FP16: use FP16 to work.
  + INT8-PTQ: use INT8-PTQ to work, it shows the usage of calibrator as well.

+ Here is a equivalent INT8-PTQ workflow in `C++` directory, the key point is the calibrator, comparing to the example in 01-SimpleDemo

```bash
cd C++
make clean && make
./main.exe
```
