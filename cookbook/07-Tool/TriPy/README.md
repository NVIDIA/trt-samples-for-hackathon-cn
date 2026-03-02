# TriPy

+ A Python programming model for TensorRT that aims to provide an excellent user experience without compromising performance.

+ GitHub [Link](https://github.com/NVIDIA/TensorRT-Incubator/tree/main/tripy)
+ [Document](https://nvidia.github.io/TensorRT-Incubator/pre0_user_guides/introduction-to-tripy.html)

+ Installation (according to GitHub)
  + Download packages from https://nvidia.github.io/TensorRT-Incubator/packages.html, we need `mlir_tensorrt_compiler`, `mlir_tensorrt_runtime` and `tripy`.
  + `pip install` all the packages.
  + According to my last test, tripy works well with these packages:
    + mlir_tensorrt_compiler-0.1.31+cuda12.trt102-cp310-cp310-linux_x86_64.whl
    + mlir_tensorrt_runtime-0.1.31+cuda12.trt102-cp310-cp310-linux_x86_64.whl
    + tripy-0.0.2-py3-none-any.whl
