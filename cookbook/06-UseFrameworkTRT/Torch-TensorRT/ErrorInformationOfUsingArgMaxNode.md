#

## Error information of using torch.argmax in the network (use the same model as 04-Parser/pyTorch-ONNX-TensorRT)
```
ERROR: [Torch-TensorRT] - Unsupported operator: aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> (Tensor)
pyTorch-TensorRT.py(133): forward
/opt/conda/lib/python3.8/site-packages/torch/nn/modules/module.py(1098): _slow_forward
/opt/conda/lib/python3.8/site-packages/torch/nn/modules/module.py(1110): _call_impl
/opt/conda/lib/python3.8/site-packages/torch/jit/_trace.py(958): trace_module
/opt/conda/lib/python3.8/site-packages/torch/jit/_trace.py(741): trace
pyTorch-TensorRT.py(193): <module>

ERROR: [Torch-TensorRT] - Method requested cannot be compiled by Torch-TensorRT.TorchScript.
Unsupported operators listed below:
  - aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> (Tensor)
You can either implement converters for these ops in your application or request implementation
https://www.github.com/nvidia/Torch-TensorRT/issues

In Module:

ERROR: [Torch-TensorRT] - Unsupported operator: aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> (Tensor)
pyTorch-TensorRT.py(133): forward
/opt/conda/lib/python3.8/site-packages/torch/nn/modules/module.py(1098): _slow_forward
/opt/conda/lib/python3.8/site-packages/torch/nn/modules/module.py(1110): _call_impl
/opt/conda/lib/python3.8/site-packages/torch/jit/_trace.py(958): trace_module
/opt/conda/lib/python3.8/site-packages/torch/jit/_trace.py(741): trace
pyTorch-TensorRT.py(193): <module>

WARNING: [Torch-TensorRT] - Dilation not used in Max pooling converter
WARNING: [Torch-TensorRT] - Dilation not used in Max pooling converter
Traceback (most recent call last):
  File "pyTorch-TensorRT.py", line 200, in <module>
    outputData = trtModel(inputData)  # run inference in TensorRT
  File "/opt/conda/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
RuntimeError: Method (but not graphs in general) require a single output. Use None/Tuple for 0 or 2+ outputs
```
