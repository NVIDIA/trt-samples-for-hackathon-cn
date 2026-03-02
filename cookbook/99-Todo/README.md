# TODO

+ To-Do list for the cookbook.

+ 05-Plugin/INT8-QDQ-Plugin
+ EinsumLayer implicit mode
+ IGpuAsyncAllocator
+ NetowrkPrinter with Loop structure.
+ Random seed appearance of Fill layer
+ TensorRT LWS?
+ TF2 API examples: TF-TRT, OnnxWorkflowWithPlugin
+ Unify 05-Plugin/PluginRepository/
+ APILanguage in 05-Plugin/APIs (C++ / Python)
+ Comment of Resize layer
+ Algorithm Selector serialize / deserialize to get deterministic engine
+ Better Network printer

+ trt.weights (https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/FoundationalTypes/Weights.html)
+ trt.Dims (https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/FoundationalTypes/Dims.html)

+ set_calibration_profile / get_calibration_profile in BuilderConfig

+ NCCL send/recv plugin

+ Torch-TensorRT
+ Deploy on Triton server
+ Usage of Nsight Deep Learning Designer

+ TF-TRT
+ Model Optimizer replaces of PyTorch-Quantization-Toolkit / TensorFlow-Quantization-Toolkit

+ BuilderFlag::kDISTRIBUTIVE_INDEPENDENCE

```Python
class StreamReaderV2(trt.IStreamReaderV2):
    def __init__(self, bytes):
        trt.IStreamReaderV2.__init__(self)
        self.bytes = bytes
        self.len = len(bytes)
        self.index = 0

    def read(self, size, cudaStreamPtr):
        assert self.index + size <= self.len
        data = self.bytes[self.index:self.index + size]
        self.index += size
        return data

    def seek(self, offset, where):
        if where == SeekPosition.SET:
            self.index = offset
        elif where == SeekPosition.CUR:
            self.index += offset
        elif where == SeekPosition.END:
            self.index = self.len - offset
        else:
            raise ValueError(f"Invalid seek position: {where}")

reader_v2 = MyStreamReaderV2("model.plan")
engine = runtime.deserialize_cuda_engine(reader_v2)
```
