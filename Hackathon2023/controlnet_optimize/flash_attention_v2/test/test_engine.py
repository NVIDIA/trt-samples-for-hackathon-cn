import tensorrt as trt
import torch
from cuda import cudart
import ctypes
from pytorch_lightning import seed_everything

plugin_lib = "./build/fMHAPlugin.so"
ctypes.cdll.LoadLibrary(plugin_lib)

trt_logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(trt_logger, '')

with open("attentionPlugin.engine", "rb") as f:
    engine_str = f.read()
engine = trt.Runtime(trt_logger).deserialize_cuda_engine(engine_str)
context = engine.create_execution_context()

seed_everything(1892849)
q = torch.randn((2,1536,8,40), dtype=torch.float16).cuda().contiguous()
k = torch.randn((2,1536,8,40), dtype=torch.float16).cuda().contiguous()
v = torch.randn((2,1536,8,40), dtype=torch.float16).cuda().contiguous()
output = torch.randn((2,1536,8,40), dtype=torch.float16).cuda().contiguous()

tensor_list = [q, k, v, output]

for i in range(4):
    name = engine.get_tensor_name(i)
    context.set_tensor_address(name, tensor_list[i].data_ptr())
_, stream = cudart.cudaStreamCreate()
context.execute_async_v3(stream)

print("finished")
