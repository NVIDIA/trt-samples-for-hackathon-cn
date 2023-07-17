import ctypes

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
from polygraphy.backend.trt import (CreateConfig, EngineFromNetwork,
                                    NetworkFromOnnxPath, TrtRunner)
from polygraphy.comparator import Comparator, CompareFunc

precision = np.float16
inp_shape = (10, 3, 32, 32)
X = np.random.normal(size=inp_shape).astype(precision)

pads = (1, 1, 1, 1)

# create ONNX model
onnx_path = "test_CircPadPlugin.onnx"
inputA = gs.Variable(name="X", shape=inp_shape, dtype=precision)
Y = gs.Variable(name="Y", dtype=precision)
myPluginNode = gs.Node(
    name="CircPadPlugin",
    op="CircPadPlugin",
    inputs=[inputA],
    outputs=[Y],
    attrs={"pads": pads},
)
graph = gs.Graph(nodes=[myPluginNode], inputs=[inputA], outputs=[Y], opset=16)
onnx.save(gs.export_onnx(graph), onnx_path)

logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')
ctypes.cdll.LoadLibrary("./circ_pad_plugin.so")

# build engine
build_engine = EngineFromNetwork(
    NetworkFromOnnxPath(onnx_path), CreateConfig(fp16=precision==np.float16)
)

# run
results = Comparator.run(
    [TrtRunner(build_engine, "trt_runner")], warm_up=10, data_loader=[{"X": X}]
)

Y_ref = np.pad(X, [[0, 0], [0, 0], [pads[0], pads[1]], [pads[2], pads[3]]], "wrap")
Y = results["trt_runner"][0]["Y"]

if np.allclose(Y, Y_ref):
    print("Inference result correct!")
