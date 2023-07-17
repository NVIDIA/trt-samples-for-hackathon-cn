import cupy as cp
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
from numba import cuda
from polygraphy.backend.trt import (CreateConfig, EngineFromNetwork,
                                    NetworkFromOnnxPath, TrtRunner)
from polygraphy.comparator import Comparator
from polygraphy.json import from_json, to_json


def volume(d):
    return np.prod(d)

# copied from https://gist.github.com/leofang/4a043e5d94b4702d04fde2b9e7dcebbd
# Converts cupy CUDA stream object to a Numba CUDA stream object
def stream_cupy_to_numba(cp_stream):
    """
    Notes:
        1. The lifetime of the returned Numba stream should be as long as the CuPy one,
           which handles the deallocation of the underlying CUDA stream.
        2. The returned Numba stream is assumed to live in the same CUDA context as the
           CuPy one.
        3. The implementation here closely follows that of cuda.stream() in Numba.
    """
    import weakref
    from ctypes import c_void_p

    # get the pointer to actual CUDA stream
    raw_str = cp_stream.ptr

    # gather necessary ingredients
    ctx = cuda.devices.get_context()
    handle = c_void_p(raw_str)
    finalizer = None  # let CuPy handle its lifetime, not Numba

    # create a Numba stream
    nb_stream = cuda.cudadrv.driver.Stream(weakref.proxy(ctx), handle, finalizer)

    return nb_stream


@cuda.jit
def circ_pad(X, all_pads, orig_dims, Y, Y_shape, Y_len):
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    stride = cuda.blockDim.x * cuda.gridDim.x

    for i in range(index, Y_len, stride):
        i3 = int(i % Y_shape[3])
        i2 = int((i // Y_shape[3]) % Y_shape[2])
        i1 = int((i // Y_shape[3] // Y_shape[2]) % Y_shape[1])
        i0 = int(i // Y_shape[3] // Y_shape[2] // Y_shape[1])

        j0 = int((i0 - all_pads[0]) % orig_dims[0])
        j1 = int((i1 - all_pads[2]) % orig_dims[1])
        j2 = int((i2 - all_pads[4]) % orig_dims[2])
        j3 = int((i3 - all_pads[6]) % orig_dims[3])

        Y[i] = X[
            int(
                orig_dims[3] * orig_dims[2] * orig_dims[1] * j0
                + orig_dims[3] * orig_dims[2] * j1
                + orig_dims[3] * j2
                + j3
            )
        ]


class CircPadPlugin(trt.IPluginV2DynamicExt):
    def __init__(self, fc=None):
        trt.IPluginV2DynamicExt.__init__(self)
        self.pads = []
        self.X_shape = []
        self.plugin_namespace = ""

        if fc is not None:
            assert fc[0].name == "pads"
            self.pads = fc[0].data

    def initialize(self):
        return 0

    def get_num_outputs(self):
        return 1

    def get_output_datatype(self, index, input_types):
        return input_types[0]

    def get_output_dimensions(self, output_index, inputs, exprBuilder):

        output_dims = trt.DimsExprs(inputs[0])

        for i in range(np.size(self.pads) // 2):
            output_dims[len(output_dims) - i - 1] = exprBuilder.operation(
                trt.DimensionOperation.SUM,
                inputs[0][len(output_dims) - i - 1],
                exprBuilder.constant(self.pads[i * 2] + self.pads[i * 2 + 1]),
            )

        return output_dims

    def get_serialization_size(self):
        return len(to_json({"pads": self.pads}))

    def serialize(self):
        return to_json({"pads": self.pads})

    def configure_plugin(self, inp, out):
        X_dims = inp[0].desc.dims
        self.X_shape = np.zeros((len(X_dims),))
        for i in range(len(X_dims)):
            self.X_shape[i] = X_dims[i]

    def supports_format_combination(self, pos, in_out, num_inputs):
        assert num_inputs == 1
        assert pos < len(in_out)

        desc = in_out[pos]
        if desc.format != trt.TensorFormat.LINEAR:
            return False

        # first input should be float16 or float32
        if pos == 0:
            return desc.type == trt.DataType.FLOAT or desc.type == trt.DataType.HALF

        # output should have the same type as the input
        if pos == 1:
            return in_out[0].type == desc.type

        assert False

    def get_workspace_size(self, input_desc, output_desc):
        return 0

    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):

        inp_dtype = trt.nptype(input_desc[0].type)

        a_mem = cp.cuda.UnownedMemory(
            inputs[0], volume(input_desc[0].dims) * cp.dtype(inp_dtype).itemsize, self
        )
        c_mem = cp.cuda.UnownedMemory(
            outputs[0],
            volume(output_desc[0].dims) * cp.dtype(inp_dtype).itemsize,
            self,
        )

        a_ptr = cp.cuda.MemoryPointer(a_mem, 0)
        c_ptr = cp.cuda.MemoryPointer(c_mem, 0)


        a = cp.ndarray((volume(input_desc[0].dims)), dtype=inp_dtype, memptr=a_ptr)
        c = cp.ndarray((volume(output_desc[0].dims)), dtype=inp_dtype, memptr=c_ptr)

        cuda_stream = cp.cuda.ExternalStream(stream)
        numba_stream = stream_cupy_to_numba(cuda_stream)

        N = len(self.X_shape)
        all_pads = np.zeros((N * 2,))
        orig_dims = np.array(self.X_shape)
        out_dims = np.array(self.X_shape)

        for i in range(np.size(pads) // 2):
            out_dims[N - i - 1] += pads[i * 2] + pads[i * 2 + 1]
            all_pads[N * 2 - 2 * i - 2] = pads[i * 2]
            all_pads[N * 2 - 2 * i - 1] = pads[i * 2 + 1]

        all_pads_d = cp.asarray(all_pads)
        orig_dims_d = cp.asarray(orig_dims)
        Y_shape_d = cp.asarray(out_dims)

        blockSize = 256
        numBlocks = int((np.prod(out_dims) + blockSize - 1) // blockSize)

        circ_pad[numBlocks, blockSize, numba_stream](
            a, all_pads_d, orig_dims_d, c, Y_shape_d, np.prod(out_dims)
        )

        return 0

    def destroy(self):
        pass

    def clone(self):
        cloned_plugin = CircPadPlugin.__new__(CircPadPlugin)
        trt.IPluginV2DynamicExt.__init__(cloned_plugin, self)
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin

    def terminate(self):
        pass

    def get_plugin_namespace(self):
        return self.namespace

    def set_plugin_namespace(self, namespace):
        self.namespace = namespace

    def get_plugin_type(self):
        return "CircPadPlugin"

    def get_plugin_version(self):
        return "1"

    def __getstate__(self):
        return {"pads": self.pads}


class CircPadPluginCreator(trt.IPluginCreator):
    def __init__(self):
        trt.IPluginCreator.__init__(self)
        self.plugin_name = "CircPadPlugin"
        self.namespace = ""
        self.version = "1"
        self.plugin_field_names = trt.PluginFieldCollection(
            [trt.PluginField("pads", np.array([]), trt.PluginFieldType.INT32)]
        )

    def get_plugin_name(self):
        return self.plugin_name

    def get_plugin_version(self):
        return self.version

    def get_field_names(self):
        return self.plugin_field_names

    def create_plugin(self, name, fc):
        return CircPadPlugin(fc)

    def deserialize_plugin(self, name, data):
        deserialized = CircPadPlugin()
        j = dict(from_json(data))
        deserialized.__dict__.update(j)
        return deserialized

    def get_plugin_namespace(self):
        return self.namespace

    def set_plugin_namespace(self, namespace):
        self.namespace = namespace


precision = np.float16

inp_shape = (10, 3, 32, 32)
X = np.random.normal(size=inp_shape).astype(precision)

pads = (1, 1, 1, 1)

# Register plugin creator
plg_registry = trt.get_plugin_registry()
my_plugin_creator = CircPadPluginCreator()
plg_registry.register_creator(my_plugin_creator, "")

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
