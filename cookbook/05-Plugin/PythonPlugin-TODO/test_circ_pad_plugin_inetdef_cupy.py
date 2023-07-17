import cupy as cp
import numpy as np
import tensorrt as trt
from polygraphy.backend.trt import (CreateConfig, TrtRunner, create_network,
                                    engine_from_network)
from polygraphy.comparator import Comparator
from polygraphy.json import from_json, to_json


def volume(d):
    return np.prod(d)

circ_pad_half_kernel = cp.RawKernel(r'''
#include <cuda_fp16.h>
extern "C" __global__
void circ_pad_half(half const* X, int const* all_pads, int const* orig_dims, half* Y, int const* Y_shape, int const* Y_len) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < *Y_len; i += stride)
    {
        int i3 = i % Y_shape[3];
        int i2 = (i / Y_shape[3]) % Y_shape[2];
        int i1 = (i / Y_shape[3] / Y_shape[2]) % Y_shape[1];
        int i0 = i / Y_shape[3] / Y_shape[2] / Y_shape[1];

        int j0 = (i0 - all_pads[0] + orig_dims[0]) % orig_dims[0];
        int j1 = (i1 - all_pads[2] + orig_dims[1]) % orig_dims[1];
        int j2 = (i2 - all_pads[4] + orig_dims[2]) % orig_dims[2];
        int j3 = (i3 - all_pads[6] + orig_dims[3]) % orig_dims[3];

        Y[i] = X[
            orig_dims[3] * orig_dims[2] * orig_dims[1] * j0
            + orig_dims[3] * orig_dims[2] * j1
            + orig_dims[3] * j2
            + j3
        ];
    }
}
''', 'circ_pad_half')

circ_pad_float_kernel = cp.RawKernel(r'''
extern "C" __global__
void circ_pad_float(float const* X, int const* all_pads, int const* orig_dims, float* Y, int const* Y_shape, int const* Y_len) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < *Y_len; i += stride)
    {
        int i3 = i % Y_shape[3];
        int i2 = (i / Y_shape[3]) % Y_shape[2];
        int i1 = (i / Y_shape[3] / Y_shape[2]) % Y_shape[1];
        int i0 = i / Y_shape[3] / Y_shape[2] / Y_shape[1];

        int j0 = (i0 - all_pads[0] + orig_dims[0]) % orig_dims[0];
        int j1 = (i1 - all_pads[2] + orig_dims[1]) % orig_dims[1];
        int j2 = (i2 - all_pads[4] + orig_dims[2]) % orig_dims[2];
        int j3 = (i3 - all_pads[6] + orig_dims[3]) % orig_dims[3];

        Y[i] = X[
            orig_dims[3] * orig_dims[2] * orig_dims[1] * j0
            + orig_dims[3] * orig_dims[2] * j1
            + orig_dims[3] * j2
            + j3
        ];
    }
}
''', 'circ_pad_float')

class CircPadPlugin(trt.IPluginV2DynamicExt):
    def __init__(self, fc=None):
        trt.IPluginV2DynamicExt.__init__(self)
        self.pads = []
        self.X_shape = []
        self.namespace = ""

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

        N = len(self.X_shape)
        all_pads = np.zeros((N * 2,))
        orig_dims = np.array(self.X_shape)
        out_dims = np.array(self.X_shape)

        for i in range(np.size(pads) // 2):
            out_dims[N - i - 1] += self.pads[i * 2] + self.pads[i * 2 + 1]
            all_pads[N * 2 - 2 * i - 2] = self.pads[i * 2]
            all_pads[N * 2 - 2 * i - 1] = self.pads[i * 2 + 1]

        self.all_pads_d = cp.asarray(all_pads, dtype=cp.int32)
        self.orig_dims_d = cp.asarray(orig_dims, dtype=cp.int32)
        self.Y_shape_d = cp.asarray(out_dims, dtype=cp.int32)
        self.Y_len_d = cp.array([np.prod(out_dims)], dtype=cp.int32)

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

        blockSize = 256
        numBlocks = int((np.prod(np.array(self.X_shape)) + blockSize - 1) // blockSize)

        with cuda_stream:
            if inp_dtype == np.float32:
                circ_pad_float_kernel((numBlocks,), (blockSize,), (a, self.all_pads_d, self.orig_dims_d, c, self.Y_shape_d, self.Y_len_d))
            elif inp_dtype == np.float16:
                circ_pad_half_kernel((numBlocks,), (blockSize,), (a, self.all_pads_d, self.orig_dims_d, c, self.Y_shape_d, self.Y_len_d))
            else:
                assert False, "inp_dtype not valid"

        return 0

    def destroy(self):
        # final steps before plugin object is destroyed
        pass

    def clone(self):
        cloned_plugin = CircPadPlugin.__new__(CircPadPlugin)
        trt.IPluginV2DynamicExt.__init__(cloned_plugin, self)
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin

    def terminate(self):
        pass

    def get_plugin_namespace(self):
        return str(self.namespace)

    def set_plugin_namespace(self, namespace):
        self.namespace = namespace

    def get_plugin_type(self):
        return "CircPadPlugin"

    def get_plugin_version(self):
        return "1"


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


precision = trt.float16

inp_shape = (100, 2, 32, 32)
X = np.random.normal(size=inp_shape).astype(trt.nptype(precision))

pads = (1, 1, 1, 1)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# Load standard plugins (if needed)
trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")

# Register plugin creator
plg_registry = trt.get_plugin_registry()
my_plugin_creator = CircPadPluginCreator()
plg_registry.register_creator(my_plugin_creator, "")

# Create plugin object
builder, network = create_network()
plg_creator = plg_registry.get_plugin_creator("CircPadPlugin", "1", "")
plugin_fields_list = [trt.PluginField("pads", np.array(pads, dtype=np.int32), trt.PluginFieldType.INT32)]
pfc = trt.PluginFieldCollection(plugin_fields_list)
plugin = plg_creator.create_plugin("CircPadPlugin", pfc)

# Populate network
input_X = network.add_input(name="X", dtype=precision, shape=X.shape)
out = network.add_plugin_v2([input_X], plugin)
out.get_output(0).name = "Y"
network.mark_output(tensor=out.get_output(0))

# Build engine
config = builder.create_builder_config()
engine = engine_from_network((builder, network), CreateConfig(fp16=precision==trt.float16))

# Run
results = Comparator.run([TrtRunner(engine, "trt_runner")], warm_up=10, data_loader=[{"X": X}])

Y_ref = np.pad(X, [[0, 0], [0, 0], [pads[0], pads[1]], [pads[2], pads[3]]], "wrap")
Y = results["trt_runner"][0]["Y"]

if np.allclose(Y, Y_ref):
    print("Inference result correct!")
