import os

import numpy as np
import tensorrt as trt
from tqdm import tqdm

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime

from tensorrt_cookbook import TRTWrapperV1, check_array

trt_file = Path("model.trt")
report_file = Path("report.txt")
time_cache_file = Path("model.TimingCache")
exclude_list = {"SHAPE", "PLUGIN", "PLUGIN_V2", "PLUGIN_V3", "CONSTANT", "ASSERTION", "SHUFFLE", "IDENTITY", "CONCATENATION", "GATHER", "SLICE", "RESIZE", "UNARY", "CONDITION", "CONDITIONAL_INPUT", "CONDITIONAL_OUTPUT", "FILL", "NON_ZERO", "ONE_HOT"}
default_onnx_file = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "model" / "model-trained.onnx"
default_data_file = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data" / "InferenceData.npy"
default_input_shape_dict = "x:1x1x28x28"

class FP16Tuning:

    def __init__(self, args):
        for name in dir(args):
            if not name.startswith("_"):
                self.__setattr__(name, args.__getattribute__(name))

        self.result_table = {}
        return

    def get_input_shape(self) -> None:
        user_dict = {}
        if len(self.input_shape_dict) > 0:
            for input in self.input_shape_dict.split(','):
                name, shape = input.split(':')
                user_dict[name] = [[int(x) for x in shape.split('x')], None]

        graph = gs.import_onnx(onnx.load(self.onnx_file))
        model_dict = {}
        for input in graph.inputs:
            name = input.name
            shape = input.shape
            dtype = input.dtype
            if name in user_dict:
                model_dict[name] = [user_dict[name][0], dtype]
            else:
                model_dict[name] = [[i if isinstance(i, int) and i > 0 else 1 for i in shape], dtype]
            print(f"Input  tensor: name={name}, shape={model_dict[name][0]},dtype={model_dict[name][1]}")

        self.input_shape_dict = model_dict
        return

    def get_reference_data(self) -> None:
        input_data = {}
        output_data = {}
        if self.data_file.exists():
            np_data = np.load(self.data_file)
            if isinstance(np_data, np.ndarray):
                if len(self.input_shape_dict) > 1:
                    print(f"Only one input data in {self.data_file}, but more than one input tensor in {self.onnx_file}")
                    raise Exception
                if self.data_file == default_data_file:  # Use default data, or we need to set name of input tensor here
                    input_data["x"] = np_data
            elif isinstance(np_data, np.lib.npyio.NpzFile):
                if set(np_data.keys()) != set(self.input_shape_dict.keys()):
                    print(f"Name of input tensor in {self.data_file} ({set(np_data.keys())}) is not equal to that in {self.onnx_file} ({set(self.input_shape_dict.keys())})")
                    raise Exception
                for name, data in np_data.items():
                    if np.all(np.array(data.shape) != np.array(self.input_shape_dict[name][0])):
                        print(f"Shape of input tensor '{name}' in {self.data_file} ({data.shape}) is not equal to that in {self.onnx_file} ({self.input_shape_dict[name][0]})")
                        raise Exception
                    input_data[name] = data
            else:
                print("IO data file format is not supported")
                raise Exception
        else:
            for name, [shape, dtype] in self.input_shape_dict.items():
                input_data[name] = np.random.rand(np.prod(shape)).astype(dtype).reshape(shape)

        session = onnxruntime.InferenceSession(self.onnx_file)
        outputList = session.run(None, input_data)

        for i, output_tensor in enumerate(session.get_outputs()):
            print(f"Output tensor: name={output_tensor.name}, shape={outputList[i].shape},dtype={outputList[i].dtype}")
            output_data[output_tensor.name] = outputList[i]

        self.input_data = input_data
        self.output_data = output_data

    def get_layer_info(self) -> None:
        tw = TRTWrapperV1(plugin_file_list=self.plugin_file_list)
        parser = trt.OnnxParser(tw.network, tw.logger)
        with open(self.onnx_file, "rb") as model:
            parser.parse(model.read())

        for i in range(tw.network.num_inputs):
            input_tensor = tw.network.get_input(i)
            shape = self.input_shape_dict[input_tensor.name][0]
            tw.profile.set_shape(input_tensor.name, shape, shape, shape)
        tw.config.add_optimization_profile(tw.profile)

        layer_list = []
        for i in range(tw.network.num_layers):
            layer = tw.network.get_layer(i)

            if str(layer.type)[10:] in exclude_list:
                continue
            #layer_list.append([layer.name, layer.type.name])
            layer_list.append(layer.name)

        self.layer_list = layer_list
        return

    def run(self, b_fp32, layer_name_in_fp32: str = "", b_test_performance: bool = False):
        if not b_fp32:
            self.result_table[layer_name_in_fp32] = {}  # Only one layer supported now

        if len(self.plugin_file_list) > 0:
            plugin_file_list = [Path(i) for i in self.plugin_file_list.split(",")]
        else:
            plugin_file_list = []

        tw = TRTWrapperV1(plugin_file_list=plugin_file_list)
        time_cache = b""
        if time_cache_file.exists():
            with open(time_cache_file, 'rb') as f:
                time_cache = f.read()

        cache = tw.config.create_timing_cache(time_cache)
        tw.config.set_timing_cache(cache, False)

        if not b_fp32:
            tw.config.set_flag(trt.BuilderFlag.FP16)
            tw.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        parser = trt.OnnxParser(tw.network, tw.logger)
        with open(self.onnx_file, "rb") as model:
            parser.parse(model.read())

        for i in range(tw.network.num_inputs):
            input_tensor = tw.network.get_input(i)
            shape = self.input_shape_dict[input_tensor.name][0]
            tw.profile.set_shape(input_tensor.name, shape, shape, shape)
        tw.config.add_optimization_profile(tw.profile)

        if not b_fp32:
            for i in range(tw.network.num_layers):
                layer = tw.network.get_layer(i)
                if layer.name == layer_name_in_fp32:
                    layer.precision = trt.float32

        tw.build()

        timing_cache = tw.config.get_timing_cache()
        timing_cache_buffer = timing_cache.serialize()
        with open(time_cache_file, "wb") as f:
            f.write(timing_cache_buffer)

        tw.setup(self.input_data, b_print_io=False)
        tw.infer(b_print_io=False)

        time = "+inf"
        if b_test_performance:
            command = f"trtexec --onnx={self.onnx_file} --useSpinWait --noDataTransfers"
            command += f" --saveEngine={trt_file}"
            #command += " " + "--verbose"  # for debug
            if not b_fp32:
                command += " --fp16"
            if len(layer_name_in_fp32) > 0:
                command += " --precisionConstraints=prefer"
                command += f" --layerPrecisions={layer_name}:fp32"
            command += " 2>/dev/null"
            output = os.popen(command)

            time = "+inf"
            for line in output.readlines():
                #print(line)  # for debug
                if "[I] GPU Compute Time" in line:
                    time = float(line.split("ms")[3].split("=")[1])

        if b_fp32:  # Save baseline result
            trt_data = {}
            for name in tw.tensor_name_list:
                trt_data[name] = tw.buffer[name][0]
            self.trt_data = trt_data
        else:  # Compute loss
            for name in tw.tensor_name_list:
                if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                    now_data = tw.buffer[name][0]
                    ref_data = self.trt_data[name]
                    if False:
                        check_array(now_data, ref_data, weak=True, info=name)
                    max_error = np.max(np.abs(now_data - ref_data))
                    mean_error = np.mean(np.abs(now_data - ref_data))
                    self.result_table[layer_name_in_fp32][name] = [time, max_error, mean_error]

        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    shape_help_info = "Dictionary of input tensor shape, for example: x:4x64x64,y:4,z:"

    parser.add_argument('--onnx_file', '-i', type=Path, default=default_onnx_file, help="Path of input onnx file.")
    parser.add_argument('--data_file', '-d', type=Path, default=default_data_file, help="Path of input data file (.npy / .npz).")
    parser.add_argument('--plugin_file_list', '-p', type=str, default="", help="List of plugins")
    parser.add_argument('--input_shape_dict', '-s', type=str, default=default_input_shape_dict, help=shape_help_info)
    args = parser.parse_args()

    model = FP16Tuning(args)
    model.get_input_shape()
    model.get_reference_data()
    model.get_layer_info()

    print("Get baseline")
    model.run(True)

    print("Tune")
    for layer in tqdm(model.layer_list):
        try:
            model.run(False, layer)
        except:
            print(f"Error build with layer {layer}")

    with open(report_file, "w") as ff:
        for layer_name, value in model.result_table.items():
            ss = f"{layer_name}:\n"
            for tensor_name, value in value.items():
                ss += f"    {tensor_name:20s}: {value}\n"
            ff.write(ss + "\n")

    print("Finish")
