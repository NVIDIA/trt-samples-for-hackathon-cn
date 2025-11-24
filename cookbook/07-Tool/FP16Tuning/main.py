# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from pathlib import Path
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime
import tensorrt as trt
from tqdm import tqdm

from tensorrt_cookbook import (TRTWrapperV1, check_array, layer_type_to_layer_type_name)

class FP16Tuning:

    def __init__(self, args):
        self.onnx_file = args.onnx_file
        self.plugin_file_list = args.plugin_file_list
        self.min_shape = args.min_shape
        self.opt_shape = args.opt_shape
        self.max_shape = args.max_shape
        self.infer_shape = args.infer_shape
        self.data_file = args.data_file
        assert args.data_file is not None or args.infer_shape is not None, "Either data_file or infer_shape must be provided"
        self.test_performance = args.test_performance
        self.output_file = args.output_file
        self.result_table = {}

        # Temporary directory ==========================================================================================
        self.temp_dir = Path("FP16TunningTemp")
        self.time_cache_file = self.temp_dir / "model.TimingCache"
        self.trt_file = self.temp_dir / "model.trt"
        Path(self.temp_dir).mkdir(exist_ok=True, parents=True)

        # Get input information ========================================================================================
        graph = gs.import_onnx(onnx.load(self.onnx_file))
        dtype_dict = {tensor.name: tensor.dtype for tensor in graph.inputs}
        onnx_name_set = {tensor.name for tensor in graph.inputs}
        del graph

        name_set = {item.split(":")[0] for item in args.opt_shape.split(",")}
        if len(name_set - onnx_name_set) > 0:
            print(f"Input tensor name [{name_set - onnx_name_set}] are in command but not in ONNX file")
            return
        elif len(onnx_name_set - name_set) > 0:
            print(f"Input tensor name [{onnx_name_set - name_set}] are in ONNX file but not in command")
            return

        shape_dict = {name: [] for name in name_set}
        for item in [args.min_shape, args.opt_shape, args.max_shape]:
            for item in item.split(","):
                name, shape = item.split(":")
                shape_dict[name].append([int(x) for x in shape.split('x')])

        self.name_set = name_set
        self.shape_dict = shape_dict

        # Get target layer =============================================================================================
        tw = self._setup_trt()

        self.target_layer_name_list = [""]  # Add a empty layer name for baseline
        for i in range(tw.network.num_layers):
            layer = tw.network.get_layer(i)
            if layer_type_to_layer_type_name(layer.type) in args.tune_type_list:
                self.target_layer_name_list.append(layer.name)

        # Get ground truth data ========================================================================================
        input_data = {}
        if self.data_file.exists():
            # Use dump data, `self.infer_shape` must be None here
            np_data = np.load(self.data_file)
            np_name_set = set(np_data.keys())
            self.infer_shape = ""
            if len(name_set - np_name_set) > 0:
                print(f"Input tensor name [{name_set - np_name_set}] are in command but not in npz file")
                return
            if len(np_name_set - name_set) > 0:
                print(f"Input tensor name [{np_name_set - name_set}] are in npz file but not in command")
                return
            for name, data in np_data.items():
                assert len(data.shape) == len(shape_dict[name][0]), f"Rank of input tensor {name} in npz file is not equal to it in command line"
                for np_length, min_length, max_length in zip(data.shape, shape_dict[name][0], shape_dict[name][2]):
                    assert min_length <= np_length <= max_length, f"Shape of input tensor {name} in npz file {data.shape} is not in range [{shape_dict[name][0]}, {shape_dict[name][2]}]"
                input_data[name] = data
                self.infer_shape += f"{name}:{'x'.join([str(x) for x in data.shape])},"
            self.infer_shape = self.infer_shape[:-1]
        else:  # Use random data, `self.infer_shape` must not be None here
            for shape_dict in self.infer_shape.split(","):
                name, shape = shape_dict.split(":")
                shape = [int(x) for x in shape.split('x')]
                if dtype_dict[name] in [np.int32, np.int64]:
                    input_data[name] = np.random.randint(1, 1024, shape).astype(dtype_dict[name]).reshape(shape)
                else:
                    input_data[name] = np.random.rand(np.prod(shape)).astype(dtype_dict[name]).reshape(shape)

        # Save binary data for trtexec
        for name, data in input_data.items():
            data.tofile(self.temp_dir / (name + ".bin"))

        self.input_data = input_data

        # Save output data for output comparison
        output_data = {}
        if args.ort_baseline:
            session = onnxruntime.InferenceSession(self.onnx_file)
            outputList = session.run(None, input_data)
            for i, output_tensor in enumerate(session.get_outputs()):
                output_data[output_tensor.name] = outputList[i]
        else:
            output_data = self._single_run("", False, tw)

        self.output_data = output_data

    def tune(self):
        print("=" * 64 + "Tune")
        for layer_name in tqdm(model.target_layer_name_list):
            try:
                model._single_run(layer_name)
            except:
                print(f"Error build with layer {layer_name}")

    def _setup_trt(self):
        plugin_file_list = [Path(i) for i in self.plugin_file_list.split(",")] if len(self.plugin_file_list) > 0 else []
        tw = TRTWrapperV1(plugin_file_list=plugin_file_list)
        parser = trt.OnnxParser(tw.network, tw.logger)
        with open(self.onnx_file, "rb") as model:
            parser.parse(model.read())

        for i in range(tw.network.num_inputs):
            input_tensor = tw.network.get_input(i)
            name = input_tensor.name
            tw.profile.set_shape(input_tensor.name, *self.shape_dict[name])
        tw.config.add_optimization_profile(tw.profile)

        return tw

    def _single_run(self, target_layer_name: str = "", b_fp16: bool = True, tw: TRTWrapperV1 = None):
        if tw is None:  # Use `tw` from arguments while running TRT-FP32 as ground truth, or use a new one while running TRT-FP16
            tw = self._setup_trt()

        time_cache = b""
        if self.time_cache_file.exists():
            with open(self.time_cache_file, 'rb') as f:
                time_cache = f.read()
        cache = tw.config.create_timing_cache(time_cache)
        tw.config.set_timing_cache(cache, False)

        if b_fp16:
            self.result_table[target_layer_name] = {}
            tw.config.set_flag(trt.BuilderFlag.FP16)
            tw.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            for i in range(tw.network.num_layers):
                layer = tw.network.get_layer(i)
                if layer.name == target_layer_name:
                    layer.precision = trt.float32

        tw.build()
        tw.serialize_engine(self.trt_file)

        timing_cache = tw.config.get_timing_cache()
        timing_cache_buffer = timing_cache.serialize()
        with open(self.time_cache_file, "wb") as f:
            f.write(timing_cache_buffer)

        tw.setup(self.input_data, b_print_io=False)
        tw.infer(b_print_io=False)

        if not b_fp16:
            output_data = {}
            for name, data in tw.buffer.items():
                if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                    output_data[name] = data[0]
            return output_data

        time = "+inf"
        if self.test_performance:
            command = f"trtexec --loadEngine={self.trt_file} --useSpinWait --noDataTransfers"
            command += f" --shapes={self.infer_shape}"
            command += f" --loadInputs="
            shape_str = ""
            for name in self.name_set:
                shape_str += f"{name}:{self.temp_dir}/{name}.bin,"
            command += shape_str[:-1]
            #command += " --verbose"  # for debug
            if b_fp16:
                command += " --fp16"
            if target_layer_name != "":
                command += " --precisionConstraints=obey"
                command += f" --layerPrecisions={target_layer_name}:fp32"
            command += " 2>/dev/null"
            output = os.popen(command)

            time = "+inf"
            for line in output.readlines():
                print(line)  # for debug
                if "[I] GPU Compute Time" in line:
                    time = float(line.split("ms")[3].split("=")[1])

        # Compute loss
        for name, data in tw.buffer.items():
            if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                now_data = data[0]
                ref_data = self.output_data[name]
                if False:
                    check_array(now_data, ref_data, weak=True, info=name)
                max_error = np.max(np.abs(now_data - ref_data))
                mean_error = np.mean(np.abs(now_data - ref_data))
                self.result_table[target_layer_name][name] = [time, max_error, mean_error]

        with open(self.output_file, "w") as ff:  # update every time
            for layer_name, value in self.result_table.items():
                ss = f"{layer_name}:\n"
                for tensor_name, value in value.items():
                    ss += f"    {tensor_name:20s}: {value}\n"
                ff.write(ss + "\n")

        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """
    shape_help_info = "Dictionary of input tensor shape in trtexec format, for example: x:4x64x64,y:4,z:"
    type_help_info = "List of layer types to be tuned, for example: -t CONVOLUTION,MATRIX_MULTIPLY"

    # yapf:disable
    parser.add_argument('--onnx_file',          '-i',   type=Path, r                equired=True,                               help="Path of input onnx file")
    parser.add_argument('--plugin_file_list',   '-pl',  type=str,                   default="",                                 help="List of plugins")
    parser.add_argument('--min_shape',          '-min', type=str,                   default="",                                 help=shape_help_info)
    parser.add_argument('--opt_shape',          '-opt', type=str,                   required=True,                              help=shape_help_info)
    parser.add_argument('--max_shape',          '-max', type=str,                   default="",                                 help=shape_help_info)
    parser.add_argument('--infer_shape',        '-s',   type=str,                   default=None,                               help=shape_help_info)
    parser.add_argument('--data_file',          '-d',   type=Path,                  default=None,                               help="Path of input data npz file")
    parser.add_argument('--ort_baseline',       '-ort', action='store_true',                                                    help="Use onnxruntime as baseline, otherwise TRT-FP32 is used")
    parser.add_argument('--tune_type_list',     '-t',   type=lambda s:s.split(','), default="CONVOLUTION,MATRIX_MULTIPLY",      help=type_help_info)
    parser.add_argument('--test_performance',   '-p',   action='store_true',                                                    help="Use trtexec for performance test")
    parser.add_argument('--output_file',        '-o',   type=Path,                  default=Path("FP16TunningTemp/report.txt"), help="Path of output report file")
    # yapf:enable
    """

    # In this example, we use hard-core arguments below rather than argparse
    data = {"x": np.load(Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data" / "InferenceData.npy")}
    np.savez("data.npz", **data)

    args = parser.parse_args()
    args.onnx_file = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "model" / "model-trained.onnx"
    args.plugin_file_list = ""
    args.min_shape = "x:1x1x28x28"
    args.opt_shape = "x:2x1x28x28"
    args.max_shape = "x:4x1x28x28"
    args.infer_shape = ""  # Use shape from file
    args.data_file = Path("data.npz")
    args.ort_baseline = False
    args.tune_type_list = ["CONVOLUTION", "MATRIX_MULTIPLY"]
    args.test_performance = True
    args.output_file = Path("report.txt")

    model = FP16Tuning(args)
    model.tune()

    print("Finish")
