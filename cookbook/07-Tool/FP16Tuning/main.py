# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import time
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
from tabulate import tabulate
from tensorrt_cookbook import TRTWrapperV1, cookbook_path, check_array, initialize_random_seed, layer_type_to_layer_type_name, compare_sets, get_cookbook_logger, parse_onnx
from tqdm import tqdm
import datetime

LOG_FILE = Path("FP16Tuning.log")
if LOG_FILE.exists():
    LOG_FILE.unlink()

logger = get_cookbook_logger(log_file=LOG_FILE)

class FP16Tuning:

    def __init__(self, **kwargs):
        logger.info("Start session %s", "=" * 64)
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.result_table = {}
        self.rng = initialize_random_seed()
        self.temp_dir = Path("FP16TuningTemp")
        self.time_cache_file = self.temp_dir / "model.TimingCache"
        self.trt_file = self.temp_dir / "model.trt"
        Path(self.temp_dir).mkdir(exist_ok=True, parents=True)

        # Get input information from ONNX file and align with command arguments ========================================
        graph = gs.import_onnx(onnx.load(self.onnx_file))
        dtype_dict = {tensor.name: tensor.dtype for tensor in graph.inputs}
        name_set_by_model = {tensor.name for tensor in graph.inputs}
        output_tensor_name_list = [tensor.name for tensor in graph.outputs]
        del graph
        logger.info("input tensor name in model: %s", sorted(name_set_by_model))
        logger.info("output tensor name in model: %s", output_tensor_name_list)

        if self.focus_tensor is None:
            self.focus_tensor = output_tensor_name_list[0]
            logger.info("focus_tensor set to: %s", self.focus_tensor)
        elif self.focus_tensor not in set(output_tensor_name_list):
            raise ValueError(f"focus_tensor `{self.focus_tensor}` is not in model outputs: {output_tensor_name_list}")
        else:
            logger.info("focus_tensor set to: %s", self.focus_tensor)

        name_set_by_arg = {item.split(":")[0] for item in self.opt_shape.split(",")}
        logger.info("input tensor name in command: %s", sorted(name_set_by_arg))
        if not compare_sets(name_set_by_model, name_set_by_arg, "ONNX file", "command", logger):
            raise ValueError("Input tensor names mismatch between ONNX file and command")

        self.name_set = name_set_by_arg
        self.shape_dict = {name: [] for name in self.name_set}
        for item in [self.min_shape, self.opt_shape, self.max_shape]:
            for item in item.split(","):
                name, shape = item.split(":")
                self.shape_dict[name].append([int(x) for x in shape.split('x')])
                # TODO: check shape range validity

        # Get target layer =============================================================================================
        tw = self._build_trt_network()  # Call this much earlier than call of `_single_run` since we need to dump information from the network

        # Add layer name for baseline, add unicode characters to make it different from any normal layer name in the model
        self.header_layer_name_list = ["Pure FP32 🟩", "Pure FP16 🟦", "FP16 + ForceFP32 🟪"]
        self.target_layer_name_list = []
        for i in range(tw.network.num_layers):
            layer = tw.network.get_layer(i)
            if layer_type_to_layer_type_name(layer.type) in self.tune_type_list:
                self.target_layer_name_list.append(layer.name)

        logger.info("All layers can be tuned: %s", self.target_layer_name_list)
        if len(self.specify_layer_name_list) > 0:
            for layer_name in self.specify_layer_name_list:
                if layer_name not in self.target_layer_name_list:
                    logger.warning("Specified layer %s is not in tunable layer list", layer_name)
            self.target_layer_name_list = self.specify_layer_name_list

        for layer_name in self.skip_layer_name_list:
            if layer_name not in self.target_layer_name_list:
                logger.warning("Skipped layer %s is not in tunable layer list", layer_name)
        for layer_name in self.force_fp32_layer_name_list:
            if layer_name not in self.target_layer_name_list:
                logger.warning("Forced FP32 layer %s is not in tunable layer list", layer_name)

        logger.info("Layers specified [%3d]: %s", len(self.specify_layer_name_list), self.specify_layer_name_list)
        logger.info("Layers skipped [%3d]: %s", len(self.skip_layer_name_list), self.skip_layer_name_list)
        logger.info("Layers forced in FP32 [%3d]: %s", len(self.force_fp32_layer_name_list), self.force_fp32_layer_name_list)
        logger.info("Layers could be tuned [%3d]: %s", len(self.target_layer_name_list), self.target_layer_name_list)

        self.target_layer_name_list = self.header_layer_name_list + self.target_layer_name_list

        # Get ground truth data and inference shape ====================================================================
        input_data = {}
        if self.data_file and self.data_file.exists():  # Use dumped data
            # `self.infer_shape` from command will be ignored and covered in this case
            np_data = np.load(self.data_file)
            name_set_by_data = set(np_data.keys())
            logger.info("input tensor name in data file: %s", sorted(name_set_by_data))
            if not compare_sets(self.name_set, name_set_by_data, "ONNX file / command", "data file", logger):
                raise ValueError("Input tensor names mismatch between ONNX file/command and data file")

            self.infer_shape = ""
            for name, data in np_data.items():
                assert len(data.shape) == len(self.shape_dict[name][0]), f"Rank of input tensor {name} in data file is not equal to it in command line"
                for np_length, min_length, max_length in zip(data.shape, self.shape_dict[name][0], self.shape_dict[name][2]):
                    assert min_length <= np_length <= max_length, f"Shape of input tensor {name} in data file {data.shape} is not in range [{self.shape_dict[name][0]}, {self.shape_dict[name][2]}]"
                input_data[name] = data
                self.infer_shape += f"{name}:{'x'.join([str(x) for x in data.shape])},"
            self.infer_shape = self.infer_shape[:-1]

        else:  # Use random data
            if self.infer_shape is None:
                self.infer_shape = self.opt_shape
            # TODO： check infer_shape validity
            for shape_dict in self.infer_shape.split(","):
                name, shape = shape_dict.split(":")
                shape = [int(x) for x in shape.split('x')]
                dtype = np.dtype(dtype_dict[name])
                if np.issubdtype(dtype, np.integer):
                    input_data[name] = self.rng.integers(1, 1024, size=shape, dtype=dtype)
                else:
                    input_data[name] = self.rng.random(shape).astype(dtype)

        # Save binary data for trtexec
        self.input_data = input_data
        for name, data in self.input_data.items():
            data.tofile(self.temp_dir / (name + ".bin"))

        # Run Pure FP32 inference and save reference output data
        self._single_run(self.target_layer_name_list[0], False, tw)

        # Update report
        self._generate_markdown()

    def tune(self):
        logger.info("Tune %s", "=" * 64)

        # Run FP16 inference without forcing FP32 layers to get the performance baseline
        original_force_fp32_layer_name_list = self.force_fp32_layer_name_list.copy()
        self.force_fp32_layer_name_list = []
        self._single_run(self.target_layer_name_list[1])
        self.force_fp32_layer_name_list = original_force_fp32_layer_name_list

        # Run other cases
        assert self.max_tune_layers >= 0, "max_tune_layers should be non-negative"
        for layer_name in tqdm(self.target_layer_name_list[2:(2 + self.max_tune_layers)]):
            try:
                logger.info("[Start] %s", layer_name)
                start_time = time.perf_counter()
                self._single_run(layer_name)
                duration_s = time.perf_counter() - start_time
                logger.info("[End  ] %s in %.6fs", layer_name, duration_s)
            except Exception:
                logger.exception("Error build with layer %s", layer_name)

            # Update report file every time
            self._generate_markdown()

    def _generate_markdown(self):
        headers = [
            "No.",
            "LayerName",
            "TensorName",  # Output tensor name
            "GPUTime (ms)",  # GPU Compute Time
            "MaxAbsError",
            "MeanAbsError",
            "BestPerf",
            "BestAcc",
        ]
        rows = []
        rank_candidates = []
        baseline_layers = set(self.header_layer_name_list)

        for index, (row_layer_name, tensor_dict) in enumerate(self.result_table.items()):
            if len(tensor_dict) == 0:
                rows.append([index + 1, row_layer_name, "-", "-", "", "", "", ""])
                continue

            for tensor_name, value in tensor_dict.items():
                row_time, row_max_error, row_mean_error = value

                row_time_value = row_time if isinstance(row_time, (int, float, np.floating)) else None
                row_max_error_value = row_max_error if isinstance(row_max_error, (int, float, np.floating)) else None

                if row_time_value is not None:
                    row_time_text = f"{float(row_time_value):.3f}"
                else:
                    row_time_text = str(row_time)

                if row_max_error_value is not None:
                    row_max_error_text = f"{float(row_max_error_value):.4e}"
                else:
                    row_max_error_text = str(row_max_error)

                if isinstance(row_mean_error, (int, float, np.floating)):
                    row_mean_error_text = f"{float(row_mean_error):.4e}"
                else:
                    row_mean_error_text = str(row_mean_error)

                rows.append([
                    index + 1,
                    row_layer_name,
                    tensor_name,
                    row_time_text,
                    row_max_error_text,
                    row_mean_error_text,
                    "",
                    "",
                ])

                if row_layer_name not in baseline_layers:
                    rank_candidates.append({
                        "row_index": len(rows) - 1,
                        "time": float(row_time_value) if row_time_value is not None else None,
                        "error": float(row_max_error_value) if (row_max_error_value is not None and tensor_name == self.focus_tensor) else None,
                    })

        def _rank_text(rank: int):
            if rank <= 5:
                return f"{rank} 🔴"
            if rank <= 10:
                return f"{rank} 🟠"
            if rank <= 15:
                return f"{rank} 🟡"
            if rank <= 20:
                return f"{rank} 🟢"
            # if rank <= 25:
            #     return f"{rank} 🔵"
            # if rank <= 30:
            #     return f"{rank} 🟣"
            return str(rank)

        perf_sorted = sorted(
            [item for item in rank_candidates if item["time"] is not None],
            key=lambda item: item["time"],
        )
        for rank, item in enumerate(perf_sorted, start=1):
            rows[item["row_index"]][6] = _rank_text(rank)

        acc_sorted = sorted(
            [item for item in rank_candidates if item["error"] is not None],
            key=lambda item: item["error"],
        )
        for rank, item in enumerate(acc_sorted, start=1):
            rows[item["row_index"]][7] = _rank_text(rank)

        if len(rows) == 0:
            rows.append(["(empty)", "-", "-", "-", "", "", "", ""])

        header_layer_count = len(self.header_layer_name_list)
        tunable_layers = self.target_layer_name_list[header_layer_count:(header_layer_count + self.max_tune_layers)]
        skipped_or_forced_layers = set(self.skip_layer_name_list) | set(self.force_fp32_layer_name_list)
        tuned_layer_count = sum(1 for layer_name in tunable_layers if layer_name not in skipped_or_forced_layers)

        ss = "# FP16 Tuning Report\n\n"
        ss += f"+ Generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        ss += f"+ Layers specified [{len(self.specify_layer_name_list):3d}]: {self.specify_layer_name_list}\n"
        ss += f"+ Layers skipped [{len(self.skip_layer_name_list):3d}] : {self.skip_layer_name_list}\n"
        ss += f"+ Layers forced in FP32 [{len(self.force_fp32_layer_name_list):3d}]: {self.force_fp32_layer_name_list}\n"
        ss += f"+ Layers could be tuned [{len(self.target_layer_name_list):3d}]: {self.target_layer_name_list}\n"
        ss += f"+ Layers actually tune in this session: {tuned_layer_count}\n\n"
        ss += f"+ Focus tensor for BestAcc ranking: {self.focus_tensor}\n"
        ss += tabulate(rows, headers=headers, tablefmt="github", stralign="left", numalign="right")
        ss += "\n"

        best_acc_layer_names = [rows[item["row_index"]][1] for item in acc_sorted[:20]]
        ss += "\n+ Layers performs best in improving accuracy (sorted by `MaxAbsError`):\n\n"
        for start in range(0, 20, 5):
            group = best_acc_layer_names[start:start + 5]
            if len(group) == 0:
                break
            ss += ", ".join([f'\"{layer_name}\"' for layer_name in group]) + ",\n"

        with open(self.output_file, "w") as ff:
            ff.write(ss)

    def _build_trt_network(self):
        tw = TRTWrapperV1(plugin_file_list=self.plugin_file_list)
        parse_onnx(self.onnx_file, tw.logger, tw.network, tw.config)

        for i in range(tw.network.num_inputs):
            input_tensor = tw.network.get_input(i)
            name = input_tensor.name
            tw.profile.set_shape(input_tensor.name, *self.shape_dict[name])
        tw.config.add_optimization_profile(tw.profile)

        return tw

    def _single_run(self, layer_name: str = "", b_fp16: bool = True, tw: TRTWrapperV1 = None):
        if layer_name in self.skip_layer_name_list:
            logger.info("%s skipped", layer_name)
            self.result_table[layer_name] = {}
            self.result_table[layer_name]["Skipped ⏩"] = ["-", "-", "-"]
            return
        if layer_name in self.force_fp32_layer_name_list:
            logger.info("%s forced FP32", layer_name)
            self.result_table[layer_name] = {}
            self.result_table[layer_name]["Forced FP32 🔒"] = ["-", "-", "-"]
            return

        if tw is None:  # Use `tw` from arguments for running FP32 as ground truth, or construct a new one while tuning in FP16
            tw = self._build_trt_network()

        time_cache = b""
        if self.time_cache_file.exists():
            with open(self.time_cache_file, "rb") as f:
                time_cache = f.read()
        cache = tw.config.create_timing_cache(time_cache)
        tw.config.set_timing_cache(cache, False)

        if b_fp16:
            tw.config.set_flag(trt.BuilderFlag.FP16)
            tw.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            for i in range(tw.network.num_layers):
                layer = tw.network.get_layer(i)
                if layer.name == layer_name or layer.name in self.force_fp32_layer_name_list:
                    layer.precision = trt.float32

        tw.build()
        tw.serialize_engine(self.trt_file, True)  # Remove previous engine file

        timing_cache = tw.config.get_timing_cache()
        timing_cache_buffer = timing_cache.serialize()
        with open(self.time_cache_file, "wb") as f:
            f.write(timing_cache_buffer)

        tw.setup(self.input_data, b_print_io=False)
        tw.infer(b_print_io=False)

        if not b_fp16:
            self.output_data = {}
            for name, data in tw.buffer.items():
                if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                    self.output_data[name] = data[0]

        gpu_time = "None"
        if self.test_performance:
            shape_str = ""
            for name in self.name_set:
                shape_str += f"{name}:{self.temp_dir}/{name}.bin,"
            command = [
                "trtexec",
                f"--loadEngine={self.trt_file}",
                "--useSpinWait",
                "--noDataTransfers",
                "--useCudaGraph",
                "--noTF32",
                "--builderOptimizationLevel=5",
                "--maxAuxStreams=5",
                f"--shapes={self.infer_shape}",
                f"--loadInputs={shape_str[:-1]}",
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            output_text = result.stdout + ("\n" + result.stderr if result.stderr else "")

            for line in output_text.splitlines():
                logger.debug(line.rstrip("\n"))
                if "[I] GPU Compute Time" in line:
                    gpu_time = float(line.split("ms")[3].split("=")[1])
            if result.returncode != 0:
                logger.warning("trtexec failed with code %d for layer %s", result.returncode, layer_name)

        # Compare output
        self.result_table[layer_name] = {}
        for name, data in tw.buffer.items():
            if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                now_data = data[0]
                ref_data = self.output_data[name]
                if False:  # For debug
                    check_array(now_data, ref_data, weak=True, info=name)
                max_error = np.max(np.abs(now_data - ref_data))
                mean_error = np.mean(np.abs(now_data - ref_data))
                self.result_table[layer_name][name] = [gpu_time, max_error, mean_error]

        return

if __name__ == "__main__":

    # Configure node name here!
    skip_layer_name_list = []
    force_fp32_layer_name_list = []

    # Do not forget the comma "," at the end of each line below
    FP16Tuning(
        onnx_file=cookbook_path("00-Data", "model", "model-trained.onnx"),
        plugin_file_list=[],
        data_file=cookbook_path("00-Data", "data", "InferenceData.npz"),
        output_file="FP16Tuning-report.md",
        # Shape in trtexec format, for example: x:4x64x64,y:4,z:"
        min_shape="x:1x1x28x28",
        opt_shape="x:2x1x28x28",
        max_shape="x:4x1x28x28",
        # Use  shape in `data_file` or value of `opt_shape` by default
        infer_shape=None,
        # Layer types to be tuned: "CONVOLUTION", "MATRIX_MULTIPLY", ...
        tune_type_list=["CONVOLUTION", "MATRIX_MULTIPLY"],
        test_performance=True,
        max_tune_layers=1000,
        # The layers we want to tune. No other layers will be tuned if this is set
        specify_layer_name_list=[],
        # The layers we never try to set back to FP32 (maybe due to large loss of performance)
        skip_layer_name_list=skip_layer_name_list,
        # The layers must stay in FP32
        force_fp32_layer_name_list=force_fp32_layer_name_list,
        # The output tensor used for BestAcc ranking, using model first output when set to None
        focus_tensor=None,
        # Useless yet, TODO: selecting more than one layers in one session
        greedy_approach=True,
    ).tune()

    logger.info("Finish")
    print("Finish")
