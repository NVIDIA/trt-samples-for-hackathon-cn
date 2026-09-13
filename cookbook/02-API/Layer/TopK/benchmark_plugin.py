# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
"""`ITopKLayer` against the `TopkLastDim` plugin that ships with TensorRT.

`main.py` is about the layer's API. This file answers a different question: TensorRT 11.1
ships a **`TopkLastDim`** plugin in its own plugin library, built on the AIR (Adaptive
Iterative Radix) sort kernel from TensorRT-LLM. It covers the same operation as
`ITopKLayer`, so the obvious question is when it is worth reaching for.

Both are measured here on the same data, on the last axis, and the answer is checked
against NumPy so a faster wrong answer cannot win.

Two things to know before using the plugin:

+ **Indices come out `int32`** -- from the plugin *and*, as measured below, from
  `ITopKLayer` in this configuration. The ONNX `TopK` specification says `int64`, so a graph
  parsed from ONNX needs a `Cast` either way; the plugin's README calls this out as a known
  deviation, but it is not a difference between the two implementations.
+ **`ITopKLayer` has a hard limit of k <= 3840**; the plugin does not. That alone decides
  the question for large k, and it is checked below rather than quoted.

The plugin is registered by `init_libnvinfer_plugins`, i.e. by `REGISTER_TENSORRT_PLUGIN`
inside TensorRT's own plugin library -- nothing to build, nothing to load.
"""

import time
from collections import OrderedDict

import cuda.bindings.runtime as cudart
import numpy as np
import tensorrt as trt

from tensorrt_cookbook import case_mark

np.random.seed(31193)

N_ROW = 256
N_COLUMN = 16384  # A long last dimension, which is what the plugin is specialised for
N_WARMUP = 10
N_INFERENCE = 50

result = OrderedDict()

def build_engine_with_layer(k: int) -> bytes:
    """The native `ITopKLayer`."""
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()

    input_tensor = network.add_input("input", trt.float32, [N_ROW, N_COLUMN])
    layer = network.add_topk(input_tensor, trt.TopKOperation.MAX, k, 1 << 1)  # axis 1 = last
    if layer is None:
        # Past the k limit `add_topk` returns None immediately -- the network API refuses,
        # the builder is never reached, and there is no error message to read.
        return None
    for index, name in enumerate(["values", "indices"]):
        layer.get_output(index).name = name
        network.mark_output(layer.get_output(index))
    engine_bytes = builder.build_serialized_network(network, builder_config)
    return None if engine_bytes is None else bytes(engine_bytes)

def build_engine_with_plugin(k: int) -> bytes:
    """The `TopkLastDim` plugin from TensorRT's own plugin library."""
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()

    plugin_creator = trt.get_plugin_registry().get_creator("TopkLastDim", "1", "")
    assert plugin_creator is not None, "TopkLastDim is not in this TensorRT's plugin library"
    field_list = [
        trt.PluginField("type_id", np.int32(int(trt.float32)), trt.PluginFieldType.INT32),
        trt.PluginField("k", np.int32(k), trt.PluginFieldType.INT32),
        trt.PluginField("is_largest", np.int32(1), trt.PluginFieldType.INT32),
        trt.PluginField("axis", np.int32(1), trt.PluginFieldType.INT32),
    ]
    plugin = plugin_creator.create_plugin("TopkLastDim", trt.PluginFieldCollection(field_list), trt.TensorRTPhase.BUILD)

    input_tensor = network.add_input("input", trt.float32, [N_ROW, N_COLUMN])
    layer = network.add_plugin_v3([input_tensor], [], plugin)
    for index, name in enumerate(["values", "indices"]):
        layer.get_output(index).name = name
        network.mark_output(layer.get_output(index))
    engine_bytes = builder.build_serialized_network(network, builder_config)
    return None if engine_bytes is None else bytes(engine_bytes)

def run(engine_bytes: bytes, input_data: np.ndarray) -> tuple:
    """Run the engine and return `(values, indices, median_latency_ms)`."""
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    name_list = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    buffer = OrderedDict()
    for name in name_list:
        shape = context.get_tensor_shape(name)
        data_type = engine.get_tensor_dtype(name)
        host = np.empty(shape, dtype=trt.nptype(data_type))
        device = cudart.cudaMalloc(max(host.nbytes, 1))[1]
        buffer[name] = [host, device, host.nbytes]
        context.set_tensor_address(name, device)

    cudart.cudaMemcpy(buffer["input"][1], np.ascontiguousarray(input_data).ctypes.data, buffer["input"][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    for _ in range(N_WARMUP):
        context.execute_async_v3(0)
    cudart.cudaStreamSynchronize(0)
    latency_list = []
    for _ in range(N_INFERENCE):
        t0 = time.time()
        context.execute_async_v3(0)
        cudart.cudaStreamSynchronize(0)
        latency_list.append((time.time() - t0) * 1000)

    for name in ["values", "indices"]:
        cudart.cudaMemcpy(buffer[name][0].ctypes.data, buffer[name][1], buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    values = buffer["values"][0].copy()
    indices = buffer["indices"][0].copy()

    for _, device, _ in buffer.values():
        cudart.cudaFree(device)
    return values, indices, float(np.median(latency_list))

@case_mark
def case_compare(k: int) -> None:
    """Same data, same axis, both implementations, checked against NumPy."""
    input_data = np.random.rand(N_ROW, N_COLUMN).astype(np.float32)
    reference = -np.sort(-input_data, axis=1)[:, :k]

    row = OrderedDict()
    for name, builder_function in [("ITopKLayer", build_engine_with_layer), ("TopkLastDim plugin", build_engine_with_plugin)]:
        engine_bytes = builder_function(k)
        if engine_bytes is None:
            print(f"    k={k:<6} {name:<20} FAILED TO BUILD")
            row[name] = None
            continue
        values, indices, latency_ms = run(engine_bytes, input_data)
        max_difference = float(np.max(np.abs(values - reference)))
        index_dtype = indices.dtype
        print(f"    k={k:<6} {name:<20} latency={latency_ms:8.3f} ms, max|diff vs numpy|={max_difference:.3e}, indices dtype={index_dtype}")
        assert max_difference < 1e-6, f"{name} does not agree with NumPy"
        row[name] = (latency_ms, str(index_dtype))
    result[k] = row
    return

@case_mark
def case_layer_k_limit() -> None:
    """`ITopKLayer` has a documented ceiling on k. Find it rather than quote it."""
    for k in [3840, 3841]:
        engine_bytes = build_engine_with_layer(k)
        print(f"    ITopKLayer with k={k}: {'built' if engine_bytes else 'REFUSED (network.add_topk returned None)'}")
        result.setdefault("k_limit", {})[k] = engine_bytes is not None
    return

def main() -> None:
    for k in [8, 64, 1024]:
        case_compare(k)
    case_layer_k_limit()

    print("\n" + "=" * 84)
    print(f"{'k':>8}{'ITopKLayer (ms)':>20}{'TopkLastDim (ms)':>20}{'speed-up':>12}{'indices dtype':>22}")
    print("-" * 84)
    for k, row in result.items():
        if k == "k_limit" or row.get("ITopKLayer") is None or row.get("TopkLastDim plugin") is None:
            continue
        layer_ms, layer_dtype = row["ITopKLayer"]
        plugin_ms, plugin_dtype = row["TopkLastDim plugin"]
        print(f"{k:>8}{layer_ms:>20.3f}{plugin_ms:>20.3f}{layer_ms / plugin_ms:>11.2f}x{f'{layer_dtype} / {plugin_dtype}':>22}")
    print("=" * 84)
    limit = result.get("k_limit", {})
    print(f"ITopKLayer k=3840 builds: {limit.get(3840)}, k=3841 builds: {limit.get(3841)}")
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
