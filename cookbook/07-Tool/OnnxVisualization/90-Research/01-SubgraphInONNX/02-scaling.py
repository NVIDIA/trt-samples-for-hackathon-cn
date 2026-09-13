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
"""What does it cost / save to fold N repeated blocks into one ONNX `Loop`?

`main.py` shows that TensorRT can build both the flat graph and the equivalent
`Loop` graph. This script sweeps the number of repeated blocks N and measures,
for both representations:

* ONNX node count and file size
* TensorRT network layer count and build time
* serialized engine size
* inference latency

The `Loop` version stores the per-iteration weights stacked into a single
initializer and picks them with `Gather(W, iter)`, so the two models are
numerically identical but the loop body exists only once.
"""

import time
from pathlib import Path

import numpy as np
import onnx
import onnx.helper as oh
import tensorrt as trt
from cuda.bindings import runtime as cudart

np.random.seed(31193)

N_BLOCK_LIST = [2, 4, 8, 16, 32, 64, 128, 256]
N_C = 256  # Feature width
N_B = 8  # Batch size
OPSET = 17
N_WARMUP = 20
N_TEST = 100

output_path = Path(__file__).parent

# ================================================================ Model builders

def make_weight(n_block: int) -> list:
    """Per-block weights, already transposed so that y = x @ W + b."""
    weight_list = []
    for _ in range(n_block):
        weight_list.append(dict(
            w1=(np.random.rand(N_C, N_C).astype(np.float32) - 0.5) / N_C ** 0.5,
            b1=(np.random.rand(N_C).astype(np.float32) - 0.5),
            w2=(np.random.rand(N_C, N_C).astype(np.float32) - 0.5) / N_C ** 0.5,
            b2=(np.random.rand(N_C).astype(np.float32) - 0.5),
        ))
    return weight_list

def build_flat_model(weight_list: list, onnx_file: Path) -> None:
    """N unrolled `MatMul -> Add -> Relu -> MatMul -> Add -> Relu` blocks."""
    node_list, initializer_list = [], []
    tensor_name = "x"
    for i, weight in enumerate(weight_list):
        for key in ["w1", "b1", "w2", "b2"]:
            value = weight[key]
            initializer_list.append(oh.make_tensor(f"{key}_{i}", onnx.TensorProto.FLOAT, value.shape, value.reshape(-1)))
        node_list += [
            oh.make_node("MatMul", [tensor_name, f"w1_{i}"], [f"t0_{i}"], f"MatMul1_{i}"),
            oh.make_node("Add", [f"t0_{i}", f"b1_{i}"], [f"t1_{i}"], f"Add1_{i}"),
            oh.make_node("Relu", [f"t1_{i}"], [f"t2_{i}"], f"Relu1_{i}"),
            oh.make_node("MatMul", [f"t2_{i}", f"w2_{i}"], [f"t3_{i}"], f"MatMul2_{i}"),
            oh.make_node("Add", [f"t3_{i}", f"b2_{i}"], [f"t4_{i}"], f"Add2_{i}"),
            oh.make_node("Relu", [f"t4_{i}"], [f"t5_{i}"], f"Relu2_{i}"),
        ]
        tensor_name = f"t5_{i}"
    node_list.append(oh.make_node("Identity", [tensor_name], ["y"], "OutputIdentity"))

    graph = oh.make_graph(
        node_list,
        "Flat",
        [oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [N_B, N_C])],
        [oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [N_B, N_C])],
        initializer_list,
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", OPSET)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, onnx_file)
    return

def build_loop_model(weight_list: list, onnx_file: Path) -> None:
    """One `Loop` node whose body gathers the weights of iteration `iter`."""
    n_block = len(weight_list)
    initializer_list = []
    for key in ["w1", "b1", "w2", "b2"]:
        value = np.stack([weight[key] for weight in weight_list])
        initializer_list.append(oh.make_tensor(key.upper(), onnx.TensorProto.FLOAT, value.shape, value.reshape(-1)))
    initializer_list += [
        oh.make_tensor("trip_count", onnx.TensorProto.INT64, [], [n_block]),
        oh.make_tensor("cond", onnx.TensorProto.BOOL, [], [True]),
    ]

    body_node_list = [
        oh.make_node("Gather", ["W1", "iter"], ["w1_i"], "GatherW1", axis=0),
        oh.make_node("Gather", ["B1", "iter"], ["b1_i"], "GatherB1", axis=0),
        oh.make_node("Gather", ["W2", "iter"], ["w2_i"], "GatherW2", axis=0),
        oh.make_node("Gather", ["B2", "iter"], ["b2_i"], "GatherB2", axis=0),
        oh.make_node("MatMul", ["x_in", "w1_i"], ["t0"], "MatMul1"),
        oh.make_node("Add", ["t0", "b1_i"], ["t1"], "Add1"),
        oh.make_node("Relu", ["t1"], ["t2"], "Relu1"),
        oh.make_node("MatMul", ["t2", "w2_i"], ["t3"], "MatMul2"),
        oh.make_node("Add", ["t3", "b2_i"], ["t4"], "Add2"),
        oh.make_node("Relu", ["t4"], ["x_out"], "Relu2"),
        oh.make_node("Identity", ["cond_in"], ["cond_out"], "CondPassThrough"),
    ]
    body_graph = oh.make_graph(
        body_node_list,
        "LoopBody",
        [
            oh.make_tensor_value_info("iter", onnx.TensorProto.INT64, []),
            oh.make_tensor_value_info("cond_in", onnx.TensorProto.BOOL, []),
            oh.make_tensor_value_info("x_in", onnx.TensorProto.FLOAT, [N_B, N_C]),
        ],
        [
            oh.make_tensor_value_info("cond_out", onnx.TensorProto.BOOL, []),
            oh.make_tensor_value_info("x_out", onnx.TensorProto.FLOAT, [N_B, N_C]),
        ],
    )
    graph = oh.make_graph(
        [oh.make_node("Loop", ["trip_count", "cond", "x"], ["y"], "MyLoop", body=body_graph)],
        "Loop",
        [oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [N_B, N_C])],
        [oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [N_B, N_C])],
        initializer_list,
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", OPSET)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, onnx_file)
    return

# ================================================================ Measurement

def measure(onnx_file: Path, input_data: np.ndarray) -> dict:
    """Build an engine from the ONNX file, then time build and inference."""
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    t0 = time.time()
    if not parser.parse_from_file(str(onnx_file)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"Failed parsing {onnx_file}")
    parse_time = time.time() - t0
    n_layer = network.num_layers

    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape("x", [N_B, N_C], [N_B, N_C], [N_B, N_C])
    config.add_optimization_profile(profile)

    t0 = time.time()
    engine_bytes = builder.build_serialized_network(network, config)
    build_time = time.time() - t0
    assert engine_bytes is not None, f"Failed building engine from {onnx_file}"

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()
    context.set_input_shape("x", input_data.shape)

    output_host = np.empty([N_B, N_C], dtype=np.float32)
    _, input_device = cudart.cudaMalloc(input_data.nbytes)
    _, output_device = cudart.cudaMalloc(output_host.nbytes)
    cudart.cudaMemcpy(input_device, input_data.ctypes.data, input_data.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    context.set_tensor_address("x", int(input_device))
    context.set_tensor_address("y", int(output_device))

    for _ in range(N_WARMUP):
        context.execute_async_v3(0)
    cudart.cudaStreamSynchronize(0)
    t0 = time.time()
    for _ in range(N_TEST):
        context.execute_async_v3(0)
    cudart.cudaStreamSynchronize(0)
    latency = (time.time() - t0) / N_TEST * 1000

    cudart.cudaMemcpy(output_host.ctypes.data, output_device, output_host.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    cudart.cudaFree(input_device)
    cudart.cudaFree(output_device)

    model = onnx.load(onnx_file)
    n_node = len(model.graph.node)
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                n_node += len(attribute.g.node)
    return dict(
        n_onnx_node=n_node,
        onnx_size=onnx_file.stat().st_size / 1024,
        n_trt_layer=n_layer,
        parse_time=parse_time,
        build_time=build_time,
        engine_size=engine_bytes.nbytes / 1024,
        latency=latency,
        output=output_host,
    )

# ================================================================ Entrance

def main() -> None:
    input_data = (np.random.rand(N_B, N_C).astype(np.float32) - 0.5)
    row_list = []
    for n_block in N_BLOCK_LIST:
        weight_list = make_weight(n_block)
        flat_file = output_path / f"scaling-flat-{n_block:02d}.onnx"
        loop_file = output_path / f"scaling-loop-{n_block:02d}.onnx"
        build_flat_model(weight_list, flat_file)
        build_loop_model(weight_list, loop_file)

        flat = measure(flat_file, input_data)
        loop = measure(loop_file, input_data)
        diff = np.max(np.abs(flat["output"] - loop["output"]))
        row_list.append((n_block, flat, loop, diff))
        print(f"N={n_block:3d} done, max|flat - loop| = {diff:.3e}")
        # The N=256 pair alone is ~260 MB, delete them right away rather than leaving them around
        flat_file.unlink()
        loop_file.unlink()

    header = (f"{'N':>4} | {'Node(F)':>8}{'Node(L)':>8} | {'Layer(F)':>9}{'Layer(L)':>9} | {'Parse(F)s':>10}{'Parse(L)s':>10} | "
              f"{'Build(F)s':>10}{'Build(L)s':>10} | {'Eng(F)MB':>10}{'Eng(L)MB':>10} | {'Lat(F)ms':>10}{'Lat(L)ms':>10} | {'MaxDiff':>10}")
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for n_block, flat, loop, diff in row_list:
        print(f"{n_block:>4} | {flat['n_onnx_node']:>8}{loop['n_onnx_node']:>8} | {flat['n_trt_layer']:>9}{loop['n_trt_layer']:>9} | "
              f"{flat['parse_time']:>10.3f}{loop['parse_time']:>10.3f} | "
              f"{flat['build_time']:>10.2f}{loop['build_time']:>10.2f} | {flat['engine_size']/1024:>10.1f}{loop['engine_size']/1024:>10.1f} | "
              f"{flat['latency']:>10.3f}{loop['latency']:>10.3f} | {diff:>10.3e}")
    print("=" * len(header))
    print("(F) = flat / unrolled graph, (L) = single `Loop` node with stacked weights")
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
