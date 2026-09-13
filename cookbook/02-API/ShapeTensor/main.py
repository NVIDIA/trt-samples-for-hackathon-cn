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
"""Shape tensors: the input whose *values* are shapes, and everything that follows from that.

TensorRT has two kinds of input, and almost every rule differs between them:

|                        | execution tensor          | **shape tensor**                    |
| ---------------------- | ------------------------- | ----------------------------------- |
| what varies at runtime | its shape                 | its **values**                      |
| lives on               | device                    | **host**                            |
| profile method         | `set_shape`               | **`set_shape_input`**               |
| context method         | `set_input_shape`         | **`set_tensor_address`** (host ptr) |
| identified by          | -                         | `engine.is_shape_inference_io(name)` |

The two `set_*_shape*` pairs are the trap: `set_shape` / `set_input_shape` name the *shape*
of a tensor, `set_shape_input` / a host address name the *contents* of one. The names are
nearly identical, the argument types are nearly identical, and using the wrong one gives an
error that talks about the other kind.

Cases:

1. `case_identify`            - which inputs are shape tensors, and how to ask.
2. `case_int32_and_int64`     - both widths work; what changes is the buffer you must hand over.
3. `case_zero_dimensional`    - a 0-D shape tensor, i.e. a single scalar that is a shape.
4. `case_host_not_device`     - a shape tensor's address must be **host** memory; giving it
                                device memory is the mistake this case makes on purpose.
5. `case_wrong_api`           - `set_input_shape` on a shape tensor, and `set_shape` in the
                                profile, and what each one actually reports.
6. `case_shape_inference`     - `infer_shapes` before `execute_async_v3`, and what it returns
                                when something is still unset.
"""

import subprocess
import sys
import textwrap
from collections import OrderedDict
from pathlib import Path

import cuda.bindings.runtime as cudart
import numpy as np
import tensorrt as trt

from tensorrt_cookbook import case_mark

np.random.seed(31193)

N_DIMENSION = 3
output_path = Path(__file__).parent
result = OrderedDict()

def build_engine(shape_dtype=trt.int32, b_zero_dimensional: bool = False):
    """`data` is an execution tensor, `newShape` is a shape tensor feeding an IShuffleLayer.

    The shape tensor is what `IShuffleLayer::setInput(1, ...)` consumes: its *values* become
    the reshape target, so TensorRT must know them on the host at build and at run time.
    """
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    profile = builder.create_optimization_profile()
    builder_config = builder.create_builder_config()

    if b_zero_dimensional:
        # A 0-D shape tensor: one scalar. `IFillLayer` in LINSPACE mode takes its `alpha`
        # (start) as a 0-D shape input, which is the smallest real use of one.
        #
        # This graph has NO execution-tensor input on purpose. An earlier version kept an
        # unused `data` input around for symmetry and segfaulted inside libnvinfer at
        # `execute_async_v3` -- an input the graph never consumes still has to be bound, and
        # a network whose only real input is a shape tensor is a legitimate thing to build.
        start = network.add_input("start", trt.int32, [])
        layer_fill = network.add_fill([8], trt.FillOperation.LINSPACE, trt.int32)
        layer_fill.set_input(1, start)
        # LINSPACE wants alpha (start) at rank 0 and delta at rank 1 -- one delta per output
        # dimension. Passing a rank-0 delta fails with "requires that input at index 2 have rank 1".
        layer_fill.set_input(2, network.add_constant([1], np.array([1], dtype=np.int32)).get_output(0))
        output = layer_fill.get_output(0)
        profile.set_shape_input("start", [0], [5], [100])
    else:
        data = network.add_input("data", trt.float32, [-1, -1, -1])
        new_shape = network.add_input("newShape", shape_dtype, [N_DIMENSION])
        layer = network.add_shuffle(data)
        layer.set_input(1, new_shape)
        output = layer.get_output(0)
        profile.set_shape("data", [1, 1, 1], [2, 3, 4], [4, 6, 8])
        # `set_shape_input` takes the VALUES the tensor may hold, not its shape
        profile.set_shape_input("newShape", [1, 1, 1], [2, 3, 4], [4, 6, 8])

    output.name = "output"
    network.mark_output(output)
    builder_config.add_optimization_profile(profile)
    engine_bytes = builder.build_serialized_network(network, builder_config)
    assert engine_bytes is not None, "Failed building engine"
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_bytes)
    return engine

# ================================================================ Cases

@case_mark
def case_identify() -> None:
    """Ask the engine which of its inputs are shape tensors."""
    engine = build_engine()
    print(f"    {'name':<10}{'mode':<8}{'dtype':<16}{'build shape':<16}{'is_shape_inference_io'}")
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        print(f"    {name:<10}{str(engine.get_tensor_mode(name))[13:]:<8}{str(engine.get_tensor_dtype(name))[9:]:<16}"
              f"{str(tuple(engine.get_tensor_shape(name))):<16}{engine.is_shape_inference_io(name)}")
    print("    `is_shape_inference_io` is the only reliable test; dtype and rank do not identify one.")
    result["identified"] = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors) if engine.is_shape_inference_io(engine.get_tensor_name(i))]
    return

@case_mark
def case_int32_and_int64() -> None:
    """A shape tensor may be INT32 or INT64, and the host buffer has to match."""
    for shape_dtype, numpy_dtype in [(trt.int32, np.int32), (trt.int64, np.int64)]:
        engine = build_engine(shape_dtype)
        context = engine.create_execution_context()
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        new_shape = np.array([4, 3, 2], dtype=numpy_dtype)

        context.set_input_shape("data", data.shape)
        # The shape tensor is bound by ADDRESS, and the address must point at host memory
        # holding values of the engine's dtype. A mismatched width reads neighbouring bytes.
        context.set_tensor_address("newShape", new_shape.ctypes.data)

        buffer = OrderedDict()
        for name in ["data", "output"]:
            shape = context.get_tensor_shape(name)
            n_byte = trt.volume(shape) * engine.get_tensor_dtype(name).itemsize
            buffer[name] = cudart.cudaMalloc(n_byte)[1]
            context.set_tensor_address(name, buffer[name])
        cudart.cudaMemcpy(buffer["data"], data.ctypes.data, data.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        context.execute_async_v3(0)
        cudart.cudaStreamSynchronize(0)
        output = np.empty(context.get_tensor_shape("output"), dtype=np.float32)
        cudart.cudaMemcpy(output.ctypes.data, buffer["output"], output.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        print(f"    {str(shape_dtype)[9:]:<8} shape tensor {new_shape.tolist()} -> output shape {output.shape}, "
              f"matches numpy reshape: {np.array_equal(output, data.reshape(4, 3, 2))}")
        for address in buffer.values():
            cudart.cudaFree(address)
    return

@case_mark
def case_zero_dimensional() -> None:
    """A 0-D shape tensor: a single scalar whose value steers the graph."""
    engine = build_engine(b_zero_dimensional=True)
    context = engine.create_execution_context()
    print(f"    inputs: {[(engine.get_tensor_name(i), tuple(engine.get_tensor_shape(engine.get_tensor_name(i))), engine.is_shape_inference_io(engine.get_tensor_name(i))) for i in range(engine.num_io_tensors)]}")

    for start_value in [5, 50]:
        start = np.array(start_value, dtype=np.int32)  # 0-D: note np.array(v), not np.array([v])
        context.set_tensor_address("start", start.ctypes.data)
        output_address = cudart.cudaMalloc(8 * np.dtype(np.int32).itemsize)[1]
        context.set_tensor_address("output", output_address)
        context.execute_async_v3(0)
        cudart.cudaStreamSynchronize(0)
        output = np.empty(context.get_tensor_shape("output"), dtype=np.int32)
        cudart.cudaMemcpy(output.ctypes.data, output_address, output.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        print(f"    start={start_value:<4} -> linspace {output.tolist()}")
        cudart.cudaFree(output_address)
    return

@case_mark
def case_host_not_device() -> None:
    """A shape tensor's address must be **host** memory, and getting this wrong is fatal.

    TensorRT reads a shape tensor on the host during shape inference, before any kernel runs.
    `set_tensor_address` takes an integer, so a device pointer is accepted without complaint --
    and then dereferenced on the host.

    Measured here: the process dies with **SIGSEGV**, `Invalid permissions`, at the device
    address. Not an exception, not an error code, not a wrong answer -- a core dump from
    inside `libnvinfer` with a stack that says nothing about shape tensors:

        Signal: Segmentation fault (11)
        Signal code: Invalid permissions (2)
        Failing at address: 0x78b3dd200800
        libnvinfer.so.11(+0x1e9ce76)

    So the demonstration runs in a **child process**: the mistake is real and reproducible, and
    containing it is the only way to show it without taking the example down with it.
    """
    script = textwrap.dedent('''
        import sys
        import numpy as np
        import tensorrt as trt
        import cuda.bindings.runtime as cudart
        sys.path.insert(0, {parent!r})
        from main import build_engine

        engine = build_engine()
        context = engine.create_execution_context()
        context.set_input_shape("data", (2, 3, 4))
        new_shape = np.array([4, 3, 2], dtype=np.int32)
        device_copy = cudart.cudaMalloc(new_shape.nbytes)[1]
        cudart.cudaMemcpy(device_copy, new_shape.ctypes.data, new_shape.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        context.set_tensor_address("newShape", device_copy)   # device pointer, read on the host
        print("bound without complaint", flush=True)
        print(tuple(context.get_tensor_shape("output")), flush=True)
    ''').format(parent=str(output_path))

    process = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=str(output_path))
    print(f"    child exit code: {process.returncode} ({'SIGSEGV' if process.returncode == -11 else 'see below'})")
    print(f"    child stdout   : {process.stdout.strip().splitlines()[:2]}")
    assert process.returncode != 0, "Expected the device-pointer binding to be fatal"

    # The correct binding, in this process, for contrast
    engine = build_engine()
    context = engine.create_execution_context()
    context.set_input_shape("data", (2, 3, 4))
    new_shape = np.array([4, 3, 2], dtype=np.int32)
    context.set_tensor_address("newShape", new_shape.ctypes.data)
    print(f"    host address   -> inferred output shape = {tuple(context.get_tensor_shape('output'))}")
    result["device_pointer_returncode"] = process.returncode
    return

@case_mark
def case_wrong_api() -> None:
    """The two nearly-identical APIs, and what each says when used on the wrong kind."""
    engine = build_engine()
    context = engine.create_execution_context()

    # 1) `set_input_shape` on a shape tensor: names a shape where a value was wanted
    try:
        context.set_input_shape("newShape", (3, ))
        print("    set_input_shape('newShape', (3,)) was accepted (it names the shape, not the values)")
    except Exception as exception:  # noqa: BLE001
        print(f"    set_input_shape on a shape tensor -> {str(exception).splitlines()[0]}")

    # 2) `set_shape` in the profile for a shape tensor
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    profile = builder.create_optimization_profile()
    try:
        profile.set_shape("newShape", [1], [3], [3])
        print("    profile.set_shape on a shape-tensor name: accepted at profile level (the profile has no engine to check against)")
    except Exception as exception:  # noqa: BLE001
        print(f"    profile.set_shape on a shape tensor -> {str(exception).splitlines()[0]}")

    print("    Neither call fails loudly at the point of the mistake; the failure surfaces later,")
    print("    as an unspecified binding or a wrong output shape.")
    return

@case_mark
def case_shape_inference() -> None:
    """`infer_shapes` names what is still missing, which is the cheapest way to debug this."""
    engine = build_engine()
    context = engine.create_execution_context()

    print(f"    nothing set yet          : infer_shapes -> {context.infer_shapes()}")
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    context.set_input_shape("data", data.shape)
    print(f"    execution tensor set     : infer_shapes -> {context.infer_shapes()}")
    new_shape = np.array([4, 3, 2], dtype=np.int32)
    context.set_tensor_address("newShape", new_shape.ctypes.data)
    print(f"    shape tensor set as well : infer_shapes -> {context.infer_shapes()} (empty list = everything resolved)")
    print(f"    resulting output shape   : {tuple(context.get_tensor_shape('output'))}")
    assert context.infer_shapes() == [], "Expected all bindings resolved"
    return

def main() -> None:
    case_identify()
    case_int32_and_int64()
    case_zero_dimensional()
    case_host_not_device()
    case_wrong_api()
    case_shape_inference()
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
