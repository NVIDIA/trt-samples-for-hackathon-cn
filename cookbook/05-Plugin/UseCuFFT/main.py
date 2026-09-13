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
"""Call cuFFT from a TensorRT plugin, and give the ONNX `DFT` operator somewhere to go.

TensorRT has no FFT layer, and its ONNX parser rejects the `DFT` operator outright:

    In node 0 with name:  and operator: DFT (checkDFT): UNSUPPORTED_NODE

So a spectral model stops at the parser. This example closes that gap the same way
`UseCuBLAS` closes the GEMM one: a plugin that forwards to a CUDA math library, here
cuFFT, covering all three transform families.

| Mode | input | output | reference |
| ---- | ----- | ------ | --------- |
| C2C forward | `[.., n, 2]` | `[.., n, 2]` | `numpy.fft.fft` |
| C2C inverse | `[.., n, 2]` | `[.., n, 2]` | `numpy.fft.ifft` (times n, see below) |
| R2C | `[.., n]` | `[.., n // 2 + 1, 2]` | `numpy.fft.rfft` |
| C2R | `[.., n // 2 + 1, 2]` | `[.., n]` | `numpy.fft.irfft` (times n, see below) |

ONNX has no complex type, so a complex tensor is a real tensor with a trailing dimension
of 2 — the convention the ONNX `DFT` operator itself uses.

**cuFFT does not normalise.** A forward transform followed by an inverse one multiplies
the signal by `n`; cuFFT's own documentation says so and leaves the scaling to the caller.
NumPy puts the `1/n` in the inverse, so `cufftExecC2C(..., CUFFT_INVERSE)` equals
`numpy.fft.ifft(...) * n` and `cufftExecC2R` equals `numpy.fft.irfft(...) * n`. Comparing
against NumPy without that factor is the first thing that goes wrong here, and it goes
wrong by an exact integer factor, which is a useful thing to recognise.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnx
import onnx.helper as oh
import tensorrt as trt

from tensorrt_cookbook import TRTWrapperV1, case_mark, check_array, load_plugin_files

np.random.seed(31193)

N_BATCH = 4
N_SIGNAL = 32  # cuFFT is happiest on lengths with small prime factors
# Must be an absolute path: `IPluginRegistry.load_library` hands the string to `dlopen`,
# which searches the system library paths for a bare filename, never the working directory,
# and then returns `None` instead of raising -- the failure only shows up later as
# "Cannot find plugin: CuFFT".
PLUGIN_FILE_LIST = [Path(__file__).parent / "CuFFTPlugin.so"]
onnx_file_dft = Path(__file__).parent / "model-onnx_dft.onnx"
onnx_file_plugin = Path(__file__).parent / "model-onnx_plugin.onnx"

MODE_C2C, MODE_R2C, MODE_C2R = 0, 1, 2

result = OrderedDict()

def complex_to_real(array: np.ndarray) -> np.ndarray:
    """`[..., n]` complex -> `[..., n, 2]` float32, the ONNX convention."""
    return np.ascontiguousarray(np.stack([array.real, array.imag], axis=-1).astype(np.float32))

def make_plugin(mode: int, signal_length: int, inverse: int = 0):
    """Instantiate the plugin through its creator, exactly as `UseCuBLAS` does."""
    plugin_creator = trt.get_plugin_registry().get_creator("CuFFT", "1", "")
    assert plugin_creator is not None, "Failed loading the CuFFT plugin, run `make build` first"
    field_list = [
        trt.PluginField("mode", np.int32(mode), trt.PluginFieldType.INT32),
        trt.PluginField("signal_length", np.int32(signal_length), trt.PluginFieldType.INT32),
        trt.PluginField("inverse", np.int32(inverse), trt.PluginFieldType.INT32),
    ]
    return plugin_creator.create_plugin("CuFFT", trt.PluginFieldCollection(field_list), trt.TensorRTPhase.BUILD)

def run_plugin(input_data: dict, input_shape: list, mode: int, inverse: int = 0) -> np.ndarray:
    """Build a one-layer engine around the plugin and run it once."""
    tw = TRTWrapperV1()  # The plugin library is loaded once in `main()`
    input_tensor = tw.network.add_input("inputT0", trt.float32, [-1] + input_shape[1:])
    tw.profile.set_shape(input_tensor.name, [1] + input_shape[1:], input_shape, input_shape)
    layer = tw.network.add_plugin_v3([input_tensor], [], make_plugin(mode, N_SIGNAL, inverse))
    tensor = layer.get_output(0)
    tensor.name = "outputT0"
    tw.build([tensor])
    tw.setup(input_data, b_print_io=False)
    tw.infer(b_print_io=False)
    return tw.buffer["outputT0"][0]

# ================================================================ Cases

@case_mark
def case_c2c_forward() -> None:
    """Complex-to-complex forward transform, against `numpy.fft.fft`."""
    signal = (np.random.rand(N_BATCH, N_SIGNAL) * 2 - 1) + 1j * (np.random.rand(N_BATCH, N_SIGNAL) * 2 - 1)
    input_array = complex_to_real(signal)
    output = run_plugin({"inputT0": input_array}, list(input_array.shape), MODE_C2C, inverse=0)

    reference = complex_to_real(np.fft.fft(signal, axis=-1))
    print(f"    input {input_array.shape} -> output {output.shape}")
    check_array(output, reference, True)
    result["C2C forward"] = float(np.max(np.abs(output - reference)))
    return

@case_mark
def case_c2c_inverse() -> None:
    """Complex-to-complex inverse transform. Note the missing 1/n."""
    spectrum = (np.random.rand(N_BATCH, N_SIGNAL) * 2 - 1) + 1j * (np.random.rand(N_BATCH, N_SIGNAL) * 2 - 1)
    input_array = complex_to_real(spectrum)
    output = run_plugin({"inputT0": input_array}, list(input_array.shape), MODE_C2C, inverse=1)

    unnormalised = complex_to_real(np.fft.ifft(spectrum, axis=-1) * N_SIGNAL)
    normalised = complex_to_real(np.fft.ifft(spectrum, axis=-1))
    print(f"    vs numpy.fft.ifft * n : max |diff| = {np.max(np.abs(output - unnormalised)):.3e}  <- cuFFT's convention")
    print(f"    vs numpy.fft.ifft     : max |diff| = {np.max(np.abs(output - normalised)):.3e}  <- off by exactly n = {N_SIGNAL}")
    check_array(output, unnormalised, True)
    result["C2C inverse"] = float(np.max(np.abs(output - unnormalised)))
    return

@case_mark
def case_r2c() -> None:
    """Real-to-complex, the transform an audio or radar front end actually wants."""
    signal = (np.random.rand(N_BATCH, N_SIGNAL) * 2 - 1).astype(np.float32)
    output = run_plugin({"inputT0": signal}, list(signal.shape), MODE_R2C)

    reference = complex_to_real(np.fft.rfft(signal, axis=-1))
    print(f"    input {signal.shape} -> output {output.shape} (n // 2 + 1 = {N_SIGNAL // 2 + 1} complex bins)")
    check_array(output, reference, True)
    result["R2C"] = float(np.max(np.abs(output - reference)))
    return

@case_mark
def case_c2r() -> None:
    """Complex-to-real, closing the loop back to a waveform."""
    signal = (np.random.rand(N_BATCH, N_SIGNAL) * 2 - 1).astype(np.float32)
    spectrum = np.fft.rfft(signal, axis=-1)
    input_array = complex_to_real(spectrum)
    output = run_plugin({"inputT0": input_array}, list(input_array.shape), MODE_C2R)

    reference = (np.fft.irfft(spectrum, n=N_SIGNAL, axis=-1) * N_SIGNAL).astype(np.float32)
    print(f"    input {input_array.shape} -> output {output.shape}")
    print(f"    round trip rfft -> irfft recovers the signal: max |diff| = {np.max(np.abs(output / N_SIGNAL - signal)):.3e}")
    check_array(output, reference, True)
    result["C2R"] = float(np.max(np.abs(output - reference)))
    return

@case_mark
def case_onnx_dft() -> None:
    """The parser gap, and how the plugin fills it.

    First build an ONNX file that uses the standard `DFT` operator and watch TensorRT
    refuse it. Then rewrite that node into a plugin node in the `trt.plugins` domain,
    which the parser resolves against the registry by op type, and watch it parse.
    """
    n_complex = N_SIGNAL // 2 + 1

    # ---- The standard operator
    graph = oh.make_graph(
        [oh.make_node("DFT", ["x", "axis"], ["y"], onesided=1, inverse=0)],
        "dft",
        [oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [N_BATCH, N_SIGNAL, 1])],
        [oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [N_BATCH, n_complex, 2])],
        [oh.make_tensor("axis", onnx.TensorProto.INT64, [], [1])],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 20)])
    model.ir_version = 10
    onnx.save(model, onnx_file_dft)

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    parsed_dft = parser.parse_from_file(str(onnx_file_dft))
    print(f"    standard ONNX `DFT` parsed by TensorRT: {parsed_dft}")
    if not parsed_dft:
        print(f"        {parser.get_error(parser.num_errors - 1)}")

    # ---- The same graph, with the node pointed at this plugin
    node = oh.make_node(
        "CuFFT",
        ["x"],
        ["y"],
        domain="trt.plugins",  # The domain the parser resolves against the plugin registry
        mode=MODE_R2C,
        signal_length=N_SIGNAL,
        inverse=0,
        plugin_version="1",
        plugin_namespace="",
    )
    graph = oh.make_graph(
        [node],
        "dft_plugin",
        [oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [N_BATCH, N_SIGNAL])],
        [oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [N_BATCH, n_complex, 2])],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 20), oh.make_opsetid("trt.plugins", 1)])
    model.ir_version = 10
    onnx.save(model, onnx_file_plugin)

    tw = TRTWrapperV1()
    parser = trt.OnnxParser(tw.network, tw.logger)
    parsed_plugin = parser.parse_from_file(str(onnx_file_plugin))
    print(f"    the same graph as a `trt.plugins` node  : {parsed_plugin}")
    if not parsed_plugin:
        for i in range(parser.num_errors):
            print(f"        {parser.get_error(i)}")
        assert False, "Failed parsing the plugin form"

    signal = (np.random.rand(N_BATCH, N_SIGNAL) * 2 - 1).astype(np.float32)
    tw.build()
    tw.setup({"x": signal}, b_print_io=False)
    tw.infer(b_print_io=False)
    output = tw.buffer["y"][0]
    reference = complex_to_real(np.fft.rfft(signal, axis=-1))
    print(f"    parsed-from-ONNX result matches numpy.fft.rfft: max |diff| = {np.max(np.abs(output - reference)):.3e}")
    check_array(output, reference, True)
    result["ONNX -> plugin"] = float(np.max(np.abs(output - reference)))
    return

# ================================================================ Entrance

def main() -> None:
    # Load the library exactly once. Calling `load_library` again for the same file fails with
    # "Cannot register the library as plugin creator of CuFFT exists already", so a per-engine
    # `plugin_file_list=` would work but would print that error for every engine built here.
    load_plugin_files(PLUGIN_FILE_LIST)

    case_c2c_forward()
    case_c2c_inverse()
    case_r2c()
    case_c2r()
    case_onnx_dft()

    print("\n" + "=" * 56)
    print(f"{'Transform':<24}{'max |diff| vs NumPy':>28}")
    print("-" * 56)
    for name, value in result.items():
        print(f"{name:<24}{value:>28.3e}")
    print("=" * 56)
    return

if __name__ == "__main__":
    main()
    print("\nFinish")
