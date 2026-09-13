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
"""Feeding a `TrtRunner` PyTorch tensors instead of NumPy arrays.

`TrtRunner` accepts either. What comes back follows what went in:

    numpy in                                  -> numpy out
    torch in, copy_outputs_to_host=True       -> torch tensor on CPU  (default)
    torch in, copy_outputs_to_host=False      -> torch tensor on GPU

The reason to care is dtype coverage. `torch.bfloat16` always exists; NumPy has
no native bfloat16, so a BF16 network is awkward to drive from NumPy and trivial
to drive from torch. `case_bfloat16_end_to_end` builds one and runs it.

Note the upstream example for this feature pairs it with an INT8 `Calibrator`,
which does not work on TensorRT 11 at all -- see `../06-Int8IsNowExplicit/`.
Only the tensor half of that example still applies.
"""

import numpy as np
import tensorrt as trt
import torch
from polygraphy import func
from polygraphy.backend.trt import CreateConfig, CreateNetwork, EngineFromNetwork, NetworkFromOnnxPath, TrtRunner
from polygraphy.logger import G_LOGGER

from tensorrt_cookbook import case_mark, cookbook_path

G_LOGGER.module_severity = G_LOGGER.ERROR

onnx_file = str(cookbook_path("00-Data", "model", "model-trained.onnx"))
SHAPE = (64, 64)

engine = None

@case_mark
def case_either_array_type_works() -> None:
    """The same engine, driven from NumPy and from torch.

    The output type mirrors the input type, so an inference pipeline written
    around torch tensors does not have to convert on the way in or out.
    """
    global engine
    engine = EngineFromNetwork(NetworkFromOnnxPath(onnx_file), config=CreateConfig())()

    numpy_input = np.load(cookbook_path("00-Data", "data", "InferenceData.npy"))
    torch_input = torch.from_numpy(numpy_input).cuda()

    # A separate runner per array type -- `case_the_first_feed_dict_decides`
    # explains why that matters.
    with TrtRunner(engine) as runner:
        from_numpy = runner.infer({"x": numpy_input})["y"]
    with TrtRunner(engine) as runner:
        from_torch = runner.infer({"x": torch_input})["y"]
        on_device = runner.infer({"x": torch_input}, copy_outputs_to_host=False)["y"]

    print(f"    numpy in -> {type(from_numpy).__name__}")
    print(f"    torch in -> {type(from_torch).__name__} on {from_torch.device}  (copy_outputs_to_host=True, the default)")
    print(f"    torch in -> {type(on_device).__name__} on {on_device.device}  (copy_outputs_to_host=False)")
    print(f"    numpy and torch paths agree: {np.array_equal(np.asarray(from_numpy), from_torch.cpu().numpy())}")
    return

@case_mark
def case_the_first_feed_dict_decides() -> None:
    """The trap: the output container is chosen once, by the first `infer()`.

    `copy_outputs_to_host=False` is documented to return a torch tensor when
    PyTorch has GPU support and a Polygraphy `DeviceView` otherwise. What is not
    documented is that the choice is made from the **first** feed dict the runner
    ever saw, not the current one.

    So a runner that was handed NumPy once keeps returning `DeviceView` even when
    later fed torch tensors. `DeviceView` has no `.device` and no `.cpu()`, so the
    failure surfaces as an `AttributeError` somewhere downstream rather than at
    the call that caused it.

    One array type per runner is the rule this implies.
    """
    numpy_input = np.load(cookbook_path("00-Data", "data", "InferenceData.npy"))
    torch_input = torch.from_numpy(numpy_input).cuda()

    with TrtRunner(engine) as runner:
        first = runner.infer({"x": torch_input}, copy_outputs_to_host=False)["y"]
    print(f"    fresh runner, torch first            -> {type(first).__name__}")

    with TrtRunner(engine) as runner:
        runner.infer({"x": numpy_input})
        later = runner.infer({"x": torch_input}, copy_outputs_to_host=False)["y"]
    print(f"    fresh runner, numpy first then torch -> {type(later).__name__}   <- same call, different type")
    print(f"    DeviceView has `.device`? {hasattr(later, 'device')}  -- downstream torch code breaks here")
    return

@case_mark
def case_staying_on_the_device() -> None:
    """`copy_outputs_to_host=False` keeps the result where the engine wrote it.

    Useful when the next step is also on the GPU: the output never crosses to
    host memory. The same caveat as everywhere else applies -- the runner owns
    that buffer and will overwrite it on the next `infer()`
    (`../04-ExtendInterop/`).
    """
    torch_input = torch.from_numpy(np.load(cookbook_path("00-Data", "data", "InferenceData.npy"))).cuda()
    with TrtRunner(engine) as runner:  # torch first, so the outputs stay torch
        device_output = runner.infer({"x": torch_input}, copy_outputs_to_host=False)["y"]
        print(f"    output stays on {device_output.device}, dtype {device_output.dtype}")
        # A GPU-side operation, no host round trip
        print(f"    argmax computed on the GPU: {torch.argmax(device_output, dim=1).item()}")
    return

@case_mark
def case_bfloat16_end_to_end() -> None:
    """The dtype argument for using torch: BF16 without leaving the type system.

    The network declares `bfloat16` (see `../05-BuildNetworkByHand/` -- precision
    is declared, not requested), and the runner takes and returns
    `torch.bfloat16` tensors directly.
    """

    @func.extend(CreateNetwork())
    def make(builder, network):
        """`y = x + 1`, entirely in BF16."""
        tensor = network.add_input(name="x", shape=SHAPE, dtype=trt.bfloat16)
        one = network.add_constant(shape=SHAPE, weights=np.ones(SHAPE, dtype=np.float32)).get_output(0)
        one_bf16 = network.add_cast(one, trt.bfloat16).get_output(0)
        output = network.add_elementwise(tensor, one_bf16, op=trt.ElementWiseOperation.SUM).get_output(0)
        output.name = "y"
        network.mark_output(output)

    with TrtRunner(EngineFromNetwork(make)) as runner:
        data = torch.rand(*SHAPE, dtype=torch.bfloat16, device="cuda")
        output = runner.infer({"x": data})["y"]
        print(f"    in {data.dtype} -> out {output.dtype}, exact match = {torch.equal(output.cuda(), data + 1)}")
    return

@case_mark
def case_what_numpy_cannot_do() -> None:
    """Why that matters: NumPy has no native bfloat16.

    Careful with this one -- `np.dtype("bfloat16")` may or may not resolve
    depending on what else has been imported. It is not a NumPy feature; the
    `ml_dtypes` package registers it, and `ml_dtypes` arrives transitively in
    some environments. Code that relies on it works by accident.

    `torch.bfloat16` needs no registration and is always there, which is the
    robust reason to hand the runner torch tensors for BF16 work.
    """
    print(f"    numpy version: {np.__version__}")
    try:
        dtype = np.dtype("bfloat16")
        print(f"    np.dtype('bfloat16') resolves to {dtype} -- registered by `ml_dtypes`, not by NumPy")
    except TypeError as e:
        print(f"    np.dtype('bfloat16') -> TypeError: {str(e)[:60]}")
    try:
        import ml_dtypes
        print(f"    ml_dtypes {ml_dtypes.__version__} is installed here, which is why it may resolve")
    except ImportError:
        print("    ml_dtypes is not installed")
    print(f"    torch.bfloat16 is always available: {torch.bfloat16}")
    return

if __name__ == "__main__":
    case_either_array_type_works()
    case_the_first_feed_dict_decides()
    case_staying_on_the_device()
    case_bfloat16_end_to_end()
    case_what_numpy_cannot_do()

    print("\nFinish")
