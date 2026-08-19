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
"""The Tripy half of the example -- **this file runs inside the private virtual environment**.

It cannot be imported by the cookbook's interpreter: `nvtripy` drags in its own `tensorrt-cu12`
and would shadow the TensorRT the rest of the cookbook uses. `main.py` creates the venv and runs
this file in it; see `README.md`.

Written against nvtripy 0.1.7. The project is pre-1.0 and the API is still moving, so every case
prints what it observed rather than relying on a remembered signature.
"""

import tempfile
import time
from pathlib import Path

import nvtripy as tp

def case_mark(f):
    """A local copy of the cookbook decorator -- `tensorrt_cookbook` is not importable in here."""

    def f_with_mark(*args, **kargs):
        print(f"\n{'=' * 30} Start [{f.__name__}]")
        result = f(*args, **kargs)
        print(f"{'=' * 30} End   [{f.__name__}]")
        return result

    return f_with_mark

class MLP(tp.Module):
    """A `tp.Module` is a PyTorch-shaped container: submodules as attributes, `load_state_dict`."""

    def __init__(self, in_dimension, hidden_dimension, out_dimension):
        self.layer_0 = tp.Linear(in_dimension, hidden_dimension)
        self.layer_1 = tp.Linear(hidden_dimension, out_dimension)

    def forward(self, x):
        return self.layer_1(tp.gelu(self.layer_0(x)))

def build_mlp(in_dimension=4, hidden_dimension=8, out_dimension=2):
    model = MLP(in_dimension, hidden_dimension, out_dimension)
    model.load_state_dict({
        "layer_0.weight": tp.ones((hidden_dimension, in_dimension)) * 0.1,
        "layer_0.bias": tp.zeros((hidden_dimension, )),
        "layer_1.weight": tp.ones((out_dimension, hidden_dimension)) * 0.2,
        "layer_1.bias": tp.zeros((out_dimension, )),
    })
    return model

@case_mark
def case_eager_then_compile():
    """The whole value proposition in one case: run it eagerly, then compile the same object.

    Nothing about the model changes between the two runs -- `tp.compile` takes the module and a
    description of which arguments are runtime inputs, and hands back an `Executable`.
    """
    model = build_mlp()
    x = tp.ones((3, 4), dtype=tp.float32)

    eager_output = model(x).tolist()  # Runs immediately, values available now
    print(f"    eager    : {eager_output[0]}")

    x.eval()  # An Executable takes evaluated tensors, see case_lazy_evaluation
    executable = tp.compile(model, args=[tp.InputInfo(shape=(3, 4), dtype=tp.float32)])
    compiled_output = executable(x).tolist()
    print(f"    compiled : {compiled_output[0]}")

    difference = max(abs(a - b) for row_a, row_b in zip(eager_output, compiled_output) for a, b in zip(row_a, row_b))
    print(f"    max |eager - compiled| = {difference:.3e}  ({difference / abs(eager_output[0][0]):.1e} relative)")
    assert difference < 1e-3, "eager and compiled disagree by more than a precision difference"

    # They are close but **not identical**, which is worth knowing before debugging numerics in
    # eager mode. Localizing it: elementwise ops agree exactly, the matmul inside `tp.Linear` does
    # not -- and it is *eager* that drifts from the exact answer.
    x_small = tp.ones((3, 4), dtype=tp.float32) * 0.7
    elementwise_eager = (tp.gelu(x_small) * 2.0).tolist()
    x_small.eval()
    elementwise_compiled = tp.compile(lambda t: tp.gelu(t) * 2.0, args=[tp.InputInfo(shape=(3, 4), dtype=tp.float32)])(x_small).tolist()
    print(f"    elementwise only, eager == compiled: {elementwise_eager == elementwise_compiled}")

    linear = tp.Linear(4, 2)
    linear.load_state_dict({"weight": tp.ones((2, 4)) * 0.1, "bias": tp.zeros((2, ))})
    x_linear = tp.ones((3, 4), dtype=tp.float32) * 0.7
    linear_eager = linear(x_linear).tolist()[0][0]
    x_linear.eval()
    linear_compiled = tp.compile(linear, args=[tp.InputInfo(shape=(3, 4), dtype=tp.float32)])(x_linear).tolist()[0][0]
    print(f"    one Linear, exact answer 0.28: eager {linear_eager!r}, compiled {linear_compiled!r}")
    print("    -> the matmul is the difference, and eager is the one that drifts")

    # Everything that is not declared an input becomes a compile-time constant. The weights are
    # folded into the engine, which is why the Executable does not take them as arguments.
    print(f"    Executable runtime inputs: {list(executable.input_infos)} (the weights were folded in)")
    return

@case_mark
def case_lazy_evaluation():
    """Tripy is lazy, and that breaks the obvious way of timing it.

    A tensor is only computed when something needs its value. Timing the *definition* measures
    graph construction; the first use pays for compilation as well as execution.
    """
    start_time = time.time()
    a = tp.gelu(tp.ones((2, 8)))
    define_time = (time.time() - start_time) * 1000

    start_time = time.time()
    a.eval()  # First use: this is where the work happens
    first_use_time = (time.time() - start_time) * 1000

    start_time = time.time()
    a.eval()
    second_use_time = (time.time() - start_time) * 1000

    print(f"    defining the tensor : {define_time:8.3f} ms")
    print(f"    first .eval()       : {first_use_time:8.3f} ms  <- compiles, then executes")
    print(f"    second .eval()      : {second_use_time:8.3f} ms  <- already evaluated, cached")
    assert first_use_time > define_time, "the definition was not the cheap half"
    print(f"    -> timing the definition understates the cost by {first_use_time / max(define_time, 1e-9):.0f}x")

    # The same laziness is why an Executable rejects an unevaluated input rather than evaluating
    # it: the error names the tensor and tells you to call `.eval()`.
    executable = tp.compile(lambda t: t + 1, args=[tp.InputInfo(shape=(2, 2), dtype=tp.float32)])
    try:
        executable(tp.ones((2, 2)))
        print("    (an unevaluated input was accepted -- nvtripy behaviour changed)")
    except Exception as e:
        print(f"    passing an unevaluated tensor to an Executable: {type(e).__name__}: {str(e).strip().splitlines()[-1].strip()}")
    return

@case_mark
def case_dynamic_shapes():
    """`InputInfo` takes a `(min, opt, max)` tuple per dimension, exactly like a TensorRT profile."""
    model = build_mlp()
    # Dimension 0 ranges over 1..8 and is tuned for 4; dimension 1 is fixed at 4
    input_info = tp.InputInfo(shape=((1, 4, 8), 4), dtype=tp.float32)
    executable = tp.compile(model, args=[input_info])

    for batch_size in [1, 4, 8]:
        x = tp.ones((batch_size, 4), dtype=tp.float32).eval()
        output = executable(x)
        print(f"    batch {batch_size}: output shape {output.shape}")

    x = tp.ones((9, 4), dtype=tp.float32).eval()
    try:
        executable(x)
        print("    batch 9 was accepted -- the range is not enforced?")
    except Exception as e:
        print(f"    batch 9 (outside 1..8): {type(e).__name__}")
    print("    -> one Executable serves the declared range; outside it you get an error, not a rebuild")
    return

@case_mark
def case_save_and_load_executable():
    """An `Executable` is the deployable artifact: save it, load it, skip the compile."""
    model = build_mlp()
    x = tp.ones((3, 4), dtype=tp.float32).eval()

    start_time = time.time()
    executable = tp.compile(model, args=[tp.InputInfo(shape=(3, 4), dtype=tp.float32)])
    compile_time = time.time() - start_time
    reference = executable(x).tolist()

    with tempfile.TemporaryDirectory() as temp_dir:
        executable_file = Path(temp_dir) / "executable.json"
        executable.save(str(executable_file))
        size_in_kib = executable_file.stat().st_size / 1024

        start_time = time.time()
        loaded_executable = tp.Executable.load(str(executable_file))
        load_time = time.time() - start_time
        loaded_output = loaded_executable(x).tolist()

    assert loaded_output == reference, "the loaded Executable does not match the compiled one"
    print(f"    compile : {compile_time * 1000:8.1f} ms")
    print(f"    load    : {load_time * 1000:8.1f} ms   ({size_in_kib:.0f} KiB on disk)")
    print(f"    -> loading is {compile_time / max(load_time, 1e-9):.0f}x cheaper than compiling, and the output is identical")
    return

if __name__ == "__main__":
    print(f"nvtripy {tp.__version__}")

    case_eager_then_compile()
    case_lazy_evaluation()
    case_dynamic_shapes()
    case_save_and_load_executable()

    print("\nFinish")
