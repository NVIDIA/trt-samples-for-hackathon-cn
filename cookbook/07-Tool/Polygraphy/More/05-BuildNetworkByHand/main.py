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
"""Building a TensorRT network by hand, then handing it back to Polygraphy.

`CreateNetwork()` produces an empty network; `@func.extend` (see
`../04-ExtendInterop/`) lets a function fill it using raw TensorRT APIs. After
that the ordinary Polygraphy loaders and runners take over.

The interesting part on TensorRT 11 is precision. `CreateNetwork()` returns a
**strongly typed** network, so `dtype` on `add_input` and on the weights is not a
hint -- it is the declaration, and it is the only way to get FP16 now that the
builder flags are gone (`../02-ComparingBackends/`, `../04-ExtendInterop/`).

`02-API/` builds networks the same way without Polygraphy; the difference is only
who owns the builder and the runner.
"""

import numpy as np
import tensorrt as trt
from polygraphy import func
from polygraphy.backend.trt import CreateNetwork, EngineFromNetwork, TrtRunner
from polygraphy.logger import G_LOGGER

from tensorrt_cookbook import case_mark

G_LOGGER.module_severity = G_LOGGER.ERROR

SHAPE = (64, 64)

def build_add_one(trt_dtype, numpy_dtype):
    """A network computing `y = x + 1`, entirely in the requested type."""

    @func.extend(CreateNetwork())
    def make(builder, network):
        """Fills the empty network with raw TensorRT calls."""
        tensor = network.add_input(name="x", shape=SHAPE, dtype=trt_dtype)
        one = network.add_constant(shape=SHAPE, weights=np.ones(SHAPE, dtype=numpy_dtype)).get_output(0)
        output = network.add_elementwise(tensor, one, op=trt.ElementWiseOperation.SUM).get_output(0)
        output.name = "y"
        network.mark_output(output)

    return make

@case_mark
def case_empty_network_is_strongly_typed() -> None:
    """What `CreateNetwork()` hands back before anything is added.

    Strongly typed is the default and there is no flag to turn it off here, so
    every tensor's `dtype` is a commitment rather than a preference.
    """
    builder, network = CreateNetwork()()
    strongly_typed = network.get_flag(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    print(f"    CreateNetwork() -> {network.num_layers} layers, STRONGLY_TYPED = {strongly_typed}")
    print(f"    inputs {network.num_inputs}, outputs {network.num_outputs}")
    return

@case_mark
def case_fill_it_and_run() -> None:
    """The whole loop: build by hand, run through Polygraphy, check the maths."""
    with TrtRunner(EngineFromNetwork(build_add_one(trt.float32, np.float32))) as runner:
        data = np.random.random_sample(SHAPE).astype(np.float32)
        output = np.asarray(runner.infer({"x": data})["y"])
    print(f"    y = x + 1 in FP32: out dtype {output.dtype}, exact match = {np.array_equal(output, data + 1)}")
    return

@case_mark
def case_precision_is_declared_not_requested() -> None:
    """FP16 comes from the `dtype` arguments, not from a builder flag.

    This is the constructive half of the strong-typing story the other examples
    only show the negative side of: `CreateConfig(fp16=True)` raises and
    `trt.BuilderFlag.FP16` no longer exists, because the answer moved here.
    """
    for trt_dtype, numpy_dtype, tag in [(trt.float32, np.float32, "float32"), (trt.float16, np.float16, "float16")]:
        with TrtRunner(EngineFromNetwork(build_add_one(trt_dtype, numpy_dtype))) as runner:
            data = np.random.random_sample(SHAPE).astype(numpy_dtype)
            output = np.asarray(runner.infer({"x": data})["y"])
        print(f"    declared {tag:<8} -> engine output dtype {output.dtype}, exact match = {np.array_equal(output, data + 1)}")
    print("    no config was touched: the network said what it wanted and got it")
    return

@case_mark
def case_mismatched_types_fail_the_build() -> None:
    """Strong typing means a type error is a build error, not a silent cast.

    Declaring the input `bfloat16` while handing the constant `float16` weights
    is exactly the kind of mistake a weakly typed builder used to paper over by
    inserting a cast. Here the engine simply does not build.
    """
    try:
        with TrtRunner(EngineFromNetwork(build_add_one(trt.bfloat16, np.float16))) as runner:
            runner.infer({"x": np.zeros(SHAPE, dtype=np.float16)})
        print("    unexpectedly built")
    except Exception as e:
        print(f"    bfloat16 input + float16 weights: {type(e).__name__}: {str(e).splitlines()[0][:74]}")
    print("    the error arrives at build time, which is where a type error belongs")
    return

if __name__ == "__main__":
    case_empty_network_is_strongly_typed()
    case_fill_it_and_run()
    case_precision_is_declared_not_requested()
    case_mismatched_types_fail_the_build()

    print("\nFinish")
