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
"""Change a network you did not build, at the last moment before it is compiled.

`build_mnist_network_trt` is somebody else's builder as far as this example is concerned: it is
imported, not edited. Every case here changes what it produces without touching a line of it.
"""

import numpy as np
import tensorrt as trt
from graph_transform import BuilderFactory, graph_transform, is_active, wrap_builder

from tensorrt_cookbook import TRTWrapperV1, build_mnist_network_trt, case_mark

shape = [1, 1, 28, 28]
data = {"x": np.random.rand(np.prod(shape)).astype(np.float32).reshape(shape) * 2 - 1}

def build_once(tw: TRTWrapperV1) -> list:
    """Build the stock MNIST network and return the engine's tensor names."""
    output_tensor_list = build_mnist_network_trt(tw)
    tw.builder = wrap_builder(tw.builder)  # No-op unless a transform is scoped
    assert tw.build(output_tensor_list), "Build failed"

    runtime = trt.Runtime(tw.logger)
    engine = runtime.deserialize_cuda_engine(tw.engine_bytes)
    return [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]

@case_mark
def case_expose_an_internal_tensor():
    """Mark a fused-away intermediate as an engine output, from outside the builder.

    This is the use that justifies the machinery. A numerical bug is somewhere in the middle of a
    network, and you want the tensor after `Activation1` on the output bindings. The builder does
    not offer that, and editing it means editing code shared with everyone else.
    """
    baseline = build_once(TRTWrapperV1())
    print(f"    Without a transform : {baseline}")

    def expose(network: trt.INetworkDefinition, engine_index: int) -> None:
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.name == "Activation1":
                tensor = layer.get_output(0)
                network.mark_output(tensor)
                # No dtype to set: the network is strongly typed, so `ITensor.dtype` is read-only
                # (`AttributeError: property of 'ITensor' object has no setter` on TensorRT 11.3)
                # and the marked output keeps whatever type the graph gives it.
                print(f"    transform(engine_index={engine_index}) marked {tensor.name!r} as an output, dtype {tensor.dtype}")
                return
        raise AssertionError("Activation1 not found; the upstream builder changed")

    with graph_transform(expose):
        patched = build_once(TRTWrapperV1())
    print(f"    With    a transform : {patched}")

    new_tensor_list = [name for name in patched if name not in baseline]
    assert len(new_tensor_list) == 1, f"Expected exactly one new binding, got {new_tensor_list}"
    print(f"    Gained {len(patched) - len(baseline)} binding: {new_tensor_list[0]}")

@case_mark
def case_engine_index_separates_builds():
    """One scope, several builds: the transform is told which engine it is looking at.

    A pipeline that builds a prefill engine and a decode engine from one request needs to treat
    them differently. The counter is per scope, so the numbering restarts for the next request.
    """
    seen = []

    def record(network: trt.INetworkDefinition, engine_index: int) -> None:
        seen.append((engine_index, network.num_layers))

    with graph_transform(record):
        build_once(TRTWrapperV1())
        build_once(TRTWrapperV1())
    print(f"    First scope  : {seen}")
    assert [index for index, _ in seen] == [0, 1], seen

    seen.clear()
    with graph_transform(record):
        build_once(TRTWrapperV1())
    print(f"    Second scope : {seen}")
    assert [index for index, _ in seen] == [0], "The counter must restart per scope"

@case_mark
def case_scope_is_the_whole_contract():
    """Outside the `with` block nothing is wrapped, and an exception still unwinds the scope."""
    assert not is_active()

    def explode(network: trt.INetworkDefinition, engine_index: int) -> None:
        raise RuntimeError("transform failed")

    try:
        with graph_transform(explode):
            assert is_active()
            build_once(TRTWrapperV1())
    except RuntimeError as e:
        print(f"    Transform raised: {e}")
    assert not is_active(), "The scope must be reset even when the transform raises"
    print("    Scope reset after the exception")

    with graph_transform(None):  # Supported no-op, so callers need no branch
        assert not is_active()
    print("    graph_transform(None) is a no-op, as documented")

@case_mark
def case_patch_the_builder_class():
    """When the builder is created out of reach, wrap `trt.Builder` instead of the instance.

    `TRTWrapperV1.__init__` calls `trt.Builder(...)` itself, so there is no object to hand to
    `wrap_builder` before it is used. Replacing the class covers that case -- and covers code that
    builds engines several layers down a call stack you do not control.
    """
    original_builder = trt.Builder
    trt.Builder = BuilderFactory(original_builder)
    try:
        calls = []

        def count_layers(network: trt.INetworkDefinition, engine_index: int) -> None:
            calls.append(network.num_layers)

        with graph_transform(count_layers):
            tw = TRTWrapperV1()  # Builder created in here, already wrapped
            assert isinstance(tw.builder, type(trt.Builder(tw.logger))), "Expected a wrapped builder"
            output_tensor_list = build_mnist_network_trt(tw)
            assert tw.build(output_tensor_list), "Build failed"
        print(f"    Transform fired {len(calls)} time(s) through the patched class, network had {calls[0]} layers")
        assert len(calls) == 1, calls
    finally:
        trt.Builder = original_builder
    print("    trt.Builder restored")

if __name__ == "__main__":
    case_expose_an_internal_tensor()
    case_engine_index_separates_builds()
    case_scope_is_the_whole_contract()
    case_patch_the_builder_class()

    print("\nFinish")
