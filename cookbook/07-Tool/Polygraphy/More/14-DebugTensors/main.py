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
"""Reading an intermediate tensor without turning it into an engine output.

`MarkDebug` marks tensors in the network; at run time a `trt.IDebugListener` is
called with a device pointer each time one is written. The selling point over the
obvious alternative -- marking the tensor as an output, which
`../13-PerLayerPrecision/` does -- is that the engine's I/O signature does not
change, so the surrounding code does not have to know.

That part holds. The part worth measuring is the cost:
`case_marking_costs_you_even_when_switched_off` finds 1.56x latency from marking
alone, paid at build time, whether or not the tensor is ever read.

Only one run-time step is actually required, and the one that looks required is a
no-op -- `case_the_listener_is_the_only_thing_you_must_do`.
"""

import time

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, MarkDebug, NetworkFromOnnxPath, PostprocessNetwork, TrtRunner
from polygraphy.logger import G_LOGGER

from tensorrt_cookbook import case_mark, cookbook_path

G_LOGGER.module_severity = G_LOGGER.ERROR

onnx_file = str(cookbook_path("00-Data", "model", "model-trained.onnx"))
feed = {"x": np.load(cookbook_path("00-Data", "data", "InferenceData.npy"))}
TARGET = "relu"  # the first ReLU's output, an ordinary intermediate tensor
N_WARMUP, N_TEST = 50, 500

class Listener(trt.IDebugListener):
    """Copies each debug tensor to the host and remembers its shape and range.

    The docstring for `process_debug_tensor` is explicit that the buffer is only
    valid for the duration of the callback, so anything worth keeping has to be
    copied out here rather than referenced.
    """

    def __init__(self) -> None:
        super().__init__()
        self.seen = {}

    def process_debug_tensor(self, addr, location, type, shape, name, stream) -> bool:
        host = np.empty(int(np.prod(shape)), dtype=np.float32)
        cudart.cudaMemcpy(host.ctypes.data, addr, host.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        self.seen[name] = (tuple(shape), float(host.min()), float(host.max()))
        return True

def benchmark(engine) -> float:
    """Mean latency in milliseconds, with no debug state switched on."""
    with TrtRunner(engine) as runner:
        for _ in range(N_WARMUP):
            runner.infer(feed)
        t0 = time.perf_counter()
        for _ in range(N_TEST):
            runner.infer(feed)
    return (time.perf_counter() - t0) * 1000 / N_TEST

def mark_target_as_output(network: trt.INetworkDefinition) -> None:
    """The alternative approach: promote the tensor to a network output."""
    for index in range(network.num_layers):
        tensor = network.get_layer(index).get_output(0)
        if tensor.name == TARGET:
            network.mark_output(tensor)
            return

@case_mark
def case_reading_an_intermediate_tensor() -> None:
    """Mark it, attach a listener, and the value arrives.

    Note where each half lives: `MarkDebug` acts on the **network** (build time),
    the listener on the **execution context** (run time). `TrtRunner` exposes its
    context, so both are reachable without leaving Polygraphy.
    """
    engine = EngineFromNetwork(MarkDebug(NetworkFromOnnxPath(onnx_file), [TARGET]), config=CreateConfig())()
    print(f"    engine.is_debug_tensor({TARGET!r}): {engine.is_debug_tensor(TARGET)}")

    listener = Listener()
    with TrtRunner(engine) as runner:
        runner.context.set_debug_listener(listener)
        # No `set_tensor_debug_state(TARGET, True)` here -- `MarkDebug` already
        # left it on. See `case_the_listener_is_the_only_thing_you_must_do`.
        output = runner.infer(feed)

    print(f"    engine I/O tensors : {[engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]}")
    print(f"    inference returned : {list(output.keys())}")
    for name, (shape, low, high) in listener.seen.items():
        print(f"    captured {name!r}: shape {shape}, range [{low:.4f}, {high:.4f}]")
    print("    the tensor came back without ever appearing in the engine's I/O")
    return

@case_mark
def case_the_listener_is_the_only_thing_you_must_do() -> None:
    """The debug state is already on, and forgetting the listener is silent.

    The obvious reading of the API is that `set_tensor_debug_state(name, True)`
    is what arms a marked tensor. It is not -- `MarkDebug` leaves the state on,
    and `get_debug_state` confirms it before anything is touched. The call is a
    no-op in the direction people write it.

    What is genuinely required is the listener, and omitting it fails the quiet
    way: the inference runs, the outputs are correct, and nothing is captured.
    An empty dict reads like "the tensor was never written" rather than "you
    forgot a call".

    The useful direction is `False`: it mutes one tensor for one run, with no
    rebuild, on an engine where every candidate was marked at build time.
    """
    engine = EngineFromNetwork(MarkDebug(NetworkFromOnnxPath(onnx_file), [TARGET]), config=CreateConfig())()

    for tag, attach_listener, state in [
        ("listener, state untouched ", True, None),
        ("listener, state set True  ", True, True),
        ("listener, state set False ", True, False),
        ("no listener               ", False, None),
    ]:
        listener = Listener()
        with TrtRunner(engine) as runner:
            if attach_listener:
                runner.context.set_debug_listener(listener)
            before = runner.context.get_debug_state(TARGET)
            if state is not None:
                runner.context.set_tensor_debug_state(TARGET, state)
            runner.infer(feed)
        print(f"    {tag}: debug state was already {before}, captured {len(listener.seen)} tensor(s)")
    print("    only `False` and the missing listener change anything -- and only one of them warns (neither)")
    return

@case_mark
def case_against_marking_it_an_output() -> None:
    """The comparison that decides which to use.

    `PostprocessNetwork` + `mark_output` gets the same value with far less
    machinery. What it costs is the engine's I/O signature, which every caller,
    every saved plan and every downstream tool now sees.
    """
    debug_engine = EngineFromNetwork(MarkDebug(NetworkFromOnnxPath(onnx_file), [TARGET]), config=CreateConfig())()
    output_engine = EngineFromNetwork(PostprocessNetwork(NetworkFromOnnxPath(onnx_file), mark_target_as_output), config=CreateConfig())()

    for tag, engine in [("MarkDebug     ", debug_engine), ("mark_output   ", output_engine)]:
        print(f"    {tag}: I/O = {[engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]}")
    print("    `MarkDebug` keeps the contract; `mark_output` changes it for everyone downstream")
    print("    but `mark_output` needs no listener, no debug state, and no callback per inference")
    return

@case_mark
def case_marking_costs_you_even_when_switched_off() -> None:
    """Marking is not free, and the bill arrives at build time.

    None of these runs attaches a listener or enables a debug state -- the debug
    tensor is never actually read. The marked engine is still measurably slower,
    because a tensor that has to remain observable cannot be fused away.

    So `MarkDebug` is a debugging build, not a flag to leave on in production.
    """
    baseline = EngineFromNetwork(NetworkFromOnnxPath(onnx_file), config=CreateConfig())()
    debug = EngineFromNetwork(MarkDebug(NetworkFromOnnxPath(onnx_file), [TARGET]), config=CreateConfig())()
    output = EngineFromNetwork(PostprocessNetwork(NetworkFromOnnxPath(onnx_file), mark_target_as_output), config=CreateConfig())()

    latency = {}
    for tag, engine in [("baseline          ", baseline), ("MarkDebug         ", debug), ("mark_output       ", output)]:
        latency[tag.strip()] = benchmark(engine)
        print(f"    {tag}: {latency[tag.strip()]:6.3f} ms, workspace {engine.device_memory_size_v2:>7} B")
    print(f"    MarkDebug costs {latency['MarkDebug'] / latency['baseline']:.2f}x over baseline, with the debug state never switched on")
    print(f"    mark_output costs {latency['mark_output'] / latency['baseline']:.2f}x -- the cheaper of the two here")
    return

@case_mark
def case_every_unfused_tensor() -> None:
    """`mark_unfused_tensors_as_debug_tensors=True` marks whatever survived fusion.

    Which tensors those are is not known until the build finishes, so nothing is
    visible on the `INetworkDefinition` -- `network.is_debug_tensor` reports zero
    either way, and looking there is what makes this kwarg seem to do nothing.
    The effect only shows up at run time.

    The names that come back are TensorRT's post-fusion internal ones
    (`__myln_k_arg__bb1_3_myl4`), so this is a way to see *that* a value is wrong,
    not to find a named layer from the original model.
    """
    for requested in [True, False]:
        engine = EngineFromNetwork(NetworkFromOnnxPath(onnx_file, mark_unfused_tensors_as_debug_tensors=requested), config=CreateConfig())()
        listener = Listener()
        with TrtRunner(engine) as runner:
            runner.context.set_debug_listener(listener)
            runner.context.unfused_tensors_debug_state = True
            runner.infer(feed)
        print(f"    mark_unfused_tensors_as_debug_tensors={str(requested):<5}: captured {len(listener.seen)} tensor(s)")
        if listener.seen:
            print(f"      names: {sorted(listener.seen)[:4]} ...")
    print("    the kwarg works -- but `network.is_debug_tensor` is the wrong place to check it")
    return

if __name__ == "__main__":
    case_reading_an_intermediate_tensor()
    case_the_listener_is_the_only_thing_you_must_do()
    case_against_marking_it_an_output()
    case_marking_costs_you_even_when_switched_off()
    case_every_unfused_tensor()

    print("\nFinish")
