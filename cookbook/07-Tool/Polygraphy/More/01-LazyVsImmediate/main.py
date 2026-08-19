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
"""Polygraphy's two API styles, and why every other example here mixes them.

Almost every Polygraphy loader exists twice:

    EngineFromNetwork(...)      CamelCase, **lazy**      -> a callable
    engine_from_network(...)    snake_case, **immediate** -> the object

They are not stylistic variants. The lazy one builds nothing until it is called,
is cheap to construct and can be pickled into another process; the immediate one
does the work now and hands you an object you own.

Read this before the other examples in `More/`: they all pick one style, and the
choice is usually the point.
"""

import copy
import pickle
import time

import numpy as np
from polygraphy import func
from polygraphy.logger import G_LOGGER
from polygraphy.backend.trt import (
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    TrtRunner,
    engine_from_network,
    network_from_onnx_path,
)

from tensorrt_cookbook import case_mark, cookbook_path

# Polygraphy prints the whole builder configuration on every build, which buries
# the two lines each case is actually about.
G_LOGGER.module_severity = G_LOGGER.ERROR

onnx_file = cookbook_path("00-Data", "model", "model-trained.onnx")
data = {"x": np.load(cookbook_path("00-Data", "data", "InferenceData.npy"))}

def output_summary(output: dict) -> str:
    """One-line description of a runner's output dict."""
    return ", ".join(f"{name}{tuple(value.shape)}" for name, value in output.items())

@case_mark
def case_two_spellings() -> None:
    """The same conversion written both ways, and what each hands back.

    The lazy call returns in milliseconds because it has not done anything: an
    `EngineFromNetwork` is a description of the work, not the result. The
    immediate call does the whole build before it returns.
    """
    t0 = time.time()
    build_engine = EngineFromNetwork(NetworkFromOnnxPath(str(onnx_file)), config=CreateConfig())
    lazy_second = time.time() - t0

    t0 = time.time()
    engine = engine_from_network(network_from_onnx_path(str(onnx_file)), config=CreateConfig())
    immediate_second = time.time() - t0

    print(f"    lazy      EngineFromNetwork(...) : {lazy_second * 1000:9.2f} ms -> {type(build_engine).__name__}")
    print(f"    immediate engine_from_network(...): {immediate_second * 1000:9.2f} ms -> {type(engine).__name__}")
    print("    milliseconds against seconds: constructing the lazy form builds nothing at all")

    # Either one can be handed to a runner: `TrtRunner` calls a callable itself.
    with TrtRunner(build_engine) as runner:
        print(f"    TrtRunner(lazy loader) -> {output_summary(runner.infer(data))}")
    with TrtRunner(engine) as runner:
        print(f"    TrtRunner(built engine) -> {output_summary(runner.infer(data))}")
    return

@case_mark
def case_lazy_loaders_are_not_memoised() -> None:
    """The trap: calling one twice builds twice.

    A lazy loader is a recipe, not a cache. Every call re-runs the whole chain,
    so handing the same `build_engine` to two runners -- or calling it once to
    inspect the engine and again to run it -- pays the build cost twice and
    yields two unrelated engines.

    Nothing warns. The only symptom is the wall clock.
    """
    build_engine = EngineFromNetwork(NetworkFromOnnxPath(str(onnx_file)), config=CreateConfig())

    t0 = time.time()
    first = build_engine()
    first_second = time.time() - t0
    t0 = time.time()
    second = build_engine()
    second_second = time.time() - t0

    print(f"    first  call : {first_second:6.2f} s")
    print(f"    second call : {second_second:6.2f} s   same object? {first is second}")
    print("    to build once and reuse, call the loader yourself and pass the result around,")
    print("    or use the immediate API, which makes that the default.")
    return

@case_mark
def case_portability() -> None:
    """Why the lazy style exists at all: a recipe travels, an engine does not.

    An `ICudaEngine` is a live CUDA object -- it cannot be pickled, so it cannot
    be sent to a worker process or a `multiprocessing` pool. A lazy loader is
    plain Python describing what to build, so it copies freely and each worker
    can build its own engine.
    """
    lazy = EngineFromNetwork(NetworkFromOnnxPath(str(onnx_file)), config=CreateConfig())
    built = engine_from_network(network_from_onnx_path(str(onnx_file)), config=CreateConfig())

    for tag, obj in [("lazy loader ", lazy), ("built engine", built)]:
        for how, function in [("deepcopy", copy.deepcopy), ("pickle", pickle.dumps)]:
            try:
                function(obj)
                verdict = "ok"
            except Exception as e:
                verdict = f"{type(e).__name__}: {str(e).splitlines()[0][:52]}"
            print(f"    {tag} {how:<9}: {verdict}")
    print("    so: lazy to describe work that will happen elsewhere, immediate to do it here")
    return

@case_mark
def case_modifying_the_network() -> None:
    """The practical cost of being lazy: you cannot just reach in and edit.

    With the immediate API the network is a normal object, so it is modified in
    place. With the lazy API there is no network yet, so the modification has to
    be described too -- that is what `polygraphy.func.extend` is for: it wraps a
    loader so your function runs on whatever it produces.

    `../04-ExtendInterop/` goes further with `extend`; this is the minimum needed
    to explain the two styles.
    """
    # Immediate: the network exists, edit it.
    builder, network, parser = network_from_onnx_path(str(onnx_file))
    n_layer_before = network.num_layers
    previous = network.get_output(0)
    network.unmark_output(previous)
    identity = network.add_identity(previous).get_output(0)
    identity.name = "y_immediate"
    network.mark_output(identity)
    print(f"    immediate: edited the network directly, {n_layer_before} -> {network.num_layers} layers")
    engine = engine_from_network((builder, network, parser), config=CreateConfig())
    with TrtRunner(engine) as runner:
        print(f"               {output_summary(runner.infer(data))}")

    # Lazy: describe the edit, and let it run when the network is created.
    @func.extend(NetworkFromOnnxPath(str(onnx_file)))
    def load_and_edit(builder, network, parser):
        """Runs on the network the wrapped loader produced; no return needed."""
        previous = network.get_output(0)
        network.unmark_output(previous)
        identity = network.add_identity(previous).get_output(0)
        identity.name = "y_lazy"
        network.mark_output(identity)

    print("    lazy     : wrapped the loader with @func.extend, nothing has run yet")
    with TrtRunner(EngineFromNetwork(load_and_edit, config=CreateConfig())) as runner:
        print(f"               {output_summary(runner.infer(data))}")
    return

if __name__ == "__main__":
    case_two_spellings()
    case_lazy_loaders_are_not_memoised()
    case_portability()
    case_modifying_the_network()

    print("\nFinish")
