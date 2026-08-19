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
"""Reaching through Polygraphy to the raw TensorRT API, with `func.extend`.

Polygraphy does not hide the backend. `@func.extend(SomeLoader(...))` wraps a
lazy loader so your function receives whatever it produced -- a network, a
config -- and can use TensorRT APIs on it directly. Nothing has to be returned;
`extend` takes care of that.

This is the escape hatch for anything Polygraphy does not wrap. It is also the
only way to edit a network while staying in the lazy style, because with lazy
loaders there is no network to edit yet (see `../01-LazyVsImmediate/`).

Two things worth knowing before relying on it, both measured below: the escape
hatch cannot resurrect an API TensorRT has removed, and a `TrtRunner` reuses its
output buffers between calls.
"""

import copy

import numpy as np
import tensorrt as trt
from polygraphy import func
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, TrtRunner
from polygraphy.logger import G_LOGGER

from tensorrt_cookbook import case_mark, cookbook_path

G_LOGGER.module_severity = G_LOGGER.ERROR

onnx_file = str(cookbook_path("00-Data", "model", "model-trained.onnx"))
data = {"x": np.load(cookbook_path("00-Data", "data", "InferenceData.npy"))}

@case_mark
def case_extend_a_network_loader() -> None:
    """The mechanism: your function runs on the network the loader produced.

    Note the decorated function takes the loader's outputs as arguments and
    returns nothing. `EngineFromNetwork` is then given the *function*, not a
    call to it -- the whole chain is still lazy.
    """

    @func.extend(NetworkFromOnnxPath(onnx_file))
    def load_and_rename(builder, network, parser):
        """Runs when the network is created."""
        network.name = "CookbookMnist"
        print(f"    inside extend: {network.num_layers} layers, name set to {network.name!r}")

    print("    before calling anything: the decorated object is just a loader")
    with TrtRunner(EngineFromNetwork(load_and_rename, config=CreateConfig())) as runner:
        output = runner.infer(data)
    print(f"    output: {', '.join(f'{k}{tuple(v.shape)}' for k, v in output.items())}")
    return

@case_mark
def case_extend_a_config_loader() -> None:
    """The same trick on `CreateConfig`, for flags Polygraphy does not expose.

    Anything settable on an `IBuilderConfig` is reachable this way. Here the
    `REFIT` flag, which `CreateConfig` does expose as `refittable=True`, is set
    the raw way to show the two are equivalent.
    """

    @func.extend(CreateConfig())
    def config_with_raw_flag(config):
        """Runs on the freshly created `IBuilderConfig`."""
        config.set_flag(trt.BuilderFlag.REFIT)

    engine = EngineFromNetwork(NetworkFromOnnxPath(onnx_file), config=config_with_raw_flag)()
    print(f"    raw   config.set_flag(BuilderFlag.REFIT) -> engine refittable = {engine.refittable}")
    engine2 = EngineFromNetwork(NetworkFromOnnxPath(onnx_file), config=CreateConfig(refittable=True))()
    print(f"    typed CreateConfig(refittable=True)      -> engine refittable = {engine2.refittable}")
    print("    same result: `extend` is a hatch, not a different mechanism")
    return

@case_mark
def case_the_hatch_cannot_undo_a_removed_api() -> None:
    """The escape hatch stops where TensorRT stops.

    The upstream example for this feature sets `trt.BuilderFlag.FP16` directly,
    presenting it as the way around Polygraphy not supporting a flag. On
    TensorRT 11 that line does not run at all: the precision flags were
    **removed** when networks became strongly typed.

    So `CreateConfig(fp16=True)` raising (see `../02-ComparingBackends/`) is not
    Polygraphy being conservative -- it is reporting an API that no longer
    exists. There is nothing to reach around.
    """
    flag_list = [name for name in dir(trt.BuilderFlag) if not name.startswith("_") and name not in ["name", "value"]]
    for name in ["FP16", "INT8", "BF16", "TF32"]:
        print(f"    trt.BuilderFlag.{name:<5}: {'present' if name in flag_list else 'REMOVED in TensorRT 11'}")
    try:
        trt.BuilderFlag.FP16
    except AttributeError as e:
        print(f"    touching it: AttributeError: {str(e)[:76]}")
    print(f"    what is left: {', '.join(sorted(flag_list))}")
    print("    precision is now declared on the network, not requested from the builder;")
    print("    see the cookbook skill `trt-strong-typing-migration`")
    return

@case_mark
def case_runner_reuses_output_buffers() -> None:
    """The trap: a `TrtRunner` owns its output buffers and reuses them.

    Upstream mentions this in a comment. What it looks like in practice: keep a
    reference to `outputs["y"]`, run `infer()` again, and the reference you kept
    now shows the **new** result. The old values are gone, nothing was raised,
    and the variable name still says what you meant.

    Anything that has to outlive the next `infer()` call needs `copy.deepcopy`.
    """
    first_input = data["x"]
    second_input = np.ones_like(first_input)

    with TrtRunner(EngineFromNetwork(NetworkFromOnnxPath(onnx_file), config=CreateConfig())) as runner:
        first_output = runner.infer({"x": first_input})
        kept_reference = first_output["y"]
        kept_copy = copy.deepcopy(first_output["y"])

        second_output = runner.infer({"x": second_input})

        print(f"    the two inputs really do give different results : {not np.array_equal(kept_copy, second_output['y'])}")
        print(f"    kept reference now matches the SECOND result    : {np.array_equal(kept_reference, second_output['y'])}  <- first result lost")
        print(f"    deepcopy still differs from the second result   : {not np.array_equal(kept_copy, second_output['y'])}  <- survived")
    print("    `outputs` is a view into the runner, not a snapshot")
    return

if __name__ == "__main__":
    case_extend_a_network_loader()
    case_extend_a_config_loader()
    case_the_hatch_cannot_undo_a_removed_api()
    case_runner_reuses_output_buffers()

    print("\nFinish")
