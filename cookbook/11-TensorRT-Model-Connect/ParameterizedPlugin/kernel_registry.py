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
"""One plugin that hosts any registered kernel, selected by name at build time.

The idea is from `core/runtime/byok/` plus `core/builder/tensorrt_model_connect/byok.py` in
https://github.com/NVIDIA/TensorRT-Model-Connect (Apache-2.0), where a single `TvmFfiKernel` plugin
takes a `kernel_name` string and a JSON `shape_spec` as plugin fields and dispatches to a TVM-FFI
function. Nothing is copied: theirs is 976 lines of C++ `IPluginV2DynamicExt`, this is Python on
`tensorrt.plugin` (the quick-deployable plugin API), which is where the cookbook's other plugin
examples have been heading.

What the pattern buys: adding a kernel is `@register_kernel("name")` on a function. No new plugin
class, no new `IPluginCreator`, no rebuild, no second registration in the plugin registry.

What it costs, and the upstream pays the same price:

+ **The arity is fixed by the registration.** `trtp.register` takes the input count from the
  decorated function's signature, so every kernel here takes exactly two inputs. Upstream solves
  this by carrying `num_inputs` / `num_outputs` in the JSON and calling `add_plugin_v2` with a
  variable list; that escape hatch does not exist in the typed Python API.
+ **The output shape has to come from somewhere.** It cannot be read off the kernel, so it is
  declared in the spec. That is what makes the spec more than decoration -- see `row_sum`, whose
  output has one fewer dimension than its input.
"""

from __future__ import annotations

import json
from typing import Any, Callable

#: name -> {"launch": callable, "rank_delta": int, "doc": str}
KERNEL_REGISTRY: dict[str, dict[str, Any]] = {}

def register_kernel(name: str, *, rank_delta: int = 0) -> Callable:
    """Register a launcher under `name`.

    Args:
        name: the string a network passes as the plugin's `kernel_name` attribute.
        rank_delta: how many dimensions the kernel removes from its first input. 0 for an
            elementwise kernel, -1 for a reduction over the last axis. The plugin needs this at
            *shape-inference* time, long before any kernel runs, which is exactly why it has to be
            declared rather than discovered.
    """

    def decorator(function: Callable) -> Callable:
        assert name not in KERNEL_REGISTRY, f"Kernel {name!r} is already registered"
        KERNEL_REGISTRY[name] = {
            "launch": function,
            "rank_delta": rank_delta,
            "doc": (function.__doc__ or "").strip().splitlines()[0] if function.__doc__ else "",
        }
        return function

    return decorator

def get_kernel(name: str) -> dict[str, Any]:
    if name not in KERNEL_REGISTRY:
        raise KeyError(f"Kernel {name!r} is not registered; known kernels: {sorted(KERNEL_REGISTRY)}")
    return KERNEL_REGISTRY[name]

def make_spec(kernel_name: str, **extra_args: Any) -> str:
    """Build the JSON spec string the plugin carries as an attribute.

    JSON rather than separate attributes for the same reason the upstream does it: the set of
    knobs is per kernel, and a plugin attribute list is fixed at registration time. One opaque
    string keeps the plugin signature stable while the kernels vary.
    """
    get_kernel(kernel_name)  # Fail here, at build time, not inside enqueue
    return json.dumps({"kernel_name": kernel_name, "extra_args": extra_args}, separators=(",", ":"))

def parse_spec(spec: str) -> tuple[str, dict[str, Any]]:
    decoded = json.loads(spec)
    return decoded["kernel_name"], decoded.get("extra_args", {})
