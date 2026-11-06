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
"""Run one in-place transform on a network immediately before it is serialized.

Adapted from `core/builder/tensorrt_model_connect/graph_transform.py` in
https://github.com/NVIDIA/TensorRT-Model-Connect (Apache-2.0). The idea is theirs; this version is
rewritten to stand alone, keeps their `ContextVar` scoping, and adds `wrap_builder` because the
cookbook's `TRTWrapperV1` already holds a builder we want to hook rather than a factory we call.

The problem it solves: you want to change a network that somebody else's code builds. The usual
answers are to fork that code, or to accept whatever it marks as output. This gives a third: wrap
the `trt.Builder` so the transform runs inside `build_serialized_network`, at the one moment when
the network is fully constructed and nothing has been compiled yet.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count
from typing import Any, Callable, Iterator

GraphTransform = Callable[[Any, int], None]
"""An in-place ``transform(network, engine_index)`` callback. The return value is ignored."""

# A ContextVar rather than a module global: two threads building different models at the same time
# each see their own transform, and an exception inside the `with` block cannot leak one build's
# transform into the next. `engine_index` comes from a counter shared by one scope, so a transform
# that fires several times can tell the engines apart.
_ACTIVE: ContextVar[tuple[GraphTransform, Iterator[int]] | None] = ContextVar("graph_transform", default=None)

class TransformingBuilder:
    """A `trt.Builder` stand-in that runs the active transform before serializing.

    Everything except `build_serialized_network` is forwarded untouched by `__getattr__`, so code
    holding this object cannot tell the difference -- including `create_network`,
    `create_builder_config` and `create_optimization_profile`, which callers reach for constantly.
    """

    def __init__(self, builder: Any, transform: GraphTransform, index_iterator: Iterator[int]) -> None:
        self._builder = builder
        self._transform = transform
        self._index_iterator = index_iterator

    def __getattr__(self, name: str) -> Any:
        return getattr(self._builder, name)

    def build_serialized_network(self, network: Any, config: Any) -> Any:
        # The transform runs here and not one line earlier in the caller, because this is the last
        # point at which the network is complete AND still mutable. After this call returns there
        # are only engine bytes, and `INetworkDefinition` changes no longer mean anything.
        self._transform(network, next(self._index_iterator))
        return self._builder.build_serialized_network(network, config)

class BuilderFactory:
    """Wrap `trt.Builder` itself, for code that constructs its own builder.

    Use this when you cannot reach the builder object -- patch `trt.Builder` with an instance of
    this class and every builder created inside an active scope comes back wrapped.
    """

    def __init__(self, builder_class: Any) -> None:
        self._builder_class = builder_class

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        builder = self._builder_class(*args, **kwargs)
        active = _ACTIVE.get()
        if active is None:
            return builder
        transform, index_iterator = active
        return TransformingBuilder(builder, transform, index_iterator)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._builder_class, name)

def wrap_builder(builder: Any) -> Any:
    """Wrap one existing builder, or return it unchanged when no scope is active.

    This is the half that `TRTWrapperV1` needs: the wrapper creates its builder in `__init__`, long
    before we get a chance to patch `trt.Builder`.
    """
    active = _ACTIVE.get()
    if active is None:
        return builder
    transform, index_iterator = active
    return TransformingBuilder(builder, transform, index_iterator)

@contextmanager
def graph_transform(transform: GraphTransform | None) -> Iterator[None]:
    """Scope a transform to the builds that happen inside the `with` block.

    Passing `None` is a supported no-op, so a caller can write
    `with graph_transform(maybe_none):` without branching.
    """
    if transform is None:
        yield
        return
    token = _ACTIVE.set((transform, count()))
    try:
        yield
    finally:
        _ACTIVE.reset(token)

def is_active() -> bool:
    """Whether a transform is currently scoped. Only used by this example's assertions."""
    return _ACTIVE.get() is not None
