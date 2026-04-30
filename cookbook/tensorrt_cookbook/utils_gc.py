# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import gc
import os
import weakref
from contextlib import contextmanager
from typing import Optional

import torch

def release_gc():
    """Release memory allocated by PyTorch and Python garbage collector explicitly and immediately."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

PYTHON_DEFAULT_GC_THRESHOLDS = gc.get_threshold()

@contextmanager
def customized_gc_thresholds(gen0_threshold: Optional[int] = None):
    try:
        if gen0_threshold:
            gc.set_threshold(gen0_threshold)
        yield
    finally:
        if gen0_threshold:
            gc.set_threshold(*PYTHON_DEFAULT_GC_THRESHOLDS)

PROFILE_RECORD_GC_ENV_VAR_NAME = "TRT_COOKBOOK_PROFILE_RECORD_GC"

class _GCNvtxHandle:
    pass

_gc_watcher_handle: Optional[_GCNvtxHandle] = None

def _setup_gc_nvtx_profiling() -> Optional[_GCNvtxHandle]:
    global _gc_watcher_handle
    if _gc_watcher_handle is not None:
        return _gc_watcher_handle
    enabled = os.environ.get(PROFILE_RECORD_GC_ENV_VAR_NAME, "0")
    if enabled != "1" or not torch.cuda.is_available():
        return None

    range_id = None

    def gc_callback(phase, _):
        nonlocal range_id
        if phase == "start":
            if range_id is None:
                range_id = torch.cuda.nvtx.range_start("Python GC")
        elif phase == "stop":
            if range_id is not None:
                torch.cuda.nvtx.range_end(range_id)
                range_id = None

    gc.callbacks.append(gc_callback)

    def gc_cleanup(callback):
        try:
            gc.callbacks.remove(callback)
        except ValueError:
            pass

    handle = _GCNvtxHandle()
    weakref.finalize(handle, gc_cleanup, gc_callback)
    _gc_watcher_handle = handle
    return handle

def enable_gc_nvtx_profiling() -> Optional[_GCNvtxHandle]:
    """Enable GC-to-NVTX callbacks when `TRT_COOKBOOK_PROFILE_RECORD_GC=1`."""
    return _setup_gc_nvtx_profiling()
