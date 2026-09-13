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
"""Configuration of the outliner."""

from dataclasses import dataclass

STRICTNESS_LIST = ["L0", "L1", "L2", "L3"]

@dataclass
class OutlineConfig:
    """Knobs of one outlining run.

    strictness
        How much has to agree before two nodes are considered the same:
        ``L0`` op_type+domain, ``L1`` +attributes, ``L2`` +constant input
        dtype/shape, ``L3`` +activation dtype/rank. See README.md.
    """

    min_repeat: int = 2  # A pattern must have at least this many non-overlapping instances
    min_size: int = 3  # A pattern must have at least this many nodes
    strictness: str = "L1"
    domain: str = "cookbook.outlined"  # Domain of the generated local functions
    preprocess: bool = True  # Constant folding etc., see preprocess.py
    # Also mine inside `Loop` / `If` bodies. Their nodes are just as repetitive as
    # the main graph's, and a sub-graph may call a model-local function, so there
    # is nothing special to do beyond registering the function on the model.
    subgraph: bool = True
    max_pattern: int = 32  # Safety stop for the "find one pattern, commit, repeat" loop
    # How many partial solutions the pattern selection carries forward. 1 is plain
    # greedy: always commit the highest-gain candidate and never reconsider. A
    # larger beam explores keeping the runner-up instead, at `beam` times the
    # search cost per round. See the measurement in ../07-ParallelRepeat/README.md.
    beam: int = 1
    # "serial"   only the 1-D reduction (repeated substring of the canonical order)
    # "parallel" only the graph-space search (WL anchors grown in lockstep)
    # "auto"     serial first, then graph space over whatever is left
    method: str = "auto"
    # Neighbourhood radius of the anchor hash. Small on purpose: a large radius
    # folds the surroundings *outside* the block into the hash, and two instances
    # of one block normally hang off different tensors, so they stop sharing a
    # bucket. Measured on 200 random planted blocks, exact recovery is
    # 95.5% at radius 1, 93.5% at radius 2 and 93.0% at radius 3.
    wl_radius: int = 1
    ordering: str = "dfs"  # "dfs" or "recency", see ordering.py
    # What a folded pattern turns into.
    # "function" a local function. TensorRT inlines it again, so the engine is
    #            untouched. This is the visualization back end and the default.
    # "loop"     an ONNX `Loop`, where possible. The only option that actually
    #            shrinks the TensorRT network, at roughly 2x the latency. Patterns
    #            a `Loop` cannot express fall back to a function.
    backend: str = "function"
    max_block_node: int = 512  # Safety stop for lockstep growth
    # How many times to fold. 1 = flat, the level-0 blocks stay the outermost
    # thing. 2 = after folding the blocks, look for repeated *groups of blocks*
    # and wrap those in a second layer of functions, and so on. One level is
    # enough for most models: even when the block body is itself complex, a
    # reader does not usually need a third layer to follow the graph.
    max_level: int = 1
    verify: str = "report"  # "report" (M1) or "strict" (fail the run on a numeric mismatch)
    # Tolerance of the onnxruntime cross check. `None` picks the right default for
    # the back end: the function back end reproduces the original bit for bit, but
    # the `Loop` back end turns the weights into the output of a runtime `Gather`,
    # so the runtime picks different kernels and the accumulation order changes.
    # Measured on a 12-layer encoder that is a relative error of ~3.6e-07, about
    # three fp32 ULP, stable across inputs. Demanding bit-exactness there would
    # reject a correct model.
    tolerance: float | None = None
    seed: int = 31193  # Seed of the random inputs used by the onnxruntime cross check

    def __post_init__(self) -> None:
        """Reject obviously wrong knob values early."""
        if self.strictness not in STRICTNESS_LIST:
            raise ValueError(f"strictness must be one of {STRICTNESS_LIST}, got {self.strictness!r}")
        if self.min_repeat < 2:
            raise ValueError(f"min_repeat must be >= 2, got {self.min_repeat}")
        if self.min_size < 1:
            raise ValueError(f"min_size must be >= 1, got {self.min_size}")
        if self.beam < 1:
            raise ValueError(f"beam must be >= 1, got {self.beam}")
        if self.max_level < 1:
            raise ValueError(f"max_level must be >= 1, got {self.max_level}")
        if self.backend not in ["function", "loop"]:
            raise ValueError(f"backend must be 'function' or 'loop', got {self.backend!r}")
        if self.ordering not in ["dfs", "recency"]:
            raise ValueError(f"ordering must be 'dfs' or 'recency', got {self.ordering!r}")
        if self.method not in ["serial", "parallel", "auto"]:
            raise ValueError(f"method must be 'serial', 'parallel' or 'auto', got {self.method!r}")
        if self.wl_radius < 0:
            raise ValueError(f"wl_radius must be >= 0, got {self.wl_radius}")
        if self.verify not in ["report", "strict"]:
            raise ValueError(f"verify must be 'report' or 'strict', got {self.verify!r}")
        if self.tolerance is None:
            self.tolerance = 0.0 if self.backend == "function" else 1e-5
        if self.tolerance < 0:
            raise ValueError(f"tolerance must be >= 0, got {self.tolerance}")
