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
"""P0 -- fold away the shape-computation plumbing before mining.

On a 6-layer TransformerEncoder this halves the graph, 522 nodes / 21 op types
down to 246 / 12. The removed nodes are `Constant` (129), `Identity` (63),
`Shape`, `Slice`, `Cast` and friends. They are not just noise: their count is
not necessarily the same in every layer, so leaving them in *breaks the
periodicity* the whole 1-D reduction relies on.
"""

from __future__ import annotations

import onnx

def preprocess(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict]:
    """Constant-fold and tidy up. Returns the new model plus a small stat dict."""
    stat = {"n_node_before": len(model.graph.node), "n_node_after": len(model.graph.node), "tool": "none"}
    try:
        import onnxslim
        slimmed = onnxslim.slim(model)
        stat["n_node_after"] = len(slimmed.graph.node)
        stat["tool"] = "onnxslim"
        return slimmed, stat
    except Exception as e:  # onnxslim is optional and can trip on exotic models
        stat["error"] = f"{type(e).__name__}: {e}"
        return model, stat
