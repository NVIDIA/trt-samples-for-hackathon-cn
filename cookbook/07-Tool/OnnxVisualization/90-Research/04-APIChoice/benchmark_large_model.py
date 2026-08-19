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
"""Do the three graph APIs survive a 1.6 GB / 6119-node model?

The worry with any object-oriented wrapper is that it materializes every weight
into a numpy array on import. This measures wall time and peak RSS for
`onnx_graphsurgeon` and `onnx_ir` on top of a plain `onnx.load`.
"""

import resource
import time

import onnx
import onnx_graphsurgeon as gs
import onnx_ir as ir

from tensorrt_cookbook import cookbook_path

onnx_file = cookbook_path("00-Data", "model") / "model-large.onnx"

def peak_rss() -> float:
    """Peak resident set size of this process, in GiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

def timeit(title: str, f):
    """Run `f`, print elapsed time and peak RSS, return the result."""
    t0 = time.time()
    result = f()
    print(f"{title:<24}: {time.time() - t0:6.2f}s  peakRSS={peak_rss():5.1f}GB")
    return result

if __name__ == "__main__":
    print(f"model: {onnx_file} ({onnx_file.stat().st_size / 1024**3:.2f} GB)")
    model = timeit("onnx.load", lambda: onnx.load(onnx_file))
    print(f"{'':24}  nodes={len(model.graph.node)}, initializers={len(model.graph.initializer)}")

    graph = timeit("gs.import_onnx", lambda: gs.import_onnx(model))
    timeit("gs.toposort", graph.toposort)
    timeit("gs.export_onnx", lambda: gs.export_onnx(graph))

    timeit("ir.from_proto", lambda: ir.from_proto(model))

    print("\nFinish")
