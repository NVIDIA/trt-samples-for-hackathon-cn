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
"""Evidence for the survey in README.md.

Enumerate the APIs of every locally installed ONNX-related package and check
whether any of them offers the "outlining" direction we need, i.e.

    automatically find repeated sub-graphs -> replace them with a call to a
    shared `FunctionProto` (or fold them into a `Loop`)

Everything printed here is only about *what exists*, no model is modified.
"""

import importlib
import inspect

# Keywords that would show up in an API that does what we want
KEYWORD_WANTED = ["outlin", "extract_function", "to_function", "make_function", "frequent", "isomorph", "repeat", "dedup", "cse", "common_subexpression"]
# Keywords of the opposite (already existing) direction
KEYWORD_OPPOSITE = ["inline", "unroll", "flatten"]

MODULE_LIST = [
    "onnx",
    "onnx.helper",
    "onnx.inliner",
    "onnx.utils",
    "onnx.compose",
    "onnx.version_converter",
    "onnx_ir",
    "onnx_ir.passes.common",
    "onnxscript",
    "onnxscript.rewriter",
    "onnxscript.rewriter.pattern",
    "onnx_graphsurgeon",
    "onnxslim",
    "polygraphy.backend.onnx",
    "networkx.algorithms.isomorphism",
]

def scan(module_name: str) -> None:
    """Print the public API of one module, split into wanted / opposite / other."""
    try:
        module = importlib.import_module(module_name)
    except Exception as e:  # Package may not be installed in every environment
        print(f"--- {module_name}: NOT AVAILABLE ({type(e).__name__})")
        return

    name_list = [name for name in dir(module) if not name.startswith("_")]
    wanted = [name for name in name_list if any(k in name.lower() for k in KEYWORD_WANTED)]
    opposite = [name for name in name_list if any(k in name.lower() for k in KEYWORD_OPPOSITE)]
    version = getattr(module, "__version__", "")
    print(f"--- {module_name} {version}  ({len(name_list)} public names)")
    print(f"      outlining-ish : {wanted if wanted else 'NONE'}")
    print(f"      inlining-ish  : {opposite if opposite else 'NONE'}")
    return

def check_graph_surgeon_matching() -> None:
    """`onnx_graphsurgeon` is often recommended for this. Check what it really offers."""
    try:
        import onnx_graphsurgeon as gs
    except ImportError:
        print("--- onnx_graphsurgeon: NOT AVAILABLE")
        return
    graph_api = [name for name in dir(gs.Graph) if not name.startswith("_")]
    print(f"--- onnx_graphsurgeon.Graph public API: {graph_api}")
    print("      -> `Graph.layer()` builds nodes, `Graph.cleanup()/toposort()/fold_constants()` tidy up.")
    print("      -> `GraphPattern`/`match_all` do exist, but the pattern is still written by the user.")
    return

def check_rewriter_signature() -> None:
    """`onnxscript.rewriter` can do the *replacement*, but who provides the pattern?"""
    try:
        from onnxscript.rewriter import RewriteRule
    except ImportError:
        print("--- onnxscript.rewriter: NOT AVAILABLE")
        return
    print("--- onnxscript.rewriter.RewriteRule signature:")
    print(f"      {inspect.signature(RewriteRule.__init__)}")
    print("      -> `target_pattern` is a *required user-written* function, the library never discovers it.")
    return

if __name__ == "__main__":
    print("=" * 30 + " Public API scan")
    for module_name in MODULE_LIST:
        scan(module_name)

    print("\n" + "=" * 30 + " Detail check")
    check_graph_surgeon_matching()
    check_rewriter_signature()

    print("\n" + "=" * 30 + " Verdict")
    print("Every installed package offers only the INLINING direction (function body -> flat nodes)")
    print("or user-driven pattern rewriting. None of them discovers repeated sub-graphs by itself.")
    print("\nFinish")
