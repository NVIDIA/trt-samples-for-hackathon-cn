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
"""Command line entrance: `python3 -m onnx_outliner model.onnx -o out.onnx`."""

import argparse
import json
from pathlib import Path

from .config import STRICTNESS_LIST, OutlineConfig
from .outliner import outline

def main() -> None:
    """Parse the command line, run the outliner, print a short summary."""
    parser = argparse.ArgumentParser(prog="onnx_outliner", description=__doc__)
    parser.add_argument("input", type=Path, help="Input ONNX file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output ONNX file, default `<input>-outlined.onnx`")
    parser.add_argument("--min-repeat", type=int, default=2, help="A pattern needs at least this many instances")
    parser.add_argument("--min-size", type=int, default=3, help="A pattern needs at least this many nodes")
    parser.add_argument("--strictness", choices=STRICTNESS_LIST, default="L1", help="How much has to agree between two nodes")
    parser.add_argument("--backend", choices=["function", "loop"], default="function", help="Fold into a local function (default) or an ONNX Loop")
    parser.add_argument("--method", choices=["serial", "parallel", "auto"], default="auto", help="Which searches to run, see README.md")
    parser.add_argument("--wl-radius", type=int, default=1, help="Neighbourhood radius of the graph-space anchor hash")
    parser.add_argument("--max-level", type=int, default=1, help="How many times to fold. 1 = flat, 2 = also wrap repeated groups of blocks, ...")
    parser.add_argument("--domain", default="cookbook.outlined", help="Domain of the generated local functions")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip constant folding")
    parser.add_argument("--no-subgraph", action="store_true", help="Do not mine inside Loop/If bodies")
    parser.add_argument("--strict", action="store_true", help="Fail the run if the onnxruntime cross check does not match")
    parser.add_argument("--beam", type=int, default=1, help="Pattern-selection beam width, 1 = plain greedy")
    parser.add_argument("--tolerance", type=float, default=None, help="Relative tolerance of the onnxruntime cross check, default 0 for function and 1e-5 for loop")
    parser.add_argument("--report", type=Path, default=None, help="Where to write the JSON report")
    args = parser.parse_args()

    output_file = args.output or args.input.with_name(f"{args.input.stem}-outlined.onnx")
    config = OutlineConfig(
        min_repeat=args.min_repeat,
        min_size=args.min_size,
        strictness=args.strictness,
        max_level=args.max_level,
        backend=args.backend,
        tolerance=args.tolerance,
        method=args.method,
        wl_radius=args.wl_radius,
        beam=args.beam,
        domain=args.domain,
        preprocess=not args.no_preprocess,
        subgraph=not args.no_subgraph,
        verify="strict" if args.strict else "report",
    )
    report = outline(args.input, output_file, config)

    print(f"{report['n_node_input']} nodes -> preprocess -> {report['n_node_mined']} -> outline -> {report['n_node_output']} "
          f"(+{report['n_function']} function, +{report['n_loop']} loop, coverage {report['coverage']:.1%})")
    if report["subgraph"]["n_subgraph"]:
        print(f"    sub-graphs: {report['subgraph']['n_subgraph']} ({report['subgraph']['n_node']} nodes), mined={report['subgraph']['mined']}")
    for pattern in report["patterns"]:
        nested = f", {pattern['n_original_node']} original nodes each" if pattern["level"] > 0 else ""
        why = f"  (loop rejected: {pattern['loop_rejected_because']})" if pattern["loop_rejected_because"] else ""
        print(f"    L{pattern['level']} {pattern['name']} [{pattern['backend']}]: {pattern['size']} nodes x {pattern['n_instance']} instances, gain {pattern['gain']}{nested}{why}")
    print(f"    verification: {report['verification']}")

    if args.report:
        args.report.write_text(json.dumps(report, indent=2))
        print(f"    report written to {args.report}")
    return

if __name__ == "__main__":
    main()
