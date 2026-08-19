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
"""Standalone entrance of the ONNX outliner: run it on **your own** model.

`../01-BasicUsage/main.py` is the cookbook demo and imports `tensorrt_cookbook`. This script
does not: it only needs the `onnx_outliner/` directory sitting beside it, which
is a self-contained copy of `tensorrt_cookbook/onnx_outliner`. Copy this whole
`standalone/` directory anywhere and run

    pip install -r requirements.txt
    python3 main.py my_model.onnx

with no `pip install tensorrt_cookbook` and no `TRT_COOKBOOK_PATH`.

Required: `onnx`, `onnx_graphsurgeon`, `numpy`.
Optional: `onnxslim` (constant folding), `onnxruntime` (numerical cross check),
`tensorrt` (parse check). Each is skipped, not fatal, when missing.
"""

import argparse
import json
import sys
from pathlib import Path

# Running `python3 main.py` already puts this directory first on `sys.path`, but
# `python3 -P main.py` or a symlinked copy does not, so be explicit.
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from onnx_outliner import OutlineConfig, outline
    from onnx_outliner.config import STRICTNESS_LIST
except ImportError as e:  # A missing third-party package, or a missing `onnx_outliner/`
    print(f"Failed to import the outliner: {type(e).__name__}: {e}")
    print("Keep the `onnx_outliner/` directory beside this script, and install the dependencies:")
    print(f"    pip install -r {Path(__file__).resolve().parent / 'requirements.txt'}")
    sys.exit(1)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command line, mirroring `python3 -m onnx_outliner`."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Fold the repeated sub-graphs of an ONNX file into shared local functions, so it becomes readable in Netron.",
        epilog="Example: python3 main.py my_model.onnx -o my_model-outlined.onnx --max-level 2 --report report.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input ONNX file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output ONNX file, default `<input>-outlined.onnx`")
    parser.add_argument("--min-repeat", type=int, default=2, help="A pattern needs at least this many instances")
    parser.add_argument("--min-size", type=int, default=3, help="A pattern needs at least this many nodes")
    parser.add_argument("--strictness", choices=STRICTNESS_LIST, default="L1", help="How much has to agree between two nodes")
    parser.add_argument("--backend", choices=["function", "loop"], default="function", help="Fold into a local function (default) or an ONNX Loop")
    parser.add_argument("--method", choices=["serial", "parallel", "auto"], default="auto", help="Which searches to run")
    parser.add_argument("--wl-radius", type=int, default=1, help="Neighbourhood radius of the graph-space anchor hash")
    parser.add_argument("--max-level", type=int, default=1, help="How many times to fold. 1 = flat, 2 = also wrap repeated groups of blocks, ...")
    parser.add_argument("--domain", default="cookbook.outlined", help="Domain of the generated local functions")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip constant folding")
    parser.add_argument("--no-subgraph", action="store_true", help="Do not mine inside Loop/If bodies")
    parser.add_argument("--strict", action="store_true", help="Fail the run if the onnxruntime cross check does not match")
    parser.add_argument("--beam", type=int, default=1, help="Pattern-selection beam width, 1 = plain greedy")
    parser.add_argument("--tolerance", type=float, default=None, help="Relative tolerance of the onnxruntime cross check, default 0 for function and 1e-5 for loop")
    parser.add_argument("--report", type=Path, default=None, help="Where to write the JSON report")
    return parser.parse_args(argv)

def show(report: dict) -> None:
    """Print the interesting rows of a report."""
    print(f"    {report['n_node_input']:>6} nodes in the input file")
    print(f"    {report['preprocess']['n_node_after']:>6} nodes after {report['preprocess']['tool']} (P0)")
    print(f"    {report['n_node_output']:>6} nodes in the main graph + {report['n_function']} local function(s) "
          f"+ {report['n_loop']} loop(s), coverage {report['coverage']:.1%}")
    for pattern in report["patterns"]:
        nested = f", {pattern['n_original_node']} original nodes each" if pattern["level"] > 0 else ""
        why = f"  (loop rejected: {pattern['loop_rejected_because']})" if pattern["loop_rejected_because"] else ""
        print(f"           L{pattern['level']} {pattern['name']} [{pattern['backend']}]: {pattern['size']:>3} nodes x {pattern['n_instance']:>2} instances, "
              f"gain {pattern['gain']:>4}, interface {pattern['n_function_input']} in / {pattern['n_function_output']} out{nested}{why}")
    if report["subgraph"]["n_subgraph"]:
        print(f"    sub-graphs: {report['subgraph']['n_subgraph']} ({report['subgraph']['n_node']} nodes), mined={report['subgraph']['mined']}")
    if report["rejected"]:
        print(f"    rejected candidates: {report['rejected']}")
    print(f"    verification: {report['verification']}")
    return

def main(argv: list[str] | None = None) -> int:
    """Outline one user model and report. Returns the process exit code."""
    args = parse_args(argv)
    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        return 1

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

    print(f"Outlining {args.input} -> {output_file}")
    # `verify="strict"` raises on a mismatch, which is exactly what `--strict`
    # asks for, but a traceback is not a useful answer for a CLI user.
    try:
        report = outline(args.input, output_file, config)
    except Exception as e:
        print(f"Failed: {type(e).__name__}: {e}")
        return 1
    show(report)

    if args.report:
        args.report.write_text(json.dumps(report, indent=2))
        print(f"    report written to {args.report}")

    if report["n_function"] + report["n_loop"] == 0:
        print("    Nothing was folded. This model may have no repeated sub-graph at this")
        print("    strictness, or the repeats are smaller than --min-size. Try --strictness L3,")
        print("    --min-size 2, or --method parallel.")
    else:
        print(f"    Open {output_file} in Netron: the repeated blocks are now single nodes.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
