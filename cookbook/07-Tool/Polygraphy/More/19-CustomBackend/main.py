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
"""Adding a backend to `polygraphy run` with an extension module.

`extension/` here is a real installable package, `polygraphy_cookbook_ref`. It
adds `--cookbook-ref`, a NumPy interpreter that supports more ops than
`--pluginref` does and can compute in float64 -- which turns
`polygraphy run --trt --cookbook-ref --cookbook-ref-precision float64` into a
one-line answer to "is this difference just float32 rounding?".

It also, as a side effect of being imported, registers the cookbook's `AddScalar`
op with Polygraphy's own `PluginRefRunner`. `../16-PluginReference/` found that
there is no command-line flag for that; an extension module is the way.

This example **installs a package into the current environment** (and uninstalls
it again at the end). That is not incidental -- `case_pythonpath_is_not_enough`
shows the package being importable and still invisible to `polygraphy run`,
because entry points are read from installed distribution metadata, not from
`sys.path`.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseRunnerArgs
from polygraphy.tools.run import Run

from tensorrt_cookbook import case_mark

G_LOGGER.module_severity = G_LOGGER.ERROR

here = Path(__file__).parent
extension_dir = here / "extension"
# Relative names, with `cwd=here` below, so the logged commands stay readable.
cancellation_onnx = "model-cancellation.onnx"
addscalar_onnx = "model-addscalar.onnx"

def run_cli(*arguments, expect_failure: bool = False) -> str:
    """Run `polygraphy` and return its combined output."""
    result = subprocess.run(["polygraphy", *arguments], capture_output=True, text=True, cwd=here)
    if bool(result.returncode) != expect_failure:
        print(f"    [unexpected exit code {result.returncode}]")
    return result.stdout + result.stderr

def summarize(output: str, *keys: str) -> None:
    for line in output.splitlines():
        if any(key in line for key in keys):
            print(f"      {line.strip()[:120]}")

def install_extension() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", str(extension_dir)], check=True, capture_output=True)

def uninstall_extension() -> None:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "polygraphy_cookbook_ref"], capture_output=True)
    # A build leaves `*.egg-info` behind, and that is enough metadata for
    # `importlib.metadata` to find the entry point again from `sys.path`.
    # See `case_pythonpath_is_not_enough`.
    for leftover in ["build", "polygraphy_cookbook_ref.egg-info"]:
        shutil.rmtree(extension_dir / leftover, ignore_errors=True)

def write_models() -> None:
    """Two graphs: one that loses precision, one that uses the cookbook's custom op."""
    x = gs.Variable("x", np.float32, (16, ))
    scaled = gs.Variable("scaled", np.float32, (16, ))
    lifted = gs.Variable("lifted", np.float32, (16, ))
    y = gs.Variable("y", np.float32, (16, ))
    third = gs.Constant("third", np.full(16, 1 / 3, dtype=np.float32))
    big = gs.Constant("big", np.full(16, 1e7, dtype=np.float32))
    # (x/3 + 1e7) - 1e7: mathematically x/3, numerically a shredder in float32.
    nodes = [
        gs.Node("Mul", inputs=[x, third], outputs=[scaled]),
        gs.Node("Add", inputs=[scaled, big], outputs=[lifted]),
        gs.Node("Sub", inputs=[lifted, big], outputs=[y]),
    ]
    onnx.save(gs.export_onnx(gs.Graph(nodes=nodes, inputs=[x], outputs=[y], opset=17)), str(here / cancellation_onnx))

    a = gs.Variable("x", np.float32, (4, ))
    b = gs.Variable("y", np.float32, (4, ))
    node = gs.Node("AddScalar", attrs={"scalar": 1.0}, inputs=[a], outputs=[b])
    onnx.save(gs.export_onnx(gs.Graph(nodes=[node], inputs=[a], outputs=[b], opset=17)), str(here / addscalar_onnx))

@case_mark
def case_pythonpath_is_not_enough() -> None:
    """The package must be *installed*, not merely importable.

    `polygraphy run` reads `importlib.metadata.entry_points()` for the group
    `polygraphy.run.plugins`. That comes from installed distribution metadata
    (`*.dist-info` / `*.egg-info`), which `sys.path` alone does not provide. So
    the package imports fine and the option is still missing, with no diagnostic
    of any kind -- `--cookbook-ref` is simply not a recognised argument.

    The half-state worth knowing about: building the package once leaves an
    `*.egg-info` directory in the source tree, and that *is* metadata. After
    `pip uninstall`, `PYTHONPATH=extension/` then keeps working, which looks like
    PYTHONPATH being sufficient. It is the stale build artifact.
    """
    uninstall_extension()
    environment = {"PYTHONPATH": str(extension_dir)}

    imported = subprocess.run([sys.executable, "-c", "import polygraphy_cookbook_ref; print('imported')"], capture_output=True, text=True, env={**dict(PATH="/usr/bin:/usr/local/bin"), **environment})
    print(f"    python3 -c 'import polygraphy_cookbook_ref' with PYTHONPATH: {imported.stdout.strip() or imported.stderr.strip().splitlines()[-1]}")

    result = subprocess.run(["polygraphy", "run", "--help"], capture_output=True, text=True, env={**dict(PATH="/usr/bin:/usr/local/bin"), **environment})
    print(f"    `polygraphy run --help` with the same PYTHONPATH mentions --cookbook-ref: {'--cookbook-ref' in result.stdout}")

    install_extension()
    print(f"    after `pip install ./extension`                              : {'--cookbook-ref' in run_cli('run', '--help')}")
    return

@case_mark
def case_what_the_extension_adds() -> None:
    """One entry point, one argument group, and the option is in `--help`.

    The entry-point group name `polygraphy.run.plugins` is fixed; the entry name
    is not. The target is a callable taking no arguments and returning argument
    group instances, parsed after all of Polygraphy's own.

    Note the scope: this is a `polygraphy run` extension, and only that.
    `convert` and `inspect` never look at the entry point, and there is no
    equivalent hook for adding a whole subcommand (see `../18-WritingACliTool/`).
    """
    output = run_cli("run", "--help")
    summarize(output, "--cookbook-ref ", "--cookbook-ref-precision", "Cookbook NumPy Reference Inference")
    for tool in ["convert", "inspect"]:
        print(f"    `polygraphy {tool} --help` mentions cookbook-ref: {'cookbook-ref' in run_cli(tool, '--help')}")
    return

@case_mark
def case_the_runner_is_run_by_generated_code() -> None:
    """`polygraphy run` writes a Python script and executes it.

    This is the part of the design that is easy to miss: `add_to_script_impl`
    does not construct a runner, it appends lines to a script. `--gen-script -`
    dumps that script to stdout, which is both how you debug an extension and how
    a CLI invocation becomes a starting point for Python code.
    """
    output = run_cli("run", cancellation_onnx, "--cookbook-ref", "--cookbook-ref-precision", "float64", "--gen-script", "-")
    inside = False
    for line in output.splitlines():
        if line.startswith("# Loaders"):
            inside = True
        if inside and line.strip():
            print(f"      {line}")
        if line.startswith("results = "):
            break
    return

@case_mark
def case_more_ops_than_pluginref() -> None:
    """The reason to write a runner at all: `--pluginref` cannot run this graph.

    `PluginRefRunner`'s op table has three entries, so an ordinary `Mul` stops it
    (`../16-PluginReference/`). The extension's runner has a bigger table and one
    thing the built-in cannot express -- a working precision.
    """
    print("    polygraphy run model-cancellation.onnx --pluginref")
    summarize(run_cli("run", cancellation_onnx, "--pluginref", expect_failure=True), "does not have a reference", "FAILED |")
    print("    polygraphy run model-cancellation.onnx --onnxrt --cookbook-ref")
    summarize(run_cli("run", cancellation_onnx, "--onnxrt", "--cookbook-ref"), "Average Metrics", "PASSED |")
    return

@case_mark
def case_float64_as_the_reference() -> None:
    """What the precision option is for.

    The model is `(x/3 + 1e7) - 1e7`, which is `x/3` on paper. In float32 the
    `+1e7` throws away every significant digit of `x/3`, and both ONNX-Runtime
    and the float32 reference reproduce that loss *exactly* -- agreeing to
    max_absdiff=0. Two backends agreeing is not evidence the answer is right.

    Switching the reference to float64 and comparing again turns the same command
    into a numerical-stability check, and the model fails it: max_absdiff 0.29 on
    values of order 0.3, i.e. max_reldiff=1, i.e. nothing survived.

    TensorRT gives the identical answer to ONNX-Runtime here, which is the useful
    part -- the error is the model's, not the backend's.
    """
    print("    float32 reference vs onnxrt:")
    summarize(run_cli("run", cancellation_onnx, "--onnxrt", "--cookbook-ref"), "Average Metrics")
    print("    float64 reference vs onnxrt:")
    summarize(run_cli("run", cancellation_onnx, "--onnxrt", "--cookbook-ref", "--cookbook-ref-precision", "float64", expect_failure=True), "Average Metrics")
    print("    float64 reference vs TensorRT:")
    summarize(run_cli("run", cancellation_onnx, "--trt", "--cookbook-ref", "--cookbook-ref-precision", "float64", expect_failure=True), "Average Metrics")
    print("    the two GPU/CPU float32 backends agree perfectly, and are both wrong by 100%")
    return

@case_mark
def case_registering_an_op_by_importing() -> None:
    """The extension also fixes `--pluginref`, without adding an option.

    `polygraphy_cookbook_ref/__init__.py` calls
    `polygraphy.backend.pluginref.references.register("AddScalar")` at import
    time. `polygraphy run` imports the module to reach the entry point, so the op
    is in the registry by the time `--pluginref` looks.

    That is the only route: `../16-PluginReference/` established that there is no
    CLI flag for registering a reference implementation. It is also a reminder
    that installing an extension module runs its code -- the side effect here is
    deliberate, and would be just as easy to introduce by accident.
    """
    print("    polygraphy run model-addscalar.onnx --pluginref   (with the extension installed)")
    summarize(run_cli("run", addscalar_onnx, "--pluginref"), "Completed 1 iteration", "PASSED |")

    uninstall_extension()
    print("    the same command after `pip uninstall`")
    summarize(run_cli("run", addscalar_onnx, "--pluginref", expect_failure=True), "does not have a reference", "FAILED |")
    install_extension()
    return

@case_mark
def case_option_names_collide() -> None:
    """`polygraphy run` already owns ~200 options, and argparse does not forgive.

    An extension whose option name matches a built-in one takes down the whole
    tool at parser-construction time -- `polygraphy run --help` stops working,
    for every user of that environment, not just for the extension's own
    invocations. Hence the convention of prefixing every option with the runner's
    own name.
    """

    class CollidingArgs(BaseRunnerArgs):
        """
        Colliding Runner: a runner whose option name is already taken.
        """

        def get_name_opt_impl(self):
            return "Colliding Runner", "colliding"

        def add_parser_args_impl(self):
            self.group.add_argument("--onnx-outputs", help="This name belongs to OnnxLoadArgs.", default=None)

        def parse_impl(self, args):
            self.value = args_util.get(args, "onnx_outputs")

        def add_to_script_impl(self, script):
            pass

    class RunWithCollision(Run):
        """
        Run inference, plus one argument group that redefines an existing option.
        """

        def get_subscriptions_impl(self):
            return super().get_subscriptions_impl() + [CollidingArgs()]

    try:
        RunWithCollision().setup_parser()
        print("    no collision detected")
    except Exception as e:
        print(f"    subscribing a group that redefines --onnx-outputs -> {type(e).__name__}: {str(e).splitlines()[0][:100]}")
    print("    it fails while building the parser, so `polygraphy run --help` breaks for everyone")
    return

if __name__ == "__main__":
    write_models()
    try:
        case_pythonpath_is_not_enough()
        case_what_the_extension_adds()
        case_the_runner_is_run_by_generated_code()
        case_more_ops_than_pluginref()
        case_float64_as_the_reference()
        case_registering_an_op_by_importing()
        case_option_names_collide()
    finally:
        # Leave the environment as we found it, even on failure.
        uninstall_extension()

    print("\nFinish")
