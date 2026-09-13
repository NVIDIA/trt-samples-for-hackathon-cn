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
"""Building a command-line tool on Polygraphy's `Tool` base class.

`./plan-size` in this directory is the tool: 50 lines that build an engine and
report how long it took and how big the plan is. What makes it worth writing that
way rather than with `argparse` is `get_subscriptions_impl` -- subscribing to
Polygraphy's argument groups hands you their command-line options, already parsed
and already wired to loaders. `case_what_you_get_for_free` counts them: the one
option `plan-size` declares itself arrives on a parser holding 95.

The bill for that comes in `case_the_dependency_graph_is_a_docstring`. Argument
groups depend on each other, the dependencies are recorded only in prose inside
class docstrings, and forgetting one is a bare `KeyError` seven frames down --
after argument parsing has already succeeded.

Two smaller things worth knowing before you start: `main()` calls `sys.exit`, so
running a tool in-process needs a different entry point
(`case_running_a_tool_without_sys_exit`), and there is no way to add your tool to
the `polygraphy` executable (`case_your_tool_stays_standalone`).
"""

import subprocess
import sys
from pathlib import Path

from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import (DataLoaderArgs, ModelArgs, OnnxInferShapesArgs, OnnxLoadArgs, TrtConfigArgs, TrtLoadEngineBytesArgs, TrtLoadNetworkArgs, TrtLoadPluginsArgs, TrtOnnxFlagArgs)
from polygraphy.tools.base import Tool
from polygraphy.tools.registry import TOOL_REGISTRY

from tensorrt_cookbook import case_mark, cookbook_path

G_LOGGER.module_severity = G_LOGGER.ERROR

tool_path = Path(__file__).parent / "plan-size"
onnx_file = str(cookbook_path("00-Data", "model", "model-trained.onnx"))

def option_count(tool: Tool) -> int:
    """How many distinct command-line options the tool ends up with."""
    parser = tool.setup_parser()
    return len({string for action in parser._actions for string in action.option_strings if string.startswith("--")})

def dependencies_of(arg_group_class) -> list:
    """The `Depends on:` section of an argument group's docstring, as a list."""
    lines = [line.strip() for line in (arg_group_class.__doc__ or "").splitlines()]
    return [line.lstrip("- ").split(":")[0] for line in lines if line.startswith("- ")]

class Minimal(Tool):
    """A tool with no subscriptions at all."""

    def __init__(self):
        super().__init__("minimal")

    def run_impl(self, args):
        return 0

class WithDataLoader(Minimal):
    """A tool that only wants Polygraphy's data loader."""

    def get_subscriptions_impl(self):
        return [DataLoaderArgs()]

class WithTheLot(Minimal):
    """A tool that wants the whole TensorRT build stack, as `./plan-size` does."""

    def get_subscriptions_impl(self):
        return [
            ModelArgs(model_opt_required=True),
            OnnxInferShapesArgs(),
            OnnxLoadArgs(),
            DataLoaderArgs(),
            TrtConfigArgs(),
            TrtLoadPluginsArgs(),
            TrtLoadNetworkArgs(),
            TrtOnnxFlagArgs(),
            TrtLoadEngineBytesArgs(),
        ]

class MissingDependency(Minimal):
    """`OnnxLoadArgs` without the `OnnxInferShapesArgs` it depends on."""

    def get_subscriptions_impl(self):
        return [
            ModelArgs(model_opt_required=True),
            OnnxLoadArgs(),
            DataLoaderArgs(),
            TrtConfigArgs(),
            TrtLoadPluginsArgs(),
            TrtLoadNetworkArgs(),
            TrtOnnxFlagArgs(),
            TrtLoadEngineBytesArgs(),
        ]

    def run_impl(self, args):
        with self.arg_groups[TrtLoadEngineBytesArgs].load_engine_bytes() as plan:
            return plan.nbytes

@case_mark
def case_what_you_get_for_free() -> None:
    """Subscribing to an argument group is what makes this worth doing.

    `LoggerArgs` is added by the base class whether or not you ask, so even the
    empty tool starts with `-v`, `--log-file` and friends. Each subscription adds
    its own options, already parsed into the argument group instance, and already
    connected to the loader that consumes them.

    That is the trade: `./plan-size` declares one option of its own and ends up
    accepting every build flag `polygraphy convert` does -- including
    `--version-compatible`, which is the subject of
    `../17-VersionCompatibility/`.
    """
    for tool in [Minimal(), WithDataLoader(), WithTheLot()]:
        subscriptions = [type(group).__name__ for group in tool.get_subscriptions()]
        print(f"    {type(tool).__name__:<14}: {option_count(tool):>3} options from {len(subscriptions)} subscription(s)")
    print("    the empty tool is not empty -- LoggerArgs is subscribed by the base class")
    return

@case_mark
def case_the_dependency_graph_is_a_docstring() -> None:
    """Argument groups depend on each other, and nothing checks that you noticed.

    The dependencies exist only as a `Depends on:` section in each class's
    docstring. There is no registry, no validation at subscription time, and no
    error until something reaches for a group that is not there -- at which point
    `ArgGroups.__getitem__` raises a plain `KeyError` with a class object in it.

    The important part is *when*: argument parsing succeeds, the tool starts, the
    ONNX file is read, and only then does it fall over. On a real model that is
    several seconds of apparent progress before the failure.
    """
    for group in [OnnxLoadArgs, TrtConfigArgs, TrtLoadNetworkArgs, TrtLoadEngineBytesArgs]:
        print(f"    {group.__name__:<24} depends on {dependencies_of(group)}")

    tool = MissingDependency()
    parser = tool.setup_parser()
    args = parser.parse_args([onnx_file])
    tool.parse(args)
    print(f"    a tool missing OnnxInferShapesArgs parses its arguments fine: model_file={Path(args.model_file).name}")
    try:
        tool.run(args)
    except KeyError as e:
        print(f"    ...and then, at run time: KeyError: {e}")
    print("    no warning at subscription time, and the class name in the KeyError is the only hint")
    return

@case_mark
def case_running_a_tool_without_sys_exit() -> None:
    """`main()` ends in `sys.exit`, so it is not the entry point for a test.

    `Tool.main()` is `setup_parser` + `parse_args` + `parse` + `sys.exit(run(...))`.
    Calling it from a test harness kills the harness. The four lines it wraps are
    public, so the in-process recipe is to call them yourself -- which is also how
    you pass a fixed argument list instead of `sys.argv`.

    `run()` returns the status code, and returns `0` when `run_impl` returns
    `None`.
    """
    tool = WithTheLot()
    parser = tool.setup_parser()
    args = parser.parse_args([onnx_file, "--trt-min-shapes", "x:[1,1,28,28]"])
    tool.parse(args)
    print(f"    parsed without touching sys.argv: min shapes {self_shapes(tool)}")
    print(f"    run() -> {tool.run(args)} (run_impl returned 0)")

    minimal = Minimal()
    minimal_args = minimal.setup_parser().parse_args([])
    minimal.parse(minimal_args)
    print(f"    a run_impl returning None still exits 0: run() -> {minimal.run(minimal_args)}")
    return

def self_shapes(tool: Tool) -> str:
    profiles = tool.arg_groups[TrtConfigArgs].profile_dicts
    return str(dict(profiles[0]) if profiles else None)

@case_mark
def case_run_versus_run_impl_and_the_missing_docstring() -> None:
    """Two ways to get a tool that does not start, both of them cheap to hit.

    The base class defines `run` as a wrapper around `run_impl`, and the upstream
    example (`examples/dev/01_writing_cli_tools/gen-data`) overrides `run`
    directly. That still works -- `main` calls `run` -- but it skips the wrapper,
    which is what logs the module versions and defaults a `None` status to `0`.
    Defining neither is a `NotImplementedError` from inside `run`.

    Separately, `setup_parser` opens with `assert self.__doc__`, so a tool class
    without a docstring fails at start-up. Under `python3 -O` the assert is
    stripped and you get an empty help output instead.
    """

    class NoRunImpl(Tool):
        """A tool that forgot to implement anything."""

        def __init__(self):
            super().__init__("no-run-impl")

    tool = NoRunImpl()
    args = tool.setup_parser().parse_args([])
    tool.parse(args)
    try:
        tool.run(args)
    except NotImplementedError as e:
        print(f"    neither run nor run_impl -> NotImplementedError: {e}")

    class NoDocstring(Tool):

        def __init__(self):
            super().__init__("no-docstring")

        def run_impl(self, args):
            return 0

    try:
        NoDocstring().setup_parser()
    except AssertionError as e:
        print(f"    no class docstring     -> AssertionError: {e}")
    print(f"    the docstring is the help text: {Minimal.__doc__.strip()!r}")
    return

@case_mark
def case_your_tool_stays_standalone() -> None:
    """There is no way to add a subcommand to the `polygraphy` executable.

    `polygraphy/tools/registry.py` is a flat list of `try_register_tool(...)`
    calls evaluated at import. No entry point, no plugin hook, no environment
    variable -- the only supported extension point in the whole CLI is
    `polygraphy.run.plugins`, which adds runners to `polygraphy run` and nothing
    else (see `../19-CustomBackend/`).

    So a `Tool` subclass is always its own executable. That is fine, and it is
    what upstream's example does too; it is just not what "integrate a tool into
    Polygraphy" in the example's closing comment sounds like.
    """
    print(f"    tools in polygraphy's registry: {[tool.name for tool in TOOL_REGISTRY]}")
    print(f"    'plan-size' among them        : {'plan-size' in [tool.name for tool in TOOL_REGISTRY]}")

    result = subprocess.run(["polygraphy", "plan-size", "--help"], capture_output=True, text=True)
    print(f"    `polygraphy plan-size` -> exit {result.returncode}: {result.stderr.strip().splitlines()[-1][:80] if result.stderr.strip() else ''}")

    result = subprocess.run([sys.executable, str(tool_path), onnx_file, "--version-compatible"], capture_output=True, text=True)
    built = [line for line in result.stdout.splitlines() + result.stderr.splitlines() if "Built in" in line]
    print(f"    `./plan-size model.onnx --version-compatible` -> exit {result.returncode}")
    print(f"      {built[0].strip() if built else 'no measurement line'}")
    print("    --version-compatible was never written in plan-size; TrtConfigArgs brought it")
    return

if __name__ == "__main__":
    case_what_you_get_for_free()
    case_the_dependency_graph_is_a_docstring()
    case_running_a_tool_without_sys_exit()
    case_run_versus_run_impl_and_the_missing_docstring()
    case_your_tool_stays_standalone()

    print("\nFinish")
