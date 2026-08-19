# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The argument group that puts `--cookbook-ref` on `polygraphy run`.

A `BaseRunnerArgs` subclass does three things: name the toggle option, declare
its own options, and emit the code that constructs the runner. That last part is
the surprise -- `polygraphy run` does not call your runner, it *generates a
Python script* that does, and `add_to_script_impl` is how you contribute lines to
it. `polygraphy run --gen-script -` prints the result.
"""

from polygraphy import mod
from polygraphy.tools.args import OnnxLoadArgs
from polygraphy.tools.args import util as args_util
from polygraphy.tools.args.base import BaseRunnerArgs
from polygraphy.tools.script import make_invocable

@mod.export()
class CookbookRefRunnerArgs(BaseRunnerArgs):
    # The first line is "Title: description" and the `Depends on:` section is the
    # only place dependencies are recorded -- see `../18-WritingACliTool/`.
    """
    Cookbook NumPy Reference Inference: running inference with the cookbook's NumPy reference runner.

    Depends on:

        - OnnxLoadArgs
    """

    def get_name_opt_impl(self):
        # (human-readable name, option name without leading dashes)
        return "Cookbook NumPy Reference", "cookbook-ref"

    def add_parser_args_impl(self):
        # Prefix every option with the toggle name, or you will collide with one
        # of `polygraphy run`'s ~200 built-in options.
        self.group.add_argument(
            "--cookbook-ref-precision",
            help="Working precision for the reference computation.",
            choices=["float32", "float64"],
            default=None,
        )

    def parse_impl(self, args):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            precision (str): The dtype to compute in.
        """
        self.precision = args_util.get(args, "cookbook_ref_precision")

    def add_to_script_impl(self, script):
        # Reuse Polygraphy's ONNX loading so that every `--onnx-*` option keeps
        # working for our runner too.
        loader_name = self.arg_groups[OnnxLoadArgs].add_to_script(script)
        script.add_import(imports=["GsFromOnnx"], frm="polygraphy.backend.onnx")
        loader_name = script.add_loader(make_invocable("GsFromOnnx", loader_name), loader_id="gs_from_onnx")

        script.add_import(imports=["CookbookRefRunner"], frm="polygraphy_cookbook_ref.backend")
        script.add_runner(make_invocable("CookbookRefRunner", loader_name, precision=self.precision))
        # Unlike a plain `BaseArgs`, a `BaseRunnerArgs` returns nothing here.
