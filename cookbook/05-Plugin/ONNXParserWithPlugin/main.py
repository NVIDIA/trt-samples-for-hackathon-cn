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

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxscript
import torch
from onnx.defs import OpSchema
from tensorrt_cookbook import case_mark, cookbook_path, parse_onnx, TRTWrapperV1

model_path = cookbook_path("00-Data", "model")
onnx_file = model_path / "model-addscalar.onnx"
data = {"inputT0": np.ones([4], dtype=np.float32)}
trt_file = Path("model.trt")
# An **absolute** path on purpose. `IPluginRegistry.load_library` hands the string to `dlopen`,
# which searches the loader's library paths and NOT the current directory, so a bare
# "AddScalarPlugin.so" silently fails -- see `case_pytorch_to_onnx_to_trt` for what that looks like.
plugin_file_list = [Path(__file__).parent / "AddScalarPlugin.so"]
onnx_file_from_torch = Path("model-addscalar-from-torch.onnx")

@case_mark
def case_normal():
    tw = TRTWrapperV1(trt_file=trt_file, plugin_file_list=plugin_file_list)
    if tw.engine_bytes is None:  # Create engine from scratch

        parse_onnx(onnx_file, tw.logger, tw.network, tw.builder_config)

        tensor = tw.network.get_input(0)
        tw.profile.set_shape(tensor.name, data["inputT0"].shape, data["inputT0"].shape, data["inputT0"].shape)

        tw.build()
        tw.serialize_engine(trt_file)

    tw.setup(data)
    tw.infer()
    return

# ------------------------------------------------------------------------------------------------
# PyTorch -> ONNX -> TensorRT, emitting a node this directory's plugin implements.
#
# The ONNX file used by `case_normal` arrives ready-made. This case produces the same node from a
# PyTorch model, which is the half that is usually missing: the cookbook could already *parse* an
# ONNX file containing a plugin node, but not *produce* one.
#
# The two mechanisms are adapted from `tensorrt_edgellm/onnx/onnx_custom_schemas.py` and
# `dynamo_translations.py` of https://github.com/NVIDIA/TensorRT-Edge-LLM (Apache-2.0), where they
# are what lets a Hugging Face checkpoint export to ONNX with TensorRT plugin nodes in it. Nothing
# is copied; those files are 2467 and 1858 lines of transformer-specific schemas, this is the same
# two mechanisms on one scalar add.
# ------------------------------------------------------------------------------------------------

def register_onnx_schema() -> bool:
    """Teach the `onnx` package what an `AddScalar` node looks like.

    Upstream calls this before every export and swallows a duplicate registration, because several
    packages may register the same schema into one process. Returns whether this call did the
    registering, which `case_pytorch_to_onnx_to_trt` prints.

    The signature has to match what `AddScalarPlugin` expects: op type `AddScalar`, domain `""`
    (the plugin's namespace is the empty string), one float attribute named `scalar`.
    """
    schema = OpSchema(
        "AddScalar",
        "",  # Domain. Must equal the plugin's namespace, which is "" in AddScalarPlugin.h
        1,  # since_version. Fixed here rather than tracking the opset, as upstream does
        inputs=[OpSchema.FormalParameter("inputT0", "T")],
        outputs=[OpSchema.FormalParameter("outputT0", "T")],
        attributes=[OpSchema.Attribute("scalar", OpSchema.AttrType.FLOAT)],
        type_constraints=[("T", ["tensor(float)"], "")],
    )
    try:
        onnx.defs.register_schema(schema)
        return True
    except Exception:  # noqa: BLE001 - already registered by an earlier call or another package
        return False

@torch.library.custom_op("cookbook::add_scalar", mutates_args=())
def add_scalar(x: torch.Tensor, scalar: float) -> torch.Tensor:
    """The PyTorch side of the operator. Eager implementation, used as the reference."""
    return x + scalar

@add_scalar.register_fake
def _add_scalar_fake(x: torch.Tensor, scalar: float) -> torch.Tensor:
    """Shape/dtype only. `torch.export` traces this, never the real kernel."""
    return torch.empty_like(x)

# Domain "" at version 1, matching the schema above.
custom_opset = onnxscript.values.Opset("", 1)

def translate_add_scalar(inputT0, scalar: float):
    """Map `torch.ops.cookbook.add_scalar` to an ONNX `AddScalar` node.

    A **plain function**, deliberately not decorated with `@onnxscript.script`. A scripted function
    becomes an ONNX local function, and inlining it then fails inside `torch.onnx.export`:

        onnx_ir.passes.PassError: An error occurred when running the
        '<onnx_ir.passes.common.inliner.InlinePass ...>' pass
        ... ValueError: The number of values and replacements must match.

    A plain function emits the node straight into the graph and sidesteps that entirely.
    """
    return custom_opset.AddScalar(inputT0, scalar=scalar)

class AddScalarModel(torch.nn.Module):

    def __init__(self, scalar: float = 1.0):
        super().__init__()
        self.scalar = scalar

    def forward(self, x):
        return torch.ops.cookbook.add_scalar(x, self.scalar)

@case_mark
def case_pytorch_to_onnx_to_trt():
    """Export a PyTorch model to an ONNX file carrying a plugin node, then build and run it."""
    model = AddScalarModel(1.0).eval()
    example_input = torch.ones(4)

    print(f"    Registered the AddScalar OpSchema: {register_onnx_schema()} (False means a previous call did it)")

    onnx_program = torch.onnx.export(
        model,
        (example_input, ),
        dynamo=True,
        opset_version=18,
        # This is the whole of the "A2" mechanism: one entry per custom operator.
        custom_translation_table={torch.ops.cookbook.add_scalar.default: translate_add_scalar},
        input_names=["inputT0"],
        output_names=["outputT0"],
        optimize=True,  # The step that needs the schema registered above
    )
    onnx_program.save(str(onnx_file_from_torch))

    # The node has to match the plugin's identity exactly, so check it rather than assume it.
    graph = onnx.load(str(onnx_file_from_torch)).graph
    node = graph.node[0]
    attribute = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    print(f"    Exported node: op_type={node.op_type!r} domain={node.domain!r} attributes={attribute}")
    assert node.op_type == "AddScalar" and node.domain == "" and attribute["scalar"] == 1.0

    # Parse it with the same plugin `case_normal` uses
    data = {"inputT0": np.ones([4], dtype=np.float32)}
    tw = TRTWrapperV1(plugin_file_list=plugin_file_list)
    parse_onnx(onnx_file_from_torch, tw.logger, tw.network, tw.builder_config)
    tensor = tw.network.get_input(0)
    tw.profile.set_shape(tensor.name, data["inputT0"].shape, data["inputT0"].shape, data["inputT0"].shape)
    assert tw.build(), "Build failed"
    tw.setup(data, b_print_io=False)
    tw.infer(b_print_io=False)

    output_trt = tw.buffer[tw.tensor_name_list[-1]][0]
    output_torch = model(example_input).detach().numpy()
    print(f"    TensorRT {output_trt} vs PyTorch eager {output_torch}")
    assert np.allclose(output_trt, output_torch), f"{output_trt} != {output_torch}"

@case_mark
def case_the_three_traps():
    """Each of the three ways this goes wrong, provoked for real in a fresh process.

    All three were hit while writing `case_pytorch_to_onnx_to_trt`, and **none of them says what is
    actually wrong**. Subprocesses rather than inline code, because all three failures depend on
    process-global state -- a registered schema and a loaded plugin library cannot be un-registered.
    """

    def run(label: str, source: str, expect: str) -> None:
        # A temporary **file**, not `python3 -c`: `@onnxscript.script` reads its function's source
        # with `inspect`, and on a dynamically compiled function that raises
        # `RuntimeError: Decorator script does not work on dynamically compiled function`,
        # which would make trap 2 fail for the wrong reason.
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(source)
            script_file = f.name
        try:
            process = subprocess.run([sys.executable, script_file], capture_output=True, text=True, cwd=str(Path(__file__).parent))
        finally:
            Path(script_file).unlink(missing_ok=True)
        output = process.stdout + process.stderr
        hit = expect in output
        print(f"    {label}")
        for line in [l for l in output.splitlines() if expect in l or l.startswith("RESULT")][:2]:
            print(f"        {line.strip()[:150]}")
        assert hit, f"expected {expect!r} in the output of {label}"

    common = """
import onnx, onnxscript, torch
from onnx.defs import OpSchema
@torch.library.custom_op("cookbook::add_scalar", mutates_args=())
def add_scalar(x: torch.Tensor, scalar: float) -> torch.Tensor: return x + scalar
@add_scalar.register_fake
def _(x, scalar): return torch.empty_like(x)
custom_opset = onnxscript.values.Opset("", 1)
class M(torch.nn.Module):
    def forward(self, x): return torch.ops.cookbook.add_scalar(x, 1.0)
def export(translate):
    return torch.onnx.export(M().eval(), (torch.ones(4),), dynamo=True, opset_version=18,
        custom_translation_table={torch.ops.cookbook.add_scalar.default: translate},
        input_names=["inputT0"], output_names=["outputT0"], optimize=True)
"""
    schema = """
onnx.defs.register_schema(OpSchema("AddScalar", "", 1,
    inputs=[OpSchema.FormalParameter("inputT0", "T")],
    outputs=[OpSchema.FormalParameter("outputT0", "T")],
    attributes=[OpSchema.Attribute("scalar", OpSchema.AttrType.FLOAT)],
    type_constraints=[("T", ["tensor(float)"], "")]))
"""

    # 1. The schema has to exist before the export, which is exactly what the upstream docstring
    #    says: "Call register_... before torch.onnx.export(..., optimize=True) so custom nodes
    #    resolve." Skip it and the exporter cannot construct the node at all.
    run(
        "1. No OpSchema registered ->",
        common + """
def translate(inputT0, scalar: float): return custom_opset.AddScalar(inputT0, scalar=scalar)
try:
    export(translate); print("RESULT exported, which was not expected")
except Exception as e:
    chain = []
    while e is not None:
        chain.append(type(e).__name__); e = e.__cause__
    print("RESULT", " <- ".join(chain))
""",
        "GraphConstructionError",
    )

    # 2. The translation must be a plain function. `@onnxscript.script` makes it an ONNX local
    #    function, and inlining that fails -- with an error naming neither the function nor the op.
    run(
        "2. Translation decorated with @onnxscript.script ->",
        common + schema + """
from onnxscript import opset18 as op
@onnxscript.script(custom_opset)
def translate(inputT0, scalar: float): return custom_opset.AddScalar(inputT0, scalar=scalar)
try:
    export(translate); print("RESULT exported, which was not expected")
except Exception as e:
    print("RESULT", type(e).__name__, str(e)[:80])
""",
        "PassError",
    )

    # 3. The one that cost the most time. A plugin path with no directory component goes to
    #    `dlopen`, which searches the loader's library paths and NOT the current directory.
    run(
        "3. load_library with a bare filename ->",
        """
import tensorrt as trt
reg = trt.get_plugin_registry()
for path in ["AddScalarPlugin.so", "./AddScalarPlugin.so"]:
    handle = reg.load_library(path)
    registered = "AddScalar" in {c.name for c in reg.all_creators}
    print(f"RESULT {path:22} -> {type(handle).__name__:10} AddScalar registered: {registered}")
""",
        "RESULT AddScalarPlugin.so",
    )
    print("        `load_library` returns None and raises nothing. The parser then reports")
    print("        `Plugin not found, are the plugin name, version, and namespace correct?`,")
    print("        which points at three things that are all fine.")

if __name__ == "__main__":
    for trt_path in Path(".").glob("*.trt"):
        trt_path.unlink(missing_ok=True)
    case_normal()
    case_normal()
    case_pytorch_to_onnx_to_trt()
    case_the_three_traps()

    print("Finish")
