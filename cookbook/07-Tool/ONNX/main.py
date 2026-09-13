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
"""The reference `onnx` library itself, from a TensorRT user's point of view.

Not onnx-graphsurgeon (`../OnnxGraphSurgeon/`), not onnxruntime (`../Onnxruntime/`), not polygraphy
(`../Polygraphy/`) - the plain `import onnx` package. The cookbook already leans on `onnx.helper`
and `onnx.checker` all over the place to *build* toy models, so this file deliberately covers the
other half of the library: the model **surgery and interop** tools that nothing else here touches,
each one shown where it changes what TensorRT does or explains a message TensorRT prints.

+ Steps to run.

```bash
python3 main.py
```
"""

import sys

import numpy as np
import onnx
import tensorrt as trt
from onnx import TensorProto, checker, compose, defs, helper, inliner, numpy_helper, parser, shape_inference
from onnx import utils as onnx_utils
from onnx import version_converter
from onnx.reference import ReferenceEvaluator

from tensorrt_cookbook import case_mark, cookbook_path

input_onnx_file = cookbook_path("00-Data", "model", "model-trained.onnx")

# The TensorRT parser writes its diagnostics straight to stderr, unbuffered, while this script's
# stdout turns block-buffered the moment it is redirected to a log file. Without this the TensorRT
# errors all land at the top of the log, detached from the case that provoked them.
sys.stdout.reconfigure(line_buffering=True)

output_extracted_file = "model_extracted.onnx"
output_metadata_file = "model_metadata.onnx"

def trt_parse(onnx_model):
    """Hand one ModelProto to the TensorRT ONNX parser. Returns (succeeded, layer count, errors)."""
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser_trt = trt.OnnxParser(network, logger)
    ok = parser_trt.parse(onnx_model.SerializeToString())
    error_list = [parser_trt.get_error(i).desc().splitlines()[0] for i in range(parser_trt.num_errors)]
    return ok, network.num_layers, error_list

def shape_of(value_info):
    """Symbolic shape of a ValueInfoProto as a list, keeping dimension *names* where present."""
    return [d.dim_param or d.dim_value for d in value_info.type.tensor_type.shape.dim]

@case_mark
def case_checker():
    """`check_model()` is weaker than it looks; `full_check=True` is the one that agrees with TensorRT.

    The default check validates the proto's *structure* - fields present, names resolvable, opset
    known. It does not run shape inference, so a model whose shapes cannot possibly work is
    "valid". That is the gap where "but the ONNX checker said it was fine" bug reports come from.
    """
    # `w` is [3], multiplied against [N, 4]: there is no broadcast that makes this legal.
    graph = helper.make_graph(
        [helper.make_node("Mul", ["a", "w"], ["b"])],
        "broken",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, ["N", 4])],
        [helper.make_tensor_value_info("b", TensorProto.FLOAT, ["N", 4])],
        [numpy_helper.from_array(np.ones(3, dtype=np.float32), "w")],
    )
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx_model.ir_version = 10

    check_result = {}
    for full_check in [False, True]:
        # The two failure modes have unrelated types: a structural defect raises the checker's own
        # `ValidationError`, while `full_check=True` fails inside shape inference with its
        # `InferenceError`, which is *not* a subclass of it. Catch both or miss half the cases.
        try:
            checker.check_model(onnx_model, full_check=full_check)
            check_result[full_check] = "passed"
        except (checker.ValidationError, onnx.shape_inference.InferenceError) as e:
            check_result[full_check] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
        print(f"    check_model(full_check={full_check!s:5s}) -> {check_result[full_check]}")
    default_check_passed = check_result[False] == "passed"

    ok, _, error_list = trt_parse(onnx_model)
    print(f"    TensorRT on the same model    -> ok={ok}, {error_list[0]}")
    assert default_check_passed is True, "the default check was supposed to pass this model"
    assert not ok, "TensorRT was supposed to reject this model"
    print("    -> the default check approves a model TensorRT rejects; `full_check=True` catches it first,")
    print("       and its message names the operator instead of a shape-context source file")

    # A model over the 2 GiB protobuf limit cannot be `onnx.load`ed into a ModelProto at all, so
    # the checker also takes a *path* and streams it. Same call on the real model, no ModelProto.
    checker.check_model(str(input_onnx_file), full_check=True)
    print(f"    check_model(<path>, full_check=True) on {input_onnx_file.name}: passed")
    print("    -> pass a path, not a ModelProto, for models above the 2 GiB protobuf ceiling")

@case_mark
def case_shape_inference():
    """Where `unk__0` comes from.

    `infer_shapes` propagates shapes forward from the graph inputs. It is *static*: it cannot look
    inside a `Reshape`'s shape tensor, so a `-1` there erases whatever symbol was flowing through.
    That anonymous dimension is the `unk__0` that then shows up in polygraphy and TensorRT output.
    """
    onnx_model = onnx.load(input_onnx_file)
    # The PyTorch exporter already wrote value_info, which would hide what inference derives.
    del onnx_model.graph.value_info[:]
    inferred = shape_inference.infer_shapes(onnx_model, data_prop=True, strict_mode=True)

    shape_dict = {v.name: shape_of(v) for v in inferred.graph.value_info}
    for name in ["max_pool2d_1", "view", "linear", "relu_2", "softmax"]:
        print(f"    {name:14s} {shape_dict[name]}")

    reshape_target = numpy_helper.to_array(next(i for i in onnx_model.graph.initializer if i.name == "val_5"))
    print(f"    the Reshape's shape tensor is {reshape_target.tolist()}")
    assert shape_dict["max_pool2d_1"][0] == "nBS" and shape_dict["view"][0] == "unk__0"
    print("    -> `nBS` survives conv and pool, then dies at the `-1` of the Reshape and never comes back")

    declared_output = {o.name: shape_of(o) for o in onnx_model.graph.output}
    print(f"    but `softmax` is back to {shape_dict['softmax']}, because the exporter *declared* y as {declared_output['y']}")
    print("    -> inference treats declared graph I/O as fact; it did not re-derive the batch symbol")
    print("       Practical reading: a symbol lost mid-graph is not recoverable by inference alone. If you")
    print("       need it back for a TensorRT optimization profile, declare it or rewrite the Reshape.")

@case_mark
def case_defs():
    """The op schema registry: the authoritative answer to "which opset do I need for this op?".

    Nothing else in the toolchain will tell you this. `since_version` is the opset in which the
    op's *current* definition was introduced, which is exactly the number you need when a parser
    complains about an op it "does not support" - often it supports the op, at another opset.
    """
    print(f"    onnx {onnx.__version__} implements opset {defs.onnx_opset_version()}, {len([s for s in defs.get_all_schemas() if s.domain == ''])} ops in the default domain")
    print(f"    {'op':10s} {'latest since_version':>20s}   revised at opsets")
    for op_type in ["Relu", "Conv", "Softmax", "Resize", "Reshape"]:
        revision_list = sorted({defs.get_schema(op_type, v).since_version for v in range(1, defs.onnx_opset_version() + 1) if _has_schema(op_type, v)})
        print(f"    {op_type:10s} {defs.get_schema(op_type).since_version:>20d}   {revision_list}")
    print("    -> `Reshape` has been revised 8 times and `Relu` 4. That churn is what the next case runs into.")

def _has_schema(op_type: str, opset_version: int) -> bool:
    try:
        defs.get_schema(op_type, opset_version)
        return True
    except defs.SchemaError:
        return False

@case_mark
def case_version_converter():
    """Opset conversion works upward and mostly does not work downward.

    Upgrading is what you do when a runtime wants a newer opset; downgrading is what people try
    when a *deployment* target is stuck on an old one. The library only ships adapters for the
    conversions someone wrote, and a missing adapter surfaces as a raw C++ assertion.
    """
    onnx_model = onnx.load(input_onnx_file)
    source_version = onnx_model.opset_import[0].version
    print(f"    {input_onnx_file.name} is opset {source_version}")

    for target_version in [26, 21, 13, 11]:
        try:
            converted = version_converter.convert_version(onnx_model, target_version)
            ok, num_layer, _ = trt_parse(converted)
            print(f"    -> opset {target_version:2d}: converted, {len(converted.graph.node)} nodes; TensorRT parses it, {num_layer} layers")
        except RuntimeError as e:
            print(f"    -> opset {target_version:2d}: {str(e).splitlines()[0].split(': ', 2)[-1]}")

    print("    Two things worth taking away:")
    print("      1. the downgrade failure is an `assert` in C++, not a Python exception with a plan. It does name")
    print("         the op, at the end of a path to a header file - read the tail of the message, not the head.")
    print("      2. TensorRT parsed opsets 18, 21 and 26 into the same 27 layers, so converting *for* TensorRT")
    print("         is almost never the fix. Convert when some other tool in the chain demands a version.")

@case_mark
def case_extract_model():
    """Cut a subgraph out by tensor name - the one-call version of what graphsurgeon does by hand.

    `../OnnxGraphSurgeon/08_isolate_subgraph.py` does this with the GS API. This is the same
    operation with no third-party dependency, and it is the fastest way to hand somebody a minimal
    reproducer for a TensorRT bug: name the two tensors that bracket the problem and ship the file.
    """
    onnx_utils.extract_model(str(input_onnx_file), output_extracted_file, ["max_pool2d_1"], ["relu_2"])
    extracted = onnx.load(output_extracted_file)

    print(f"    {input_onnx_file.name}: {len(onnx.load(input_onnx_file).graph.node)} nodes")
    print(f"    {output_extracted_file}: {len(extracted.graph.node)} nodes {[n.op_type for n in extracted.graph.node]}")
    print(f"    kept initializers  : {[i.name for i in extracted.graph.initializer]}")
    print(f"    new graph input    : {[(i.name, shape_of(i)) for i in extracted.graph.input]}")
    checker.check_model(extracted, full_check=True)
    ok, num_layer, _ = trt_parse(extracted)
    print(f"    the cut-out model is valid and TensorRT builds it: ok={ok}, {num_layer} layers")
    print("    -> it collected the initializers the cut needs and gave the new input the *inferred* shape,")
    print("       symbol included. The two tensors you name are the only thing you have to know.")

@case_mark
def case_compose():
    """Merging two models: `onnx.compose` refuses ambiguity instead of guessing.

    `../OnnxGraphSurgeon/13_merge_two_models.py` splices two models by editing tensors directly,
    which means *you* are responsible for name collisions and opset agreement. `merge_models` makes
    both of those hard errors, which is the entire reason to prefer it.
    """

    def tiny_model(graph_name, op_type, opset_version=18):
        graph = helper.make_graph(
            [helper.make_node(op_type, ["a"], ["b"], name=f"{graph_name}_node")],
            graph_name,
            [helper.make_tensor_value_info("a", TensorProto.FLOAT, ["N", 4])],
            [helper.make_tensor_value_info("b", TensorProto.FLOAT, ["N", 4])],
        )
        onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset_version)])
        onnx_model.ir_version = 10
        return onnx_model

    first_model = tiny_model("g1", "Relu")
    second_model = tiny_model("g2", "Sigmoid")

    # Both models call their tensors `a` and `b`, which is what every independently exported pair does.
    try:
        compose.merge_models(first_model, second_model, io_map=[("b", "a")])
    except ValueError as e:
        print(f"    naive merge          -> ValueError: {str(e).splitlines()[0]}")

    prefixed_model = compose.add_prefix(second_model, "second/")
    merged = compose.merge_models(first_model, prefixed_model, io_map=[("b", "second/a")])
    print(f"    after add_prefix     -> {[(n.name, n.op_type) for n in merged.graph.node]}")
    checker.check_model(merged, full_check=True)

    # Same models, different opsets: exactly the case where hand-splicing silently produces a model
    # whose declared opset does not match half of its nodes.
    try:
        compose.merge_models(first_model, compose.add_prefix(tiny_model("g3", "Sigmoid", opset_version=13), "third/"), io_map=[("b", "third/a")])
    except ValueError as e:
        print(f"    opset 18 + opset 13  -> ValueError: {str(e).splitlines()[0]}")
    print("    -> both refusals are the feature. Splicing by hand would have produced a file that loads,")
    print("       passes the default checker, and is wrong.")

@case_mark
def case_parser_and_reference():
    """Two parts of `onnx` that need no other package at all: a text format, and an evaluator.

    `onnx.parser` reads ONNX's own textual IR, which is a far shorter way to write a reproducer
    than a page of `make_node` calls. `onnx.reference.ReferenceEvaluator` then executes it in pure
    Python - the *specification's* answer, useful precisely when the question is "is onnxruntime
    right or is TensorRT right?", because it is a third opinion that belongs to neither.
    """
    text_model = """
        <ir_version: 10, opset_import: ["": 18]>
        agraph (float[N, 4] X) => (float[N, 4] Y) {
            two = Constant <value_float: float = 2.0> ()
            scaled = Mul(X, two)
            Y = Relu(scaled)
        }
    """
    onnx_model = parser.parse_model(text_model)
    checker.check_model(onnx_model, full_check=True)
    print(f"    parsed from text: {[n.op_type for n in onnx_model.graph.node]}, {len(text_model.splitlines())} lines of source")

    input_data = np.array([[-1, 0, 1, 2], [3, 4, 5, 6]], dtype=np.float32)
    reference_output = ReferenceEvaluator(onnx_model).run(None, {"X": input_data})[0]
    print(f"    ReferenceEvaluator: {reference_output.tolist()}")
    np.testing.assert_allclose(reference_output, np.maximum(input_data * 2, 0))
    print("    -> pure Python, no onnxruntime, no GPU. Slow, but it is the definition rather than an implementation.")

@case_mark
def case_inliner():
    """Local functions, and whether you need to inline them before TensorRT (you do not).

    A local function is a reusable subgraph carried inside the model. `onnx.inliner` expands the
    call sites into ordinary nodes. The question worth answering is whether that is a *required*
    step on the TensorRT path.
    """
    text_model = """
        <ir_version: 10, opset_import: ["": 18, "local": 1]>
        agraph (float[N, 4] X) => (float[N, 4] Y) {
            T = local.Scaled <s: float = 3.0> (X)
            Y = local.Scaled <s: float = 2.0> (T)
        }
        <domain: "local", opset_import: ["": 18]>
        Scaled <s> (x) => (y) {
            k = Constant <value_float: float = @s> ()
            y = Mul(x, k)
        }
    """
    function_model = parser.parse_model(text_model)
    inlined_model = inliner.inline_local_functions(function_model)

    for tag, onnx_model in [("with functions", function_model), ("inlined", inlined_model)]:
        ok, num_layer, _ = trt_parse(onnx_model)
        node_list = [f"{n.domain + '.' if n.domain else ''}{n.op_type}" for n in onnx_model.graph.node]
        print(f"    {tag:14s}: {len(onnx_model.functions)} function(s), nodes {node_list}")
        print(f"    {'':14s}  TensorRT ok={ok}, {num_layer} layers")

    input_data = np.ones((2, 4), dtype=np.float32)
    assert np.array_equal(ReferenceEvaluator(function_model).run(None, {"X": input_data})[0], ReferenceEvaluator(inlined_model).run(None, {"X": input_data})[0])
    print("    -> identical layer counts: the TensorRT parser already inlines local functions itself, so")
    print("       inlining is not a prerequisite for building. Inline for the *readers* - Netron, a diff, a")
    print("       tool that predates functions - not for TensorRT.")

@case_mark
def case_metadata():
    """The provenance fields, which nothing enforces and everybody wishes were filled in.

    Every one of these survives a save/load round trip and none of them affects the engine. They
    are the only place to record which script, commit and quantization recipe produced a `.onnx`
    that will outlive the terminal it was built in.
    """
    onnx_model = onnx.load(input_onnx_file)
    print(f"    as exported : producer={onnx_model.producer_name!r} version={onnx_model.producer_version!r} model_version={onnx_model.model_version} metadata_props={len(onnx_model.metadata_props)}")

    onnx_model.producer_name = "tensorrt-cookbook"
    onnx_model.producer_version = "07-Tool/ONNX"
    onnx_model.model_version = 1
    onnx_model.domain = "ai.cookbook"
    onnx_model.doc_string = "MNIST CNN, re-stamped to show what the metadata fields carry"
    for key, value in [("build.commit", "deadbeef"), ("quantization", "none")]:
        entry = onnx_model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(onnx_model, output_metadata_file)

    reloaded = onnx.load(output_metadata_file)
    print(f"    round-tripped: producer={reloaded.producer_name!r} version={reloaded.producer_version!r} model_version={reloaded.model_version} domain={reloaded.domain!r}")
    print(f"                   doc_string={reloaded.doc_string!r}")
    print(f"                   metadata_props={{{', '.join(f'{p.key!r}: {p.value!r}' for p in reloaded.metadata_props)}}}")
    assert {p.key for p in reloaded.metadata_props} == {"build.commit", "quantization"}

    ok, num_layer, _ = trt_parse(reloaded)
    print(f"    TensorRT: ok={ok}, {num_layer} layers - unchanged, the parser ignores all of it")
    print("    -> free-form and never validated, so it is on you to write it. `ir_version` and `opset_import`")
    print("       are the two fields in this area that *do* change behaviour; see `case_version_converter`.")

if __name__ == "__main__":
    # `check_model()` is not the guarantee it looks like
    case_checker()
    # Where `unk__0` comes from
    case_shape_inference()
    # Which opset does this op need?
    case_defs()
    # Opset up-conversion works, down-conversion mostly does not
    case_version_converter()
    # Cut a subgraph out by tensor name
    case_extract_model()
    # Merge two models without hand-splicing them
    case_compose()
    # ONNX text format, and a pure-Python evaluator
    case_parser_and_reference()
    # Local functions, and whether TensorRT needs them inlined
    case_inliner()
    # The provenance fields
    case_metadata()

    print("Finish")
