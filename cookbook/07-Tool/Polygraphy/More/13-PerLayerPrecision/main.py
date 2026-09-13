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
"""Per-layer precision and per-tensor formats, against a strongly-typed network.

Polygraphy has four network post-processing loaders in this area. On TensorRT 11
they no longer behave the same way, because **every** network is now strongly
typed and types are declared rather than requested:

    SetLayerPrecisions   clean refusal   -- `ILayer.precision` was removed
    SetTensorDatatypes   raw pybind error -- `ITensor.dtype` has no setter
    SetTensorFormats     works           -- layout is not type
    PostprocessNetwork   works           -- the general escape hatch

The interesting part is the last one still standing. Format is the only per-tensor
knob left, and `case_formats_you_cannot_reach` shows it is itself boxed in by the
dtype that can no longer be changed.

Types are declared at construction instead -- `../05-BuildNetworkByHand/`.
"""

import numpy as np
import tensorrt as trt
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, PostprocessNetwork, SetLayerPrecisions, SetTensorDatatypes, SetTensorFormats, TrtRunner
from polygraphy.logger import G_LOGGER

from tensorrt_cookbook import case_mark, cookbook_path

G_LOGGER.module_severity = G_LOGGER.ERROR

onnx_file = str(cookbook_path("00-Data", "model", "model-trained.onnx"))

def first_layer_name() -> str:
    """The name of the first parsed layer, for `SetLayerPrecisions`."""
    _, network, _ = NetworkFromOnnxPath(onnx_file)()
    return network.get_layer(0).name

def try_build(loader, tag: str):
    """Build and report, without letting a failure stop the case."""
    try:
        engine = EngineFromNetwork(loader, config=CreateConfig())()
        print(f"    {tag}: built")
        return engine
    except Exception as e:
        print(f"    {tag}: {type(e).__name__}: {str(e).splitlines()[0][:88]}")
        return None

@case_mark
def case_everything_is_strongly_typed_now() -> None:
    """The premise the rest of this example rests on.

    `create_network()` with no flags at all already carries `STRONGLY_TYPED`.
    There is no flag to turn it off, and the weakly-typed network -- where
    TensorRT was free to pick a layer's precision and insert casts -- is gone.

    `NetworkFromOnnxPath` still accepts `strongly_typed=False`, and **ignores
    it**. No warning, no error, the flag comes back set.
    """
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network()
    print(f"    trt.Builder.create_network() with no flags -> STRONGLY_TYPED = {network.get_flag(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)}")

    for requested in [True, False, None]:
        _, parsed, _ = NetworkFromOnnxPath(onnx_file, strongly_typed=requested)()
        print(f"    NetworkFromOnnxPath(strongly_typed={str(requested):<5}) -> STRONGLY_TYPED = {parsed.get_flag(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)}")
        del parsed
    print("    `strongly_typed=False` is accepted and silently ignored -- the kwarg has no effect left")
    return

@case_mark
def case_setting_a_layer_precision() -> None:
    """`SetLayerPrecisions` refuses, and refuses well.

    `trt.ILayer.precision` / `.precision_is_set` were removed, along with the
    `OBEY_PRECISION_CONSTRAINTS` and `PREFER_PRECISION_CONSTRAINTS` builder flags
    that gave them their meaning. Polygraphy checks for the attribute and raises a
    message naming the version, which is the most useful thing it could do here.

    Compare `Calibrator` in `../06-Int8IsNowExplicit/` and `TacticRecorder` in
    `../12-TacticsAndReproducibility/`, which construct happily and fail later.
    """
    print(f"    trt.ILayer.precision exists: {hasattr(trt.ILayer, 'precision')}")
    print(f"    BuilderFlag.*PRECISION_CONSTRAINTS: {[f for f in dir(trt.BuilderFlag) if 'PRECISION' in f] or 'none left'}")
    try_build(SetLayerPrecisions(NetworkFromOnnxPath(onnx_file), {first_layer_name(): trt.float16}), "SetLayerPrecisions")
    print("    a named, version-stamped refusal -- the good kind of removal")
    return

@case_mark
def case_setting_a_tensor_datatype() -> None:
    """`SetTensorDatatypes` fails, and fails badly.

    `ITensor.dtype` is read-only on a strongly-typed network, so the assignment
    inside the loader hits pybind's setter check. The message that reaches the
    caller says nothing about strong typing, nothing about the tensor, and
    nothing about the version -- it just names a property.
    """
    try_build(SetTensorDatatypes(NetworkFromOnnxPath(onnx_file), {"y": trt.float16}), "SetTensorDatatypes")
    print("    `AttributeError: property of 'ITensor' object has no setter` is a raw pybind message,")
    print("    not a diagnosis -- worth recognising, because the cause is two layers away")
    print("    the replacement is to declare the type at `add_input`, see ../05-BuildNetworkByHand/")
    return

@case_mark
def case_setting_a_tensor_format() -> None:
    """`SetTensorFormats` still works, because a format is a layout, not a type.

    This is the one per-tensor knob strong typing did not take away. It changes
    how the engine expects the tensor to be laid out in memory, which matters when
    the producer upstream already has the data in that layout.
    """
    for name in ["LINEAR", "CHW32", "HWC"]:
        engine = try_build(SetTensorFormats(NetworkFromOnnxPath(onnx_file), {"x": [getattr(trt.TensorFormat, name)]}), f"format {name:<7}")
        if engine is not None:
            print(f"      engine reports x as {engine.get_tensor_format('x')}, dtype {engine.get_tensor_dtype('x')}")
    print("    the request is honoured and visible on the built engine")
    return

@case_mark
def case_formats_you_cannot_reach() -> None:
    """And the format knob is boxed in by the dtype that can no longer be changed.

    `CHW4` and `HWC8` are defined for INT8 and FP16 respectively. Asking for one
    on a float tensor is rejected at build time -- and the usual fix, changing the
    tensor's dtype, is exactly what `case_setting_a_tensor_datatype` cannot do.

    So on a parsed ONNX the reachable set of formats is decided by the ONNX file,
    not by this loader.
    """
    for name in ["CHW4", "HWC8"]:
        try_build(SetTensorFormats(NetworkFromOnnxPath(onnx_file), {"x": [getattr(trt.TensorFormat, name)]}), f"format {name:<7}")
    print("    `has dataType Float unsupported by tensor's allowed TensorFormats`")
    print("    to reach these, the input must be declared INT8/FP16 when the network is built by hand")
    return

@case_mark
def case_the_general_escape_hatch() -> None:
    """`PostprocessNetwork` takes an arbitrary function over the network.

    The three loaders above are conveniences over this. When the dedicated one is
    gone -- or was never there -- this is the way in, and it stays lazy so the
    result is still a loader the rest of the chain accepts.

    Used here to mark an intermediate tensor as an extra output, which none of the
    dedicated loaders in this file can do.
    """

    def mark_first_activation(network: trt.INetworkDefinition) -> None:
        """Add the first ReLU's output to the network outputs."""
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            if layer.type == trt.LayerType.ACTIVATION:
                network.mark_output(layer.get_output(0))
                print(f"      marked `{layer.get_output(0).name}` from layer `{layer.name}`")
                return

    loader = PostprocessNetwork(NetworkFromOnnxPath(onnx_file), mark_first_activation)
    with TrtRunner(EngineFromNetwork(loader, config=CreateConfig())) as runner:
        output = runner.infer({"x": np.load(cookbook_path("00-Data", "data", "InferenceData.npy"))})
    print(f"    engine outputs: {list(output.keys())}")
    print("    the same mechanism the dedicated loaders are built on, with none of their restrictions")
    return

if __name__ == "__main__":
    case_everything_is_strongly_typed_now()
    case_setting_a_layer_precision()
    case_setting_a_tensor_datatype()
    case_setting_a_tensor_format()
    case_formats_you_cannot_reach()
    case_the_general_escape_hatch()

    print("\nFinish")
