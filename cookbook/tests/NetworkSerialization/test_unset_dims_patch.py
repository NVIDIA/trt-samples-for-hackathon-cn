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
"""
Watchdog for how TensorRT represents a `trt.Dims` layer attribute that was never assigned.

On TensorRT 11.3 - the only version this cookbook supports - there are two spellings:

| representation | seen on | symptom |
| --- | --- | --- |
| `None` | `IShuffleLayer.reshape_dims`, `ISliceLayer.axes` / `.start` / `.shape` / `.stride`, `IResizeLayer.shape` in scale mode | the attribute reads back as `None` |
| a raising getter | `I(De)QuantizeLayer.block_shape` | reading it answers `RuntimeError: Could not get block_shape.`, which neither `getattr(obj, name, default)` nor `hasattr` catches - both of those catch only `AttributeError` |

Up to 11.2 there was a third: the `nbDims == -1` sentinel, whose `repr()` printed a garbage rank of
`(80)` or `(81)` while `len()` raised. The serializer carried a `use_patch_80` flag for it. Both the
sentinel and the flag are gone.

What is **not** gone is the need to normalise: `ISliceLayer` still hands back `None` for a dynamic
slice's static `start` / `shape` / `stride`, and `add_slice()` takes `trt.Dims`, not `None`. The
serializer now writes `()` for those unconditionally. `TestSliceNormalisationIsLoadBearing` is the
test that says so, and `tests/NetworkSerialization/test_slice.py` covers the round trip.

`TestUnsetDimsRepresentation` does not pin a representation. It asserts the invariant that matters -
`is_dims_unset` recognises every one of them - and separately reports which representation each
attribute uses, failing only if an unknown one appears. Pinning a specific representation is what
made the earlier version of this file go red on 11.3 while nothing was actually broken.
"""

import json as json_module

import numpy as np
import pytest
import tensorrt as trt
from tensorrt_cookbook import NetworkSerialization, TRTWrapperV2, is_dims_unset

# Every (description, object, attribute name) below names a layer attribute TensorRT leaves unset.
# The attribute is deliberately *not* read here: reading `block_shape` is itself the failure.
def _build_probe_network():
    """Return one network holding every layer whose `trt.Dims` attribute can be left unset."""
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network()
    tensor = network.add_input("tensor", trt.float32, [2, 3, 4, 5])

    probe = {}

    # Shuffle used for transposing only, `reshape_dims` never assigned
    probe["IShuffleLayer.reshape_dims"] = (network.add_shuffle(tensor), "reshape_dims")

    # Slice never given an explicit `axes`
    probe["ISliceLayer.axes"] = (network.add_slice(tensor, [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]), "axes")

    # Slice whose output shape comes from a shape tensor, so the static `shape` stays unset
    layer_slice_dynamic = network.add_slice(tensor, [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1])
    layer_slice_dynamic.set_input(2, network.add_shape(tensor).get_output(0))
    probe["ISliceLayer.shape"] = (layer_slice_dynamic, "shape")

    # Per-tensor quantization, so `block_shape` stays unset
    scale = network.add_constant([1], np.ascontiguousarray(np.array([1.0], dtype=np.float32))).get_output(0)
    probe["IQuantizeLayer.block_shape"] = (network.add_quantize(tensor, scale, trt.int8), "block_shape")

    # Resize driven by `scales`, so the static `shape` stays unset
    layer_resize = network.add_resize(tensor)
    layer_resize.scales = [1, 1, 2, 2]
    probe["IResizeLayer.shape"] = (layer_resize, "shape")

    return builder, network, probe

def _classify(layer, attribute_name: str) -> tuple[str, str]:
    """Return `(representation, detail)` for one unset attribute, without letting it raise."""
    try:
        value = getattr(layer, attribute_name)
    except Exception as error:
        return "raising-getter", f"{type(error).__name__}: {error}"
    if value is None:
        return "none", "None"
    try:
        rank = value.__len__()
    except Exception as error:
        return "unknown", f"__len__ raised {type(error).__name__}: {error}"
    if rank < 0:
        return "negative-rank-sentinel", f"__len__() == {rank}, repr == {value!r}"
    # A non-negative rank means the attribute read back as an ordinary value. For an attribute that
    # was never assigned that is the *fixed* state - and also an ambiguous one, since it is then
    # indistinguishable from an explicitly assigned `[]`.
    return "readable", f"__len__() == {rank}, repr == {value!r}"

_HOSTILE_REPRESENTATION = {"none", "raising-getter"}
_KNOWN_REPRESENTATION = _HOSTILE_REPRESENTATION | {"readable", "negative-rank-sentinel"}

class TestUnsetDimsRepresentation:
    """Assert the invariant; report the representation; fail only on an unknown one."""

    def test_is_dims_unset_recognises_every_representation(self):
        """The one thing that must hold no matter how TensorRT spells "unset"."""
        _builder, _network, probe = _build_probe_network()
        for description, (layer, attribute_name) in probe.items():
            representation, detail = _classify(layer, attribute_name)
            assert is_dims_unset(layer, attribute_name), \
                f"{description}: is_dims_unset() should report True ({representation}: {detail})"
            print(f"{description:32s}: {representation:24s} {detail}")

    def test_no_unknown_representation_has_appeared(self):
        """The watchdog. A new spelling of "unset" means `is_dims_unset` needs another branch."""
        _builder, _network, probe = _build_probe_network()
        unknown = {}
        for description, (layer, attribute_name) in probe.items():
            representation, detail = _classify(layer, attribute_name)
            if representation not in _KNOWN_REPRESENTATION:
                unknown[description] = detail
        assert not unknown, \
            f"TensorRT grew a new representation of an unset Dims: {unknown}. " \
            f"Add a branch to `tensorrt_cookbook.is_dims_unset` and a row to this file's table."

    def test_the_80_81_sentinel_is_really_gone(self):
        """If this fails, the pre-11.3 garbage rank is back and the removed workaround is needed."""
        _builder, _network, probe = _build_probe_network()
        sentinel = {d: detail for d, (layer, name) in probe.items() if _classify(layer, name)[0] == "negative-rank-sentinel" for detail in [_classify(layer, name)[1]]}
        assert not sentinel, \
            f"the `nbDims == -1` sentinel is back: {sentinel}. The serializer used to repair it " \
            f"through a `use_patch_80` flag, removed when the cookbook moved to TensorRT 11.3-only."

    def test_reading_an_unset_attribute_is_still_not_safe(self):
        """At least one attribute still refuses a plain read, so the helper is still earning its keep."""
        _builder, _network, probe = _build_probe_network()
        classified = {description: _classify(layer, name) for description, (layer, name) in probe.items()}
        hostile = {d: detail for d, (representation, detail) in classified.items() if representation in _HOSTILE_REPRESENTATION}
        assert hostile, \
            f"every unset Dims attribute now reads back cleanly ({classified}). " \
            f"Re-check whether `is_dims_unset` is still needed - and note that a readable unset " \
            f"attribute is ambiguous with an explicitly assigned `[]`."

    def test_assigned_dims_are_never_flagged(self):
        """`is_dims_unset` must not fire on a real value, including an explicit empty `[]`."""
        builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
        network = builder.create_network()
        tensor = network.add_input("tensor", trt.float32, [2, 3, 4, 5])

        for value, expected_length in [([], 0), ([0, 0, 0, 0], 4), ([2, 3, 20], 3), ([120], 1)]:
            layer = network.add_shuffle(tensor)
            layer.reshape_dims = value
            assert len(layer.reshape_dims) == expected_length
            assert not is_dims_unset(layer, "reshape_dims"), f"reshape_dims = {value} wrongly reported as unset"
            print(f"reshape_dims = {str(value):14s}: len() == {expected_length}, not flagged")

class TestSliceNormalisationIsLoadBearing:
    """The one normalisation that is not cosmetic: `None` -> `()` for a dynamic slice."""

    def test_add_slice_rejects_none(self):
        """Why the serializer cannot just write what it reads."""
        builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
        network = builder.create_network()
        tensor = network.add_input("tensor", trt.float32, [2, 3, 4])
        with pytest.raises(TypeError):
            network.add_slice(tensor, [0, 0, 0], None, [1, 1, 1])

    def test_dynamic_slice_round_trip(self, tmp_path):
        """A slice whose shape comes from a shape tensor survives serialize -> deserialize."""
        tw = TRTWrapperV2(logger="error")
        tensor = tw.network.add_input("tensor", trt.float32, (2, 3, 4))
        layer = tw.network.add_slice(tensor, [0, 0, 0], [1, 1, 1], [1, 1, 1])
        layer.set_input(2, tw.network.add_shape(tensor).get_output(0))
        tw.build([layer.get_output(0)])

        json_file, para_file = tmp_path / "slice.json", tmp_path / "slice.npz"
        ns = NetworkSerialization(json_file, para_file)
        ns.serialize(
            logger=tw.logger,
            builder=tw.builder,
            builder_config=tw.builder_config,
            network=tw.network,
            optimization_profile_list=[tw.profile],
        )
        layer_dict = next(d for d in json_module.loads(json_file.read_text())["layer"] if "Slice" in d["name"])
        assert layer_dict["shape"] == [], f'the unset static `shape` should be normalised to [], got {layer_dict["shape"]}'

        ns = NetworkSerialization(json_file, para_file)
        ns.deserialize()
        tw_rebuild = TRTWrapperV2(logger="error")
        tw_rebuild.builder, tw_rebuild.network, tw_rebuild.builder_config = ns.builder, ns.network, ns.builder_config
        tw_rebuild.build([])
        assert tw_rebuild.engine_bytes is not None, "the rebuilt dynamic slice should still build"

class TestTransposeOnlyShuffle:
    """A transpose-only Shuffle serializes as `null` and must stay unassigned on rebuild."""

    def test_round_trip(self, tmp_path):
        tw = TRTWrapperV2(logger="error")
        tensor = tw.network.add_input("tensor", trt.float32, (3, 4, 5))
        layer = tw.network.add_shuffle(tensor)  # `reshape_dims` deliberately never assigned
        layer.first_transpose = (2, 0, 1)
        tw.build([layer.get_output(0)])

        json_file, para_file = tmp_path / "shuffle.json", tmp_path / "shuffle.npz"
        ns = NetworkSerialization(json_file, para_file)
        ns.serialize(
            logger=tw.logger,
            builder=tw.builder,
            builder_config=tw.builder_config,
            network=tw.network,
            optimization_profile_list=[tw.profile],
        )
        serialized = json_module.loads(json_file.read_text())["layer"][0]["reshape_dims"]
        assert serialized is None, f"expected the unset marker to serialize as null, got {serialized}"
        assert not (isinstance(serialized, list) and len(serialized) == 1 and serialized[0] > trt.Dims.MAX_DIMS), \
            "a rank above MAX_DIMS means the pre-11.3 garbage rank is back"

        ns = NetworkSerialization(json_file, para_file)
        ns.deserialize()
        tw_rebuild = TRTWrapperV2(logger="error")
        tw_rebuild.builder, tw_rebuild.network, tw_rebuild.builder_config = ns.builder, ns.network, ns.builder_config
        tw_rebuild.build([])
        assert tw_rebuild.engine_bytes is not None, "rebuilding a transpose-only Shuffle should succeed"

class TestScaleModeResize:
    """A scale-mode Resize rebuilds fine either way, but its `shape` must not reach the JSON raw."""

    def test_shape_is_normalised(self, tmp_path):
        tw = TRTWrapperV2(logger="error")
        tensor = tw.network.add_input("tensor", trt.float32, (1, 3, 8, 8))
        layer = tw.network.add_resize(tensor)
        layer.scales = [1, 1, 2, 2]  # `shape` deliberately never assigned
        tw.build([layer.get_output(0)])

        json_file, para_file = tmp_path / "resize.json", tmp_path / "resize.npz"
        ns = NetworkSerialization(json_file, para_file)
        ns.serialize(
            logger=tw.logger,
            builder=tw.builder,
            builder_config=tw.builder_config,
            network=tw.network,
            optimization_profile_list=[tw.profile],
        )

        layer_dict = next(d for d in json_module.loads(json_file.read_text())["layer"] if "Resize" in d["name"])
        assert layer_dict["is_static_scale_mode"] is True
        assert layer_dict["shape"] == [], f'expected a normalised empty "shape", got {layer_dict["shape"]}'
