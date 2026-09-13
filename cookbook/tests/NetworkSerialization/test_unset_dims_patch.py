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
Watchdog for the "80/81" TensorRT bug that `NetworkSerialization.use_patch_80` works around.

A `trt.Dims`-valued layer attribute that has never been assigned is left at TensorRT's internal
"not set" sentinel `nbDims == -1`. Reading it from Python is hostile:

+ `len(dims)` raises `ValueError: __len__() should return >= 0` - the `len()` builtin refuses the
  negative value the binding returns. Anything built on `len()` fails the same way: `list()`,
  `bool()`, iteration, `in`.
+ `repr(dims)` / `str(dims)` do NOT raise, they print a garbage rank - `(80)` or `(81)` depending
  on the TensorRT build. That asymmetry is why the value is visible in a VS Code watch window
  while the identical expression throws in a normal script, and it is where the name "80/81"
  comes from.

`TestUnsetDimsSentinel` pins the buggy behaviour down. **When TensorRT fixes this, these tests
start failing** - that is the point. At that moment `use_patch_80` and
`tensorrt_cookbook.is_dims_unset` can be deleted.

`TestPatchIsLoadBearing` proves the workaround is still doing real work, by turning it off and
watching a round trip break.
"""

import numpy as np
import pytest
import tensorrt as trt
from tensorrt_cookbook import NetworkSerialization, TRTWrapperV2, is_dims_unset

# Every (description, builder) pair below yields a layer attribute TensorRT leaves unset.
def _build_probe_network():
    """Return one network holding every layer whose `trt.Dims` attribute can be left unset."""
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network()
    tensor = network.add_input("tensor", trt.float32, [2, 3, 4, 5])

    probe = {}

    # Shuffle used for transposing only, `reshape_dims` never assigned
    probe["IShuffleLayer.reshape_dims"] = network.add_shuffle(tensor).reshape_dims

    # Slice never given an explicit `axes`
    layer_slice = network.add_slice(tensor, [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1])
    probe["ISliceLayer.axes"] = layer_slice.axes

    # Slice whose output shape comes from a shape tensor, so the static `shape` stays unset
    layer_slice_dynamic = network.add_slice(tensor, [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1])
    layer_slice_dynamic.set_input(2, network.add_shape(tensor).get_output(0))
    probe["ISliceLayer.shape"] = layer_slice_dynamic.shape

    # Per-tensor quantization, so `block_shape` stays unset
    scale = network.add_constant([1], np.ascontiguousarray(np.array([1.0], dtype=np.float32))).get_output(0)
    probe["IQuantizeLayer.block_shape"] = network.add_quantize(tensor, scale, trt.int8).block_shape

    # Resize driven by `scales`, so the static `shape` stays unset
    layer_resize = network.add_resize(tensor)
    layer_resize.scales = [1, 1, 2, 2]
    probe["IResizeLayer.shape"] = layer_resize.shape

    return builder, network, probe

class TestUnsetDimsSentinel:
    """Characterise the bug. These tests failing is good news: it means TensorRT fixed it."""

    def test_len_builtin_still_raises(self):
        _builder, _network, probe = _build_probe_network()
        for description, dims in probe.items():
            with pytest.raises(ValueError, match="should return >= 0"):
                len(dims)
            print(f"{description:32s}: len() raises as expected")

    def test_sentinel_is_negative_rank(self):
        """The `__len__` slot itself returns -1 cleanly - this is what `is_dims_unset` relies on."""
        _builder, _network, probe = _build_probe_network()
        for description, dims in probe.items():
            assert dims.__len__() == -1, f"{description}: expected the -1 sentinel, got {dims.__len__()}"
            assert is_dims_unset(dims), f"{description}: is_dims_unset() should report True"
            print(f"{description:32s}: __len__() == -1")

    def test_repr_prints_a_garbage_rank(self):
        """`repr` does not raise and reports a rank far above `trt.Dims.MAX_DIMS` - the "80/81"."""
        _builder, _network, probe = _build_probe_network()
        for description, dims in probe.items():
            text = repr(dims)
            assert text.startswith("(") and text.endswith(")"), f"{description}: unexpected repr {text}"
            garbage_rank = int(text[1:-1])
            assert garbage_rank > trt.Dims.MAX_DIMS, f"{description}: repr rank {garbage_rank} is no longer garbage"
            print(f"{description:32s}: repr() == {text} (MAX_DIMS = {trt.Dims.MAX_DIMS})")

    def test_assigned_dims_are_never_flagged(self):
        """`is_dims_unset` must not fire on a real value, including an explicit empty `[]`."""
        builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
        network = builder.create_network()
        tensor = network.add_input("tensor", trt.float32, [2, 3, 4, 5])

        for value, expected_length in [([], 0), ([0, 0, 0, 0], 4), ([2, 3, 20], 3), ([120], 1)]:
            layer = network.add_shuffle(tensor)
            layer.reshape_dims = value
            assert len(layer.reshape_dims) == expected_length
            assert not is_dims_unset(layer.reshape_dims), f"reshape_dims = {value} wrongly reported as unset"
            print(f"reshape_dims = {str(value):14s}: len() == {expected_length}, not flagged")

class TestPatchIsLoadBearing:
    """Turn `use_patch_80` off and watch the same round trip break."""

    @staticmethod
    def _round_trip(tmp_path, use_patch_80: bool):
        """Serialize + deserialize a transpose-only Shuffle, return (serialized reshape_dims, rebuilt engine)."""
        data = {"tensor": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}

        tw = TRTWrapperV2(logger="error")
        tensor = tw.network.add_input("tensor", trt.float32, (3, 4, 5))
        layer = tw.network.add_shuffle(tensor)  # `reshape_dims` deliberately never assigned
        layer.first_transpose = (2, 0, 1)
        tw.build([layer.get_output(0)])

        json_file = tmp_path / f"patch_{use_patch_80}.json"
        para_file = tmp_path / f"patch_{use_patch_80}.npz"
        ns = NetworkSerialization(json_file, para_file)
        ns.use_patch_80 = use_patch_80
        ns.serialize(
            logger=tw.logger,
            builder=tw.builder,
            builder_config=tw.builder_config,
            network=tw.network,
            optimization_profile_list=[tw.profile],
        )
        del tw, ns

        import json as json_module
        serialized = json_module.loads(json_file.read_text())["layer"][0]["reshape_dims"]

        ns = NetworkSerialization(json_file, para_file)
        ns.deserialize()
        tw_rebuild = TRTWrapperV2(logger="error")
        tw_rebuild.builder, tw_rebuild.network, tw_rebuild.builder_config = ns.builder, ns.network, ns.builder_config
        tw_rebuild.build([])

        return serialized, tw_rebuild.engine_bytes

    def test_with_patch_the_round_trip_survives(self, tmp_path):
        serialized, engine_bytes = self._round_trip(tmp_path, use_patch_80=True)
        assert serialized == [], f"expected the unset marker to be normalised to [], got {serialized}"
        assert engine_bytes is not None, "rebuilding the network should succeed while the patch is on"

    def test_scale_mode_resize_does_not_leak_the_sentinel(self, tmp_path):
        """A scale-mode Resize rebuilds fine either way, but its `shape` must not reach the JSON."""
        import json as json_module

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

    def test_without_patch_the_garbage_rank_leaks_into_the_json(self, tmp_path):
        """The sentinel is dumped verbatim, so the rebuilt Shuffle reshapes to a nonsense shape."""
        serialized, engine_bytes = self._round_trip(tmp_path, use_patch_80=False)

        assert len(serialized) == 1 and serialized[0] > trt.Dims.MAX_DIMS, \
            f"expected the garbage rank to leak into the JSON, got reshape_dims = {serialized}"
        assert engine_bytes is None, \
            f"rebuilding should fail with reshape_dims = {serialized}; if it now succeeds, re-check whether the patch is still needed"
