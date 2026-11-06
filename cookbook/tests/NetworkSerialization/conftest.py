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

from collections import OrderedDict

import pytest
import tensorrt as trt
from tensorrt_cookbook import NetworkSerialization, TRTWrapperV2, check_array

@pytest.fixture(scope="session")
def serialzation_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("trt_ser")
    return root / "net.json", root / "net.npz"

@pytest.fixture
def trt_cookbook_tester(serialzation_files, request):

    print(f"Test {request.node.name}")
    json_file, para_file = serialzation_files

    def _extract(extra_args_list, runtime_data=None):
        default_map = OrderedDict(  # Default map, add more item here if needed
            runtime_data=runtime_data,
            plugin_info_dict={},
            b_provide_plugin_so=True,
        )
        if extra_args_list:  # Overwrite the map if `extra_args_list` is not empty
            for key, default_value in default_map.items():
                default_map[key] = extra_args_list[0].get(key, default_value)

        return default_map.values()  # Output as list

    def _build_and_run(tw: TRTWrapperV2, output_tensor_list: list | None = None, expect_fail_building: bool = False, runtime_data=None, extra_profile_list: list | None = None):
        output_tensor_list = output_tensor_list or []
        extra_profile_list = extra_profile_list or []
        tw.build(output_tensor_list, extra_profile_list=extra_profile_list)
        if expect_fail_building:
            return tw.engine_bytes is None
        else:
            tw.setup(runtime_data)
            tw.infer()
            return {name: tw.buffer[name][0] for name in tw.buffer.keys() if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT}

    def _main_process(
        network_builder,
        *,
        expect_fail_building: bool = False,
        expect_fail_comparsion: bool = False,
        expect_exception: type[Exception] | None = None,
        plugin_file_list: list | None = None,
    ):
        plugin_file_list = plugin_file_list or []
        # Build and run original network
        tw = TRTWrapperV2(logger="error", plugin_file_list=plugin_file_list)
        output_tensor_list, data, *extra_args_list = network_builder(tw)
        output_ref = _build_and_run(tw, output_tensor_list, expect_fail_building, data, [])

        # Serilize the network
        runtime_data, plugin_info_dict, b_provide_plugin_so = _extract(extra_args_list, runtime_data=data)  # Extract extra arguments for special cases
        ns = NetworkSerialization(json_file, para_file)
        ns.serialize(
            logger=tw.logger,
            builder=tw.builder,
            builder_config=tw.builder_config,
            network=tw.network,
            optimization_profile_list=[tw.profile],
            plugin_info_dict=plugin_info_dict,
        )
        del tw, ns

        # Deserialize the network
        ns = NetworkSerialization(json_file, para_file)
        ns.deserialize(plugin_file_list=(plugin_file_list if b_provide_plugin_so else []), )

        tw = TRTWrapperV2(plugin_file_list=plugin_file_list)
        tw.builder, tw.network, tw.builder_config = ns.builder, ns.network, ns.builder_config

        # Check result
        if expect_exception is not None:
            with pytest.raises(expect_exception):
                _build_and_run(tw, [], expect_fail_building, runtime_data, [False])
            return True

        output_rebuild = _build_and_run(tw, [], expect_fail_building, runtime_data, [False])

        if expect_fail_building:
            return output_ref and output_rebuild  # Both of them are True

        if expect_fail_comparsion:
            return True

        # A **relative** tolerance, not just the absolute default. The two sides here are two
        # independently built engines, so TensorRT times tactics twice and may pick different
        # kernels with different accumulation orders. `test_convolution.py::test_case_simple` is
        # the case that forced this: its kernel weights span 1e4 down to 1e-4 on purpose (so that
        # the output reads back as 12345.678 and spells out the input digits), which puts the
        # smallest taps at 7.3e-8 of the sum - below float32's 5.96e-8 relative resolution. Whether
        # they survive depends entirely on summation order, and the two engines differed by exactly
        # one ULP: 7.8e-3 at a value of 78,912. The old fixed 1e-5 bound is 0.0013 ULP there, so it
        # was demanding bit-identical engines and failed intermittently.
        # 1e-6 relative is ~10 ULP: far tighter than any real serialization bug, which would change
        # the numbers outright rather than in the last bit.
        return all(check_array(output_rebuild[name], output_ref[name], des=name, weak=True, relative_epsilon=1e-6) for name in output_ref.keys())

    return _main_process
