#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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

    def _build_and_compare(network_builder, *, expect_fail_building: bool = False, expect_exception: type[Exception] | None = None, plugin_file_list: list = []):
        tw = TRTWrapperV2(logger_level="VERBOSE", plugin_file_list=plugin_file_list)
        output_tensor_list, data, *extra_args_list = network_builder(tw)

        tw.build(output_tensor_list)
        if expect_fail_building:
            assert tw.engine_bytes is None
        else:
            tw.setup(data)
            tw.infer()
            output_ref = {name: tw.buffer[name][0] for name in tw.buffer.keys() if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT}

        ns = NetworkSerialization(json_file, para_file)

        plugin_info_dict = {}
        if len(extra_args_list) > 0 and "plugin_info_dict" in extra_args_list[0]:  # Special case for Plugin*Layer test
            plugin_info_dict = extra_args_list[0]["plugin_info_dict"]

        ns.serialize(
            logger=tw.logger,
            builder=tw.builder,
            builder_config=tw.config,
            network=tw.network,
            optimization_profile_list=[tw.profile],
            plugin_info_dict=plugin_info_dict,
        )

        del tw, ns

        ns = NetworkSerialization(json_file, para_file)
        ns.deserialize()

        tw = TRTWrapperV2()
        tw.builder, tw.network, tw.config = ns.builder, ns.network, ns.builder_config

        def _run_tw():
            tw.build()
            if expect_fail_building:
                assert tw.engine_bytes is None
                return
            if len(extra_args_list) > 0 and "runtime_data" in extra_args_list[0]:  # Special case for AssertLayer runtime test
                runtime_data = extra_args_list[0]["runtime_data"]
            else:
                runtime_data = data
                print(runtime_data)
            tw.setup(runtime_data)
            tw.infer()
            output_rebuild = {name: tw.buffer[name][0] for name in tw.buffer.keys() if tw.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT}
            return output_rebuild

        if expect_exception is not None:
            with pytest.raises(expect_exception):
                _run_tw()
            return

        output_rebuild = _run_tw()

        if expect_fail_building:
            # The result has been checked in _run_tw()
            return

        for name in output_ref.keys():
            check_array(output_rebuild[name], output_ref[name], des=name)

    return _build_and_compare
