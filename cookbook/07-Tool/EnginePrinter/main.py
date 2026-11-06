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

from pathlib import Path

from tensorrt_cookbook import (TRTWrapperV1, build_mnist_network_trt, case_mark, print_engine_io_information)

output_path = Path(__file__).parent
trt_file = output_path / "model-trained.trt"

@case_mark
def case_simple():
    tw = TRTWrapperV1()
    output_tensor_list = build_mnist_network_trt(tw)

    # Add the second optimization profile (the first one is initialized in function `build_mnist_network_trt`)
    profile_large_batch = tw.builder.create_optimization_profile()
    profile_large_batch.set_shape("x", [8, 1, 28, 28], [32, 1, 28, 28], [64, 1, 28, 28])

    tw.build(output_tensor_list, extra_profile_list=[profile_large_batch])
    tw.serialize_engine(trt_file)
    print(f"    Built {trt_file.name}, {trt_file.stat().st_size / (1 << 20):.2f} MiB")

    # Print engine metadata
    # Function `print_engine_information` reads TensorRT engine internal metadata which is not public, so it is not included in the cookbook now.
    # Please contact the author for more information.
    try:
        from tensorrt_cookbook import print_engine_information
        print_engine_information(trt_file=trt_file, plugin_file_list=[], device_index=0)
    except ImportError:
        print("Skip usage of `print_engine_information` since it reads TensorRT engine internal metadata which is not public.")

    # Print engine input / output information
    print_engine_io_information(trt_file=trt_file)

if __name__ == "__main__":
    case_simple()

    print("Finish")
