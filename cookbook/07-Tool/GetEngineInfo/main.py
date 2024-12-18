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

# Change of this file should be synchronize with 09-TRTLLM/GetEngineInfo/main.py.

import argparse
from pathlib import Path

from tensorrt_cookbook import print_engine_information, print_io_information

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trt_file",
        "-i",
        type=Path,
        required=True,
        help="Path of TensorRT engine (required)",
    )
    parser.add_argument(
        "--device_index",
        "-d",
        type=int,
        default=0,
        help="Index of current CUDA device (default: 0)",
    )
    parser.add_argument(
        "--plugin_file_list",
        "-p",
        type=Path,
        nargs='+',
        default=[],
        help="Paths of custom plugins (default: None)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # We can use thsese functions independently
    print_engine_information(args.trt_file, args.plugin_file_list, args.device_index)
    print_io_information(args.trt_file, args.plugin_file_list)
    print("\nFinish")
