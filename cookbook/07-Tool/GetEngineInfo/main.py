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

from pathlib import Path

from tensorrt_cookbook import print_engine_information, print_engine_io_information

trt_file = Path("model.trt")

def case_simple():
    print_engine_information(trt_file=trt_file, plugin_file_list=[], device_index=0)

    print_engine_io_information(trt_file=trt_file, plugin_file_list=[])

if __name__ == "__main__":

    case_simple()

    print("\nFinish")
