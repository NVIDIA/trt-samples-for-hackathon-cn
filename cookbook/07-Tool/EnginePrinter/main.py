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

from pathlib import Path

from tensorrt_cookbook import case_mark, export_engine_as_onnx

mnist_json_file = Path("model-trained.json")
large_json_file = Path("model-large.json")
output_mnist_onnx_file = Path("model-trained-network.onnx")
output_large_onnx_file = Path("model-large-network.onnx")

@case_mark
def case_mnist():
    export_engine_as_onnx(mnist_json_file, output_mnist_onnx_file)

@case_mark
def case_large():
    export_engine_as_onnx(large_json_file, output_large_onnx_file)

if __name__ == "__main__":
    # Use a network of MNIST
    case_mnist()
    # Use large encodernetwork
    case_large()

    print("Finish")
