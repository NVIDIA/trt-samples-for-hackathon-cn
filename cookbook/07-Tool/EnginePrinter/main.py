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

import os
from tensorrt_cookbook import case_mark, export_engine_as_onnx

@case_mark
def case_simple(model_name):

    # Build engine with dumped json file
    command = f"trtexec --onnx=$TRT_COOKBOOK_PATH/00-Data/model/{model_name}.onnx --profilingVerbosity=detailed --exportLayerInfo={mdeol_name}.json --skipInference"
    os.syatem(command)

    # Convert engine to a ONNX-like file
    export_engine_as_onnx(model_name + ".json", model_name + "-network.onnx")

if __name__ == "__main__":
    # Use a network of MNIST
    case_single("model-trained")
    # Use large encodernetwork
    case_simple("model-large")

    print("Finish")
