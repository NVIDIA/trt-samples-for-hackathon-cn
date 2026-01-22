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

import os
from pathlib import Path

from tensorrt_cookbook import (case_mark, export_engine_as_onnx, print_engine_information, print_engine_io_information)

@case_mark
def case_simple(model_name):

    # Build engine with dumped json file
    command = f"""trtexec \
        --onnx=$TRT_COOKBOOK_PATH/00-Data/model/{model_name}.onnx \
        --profilingVerbosity=detailed \
        --exportLayerInfo={model_name}.json \
        --saveEngine={model_name}.trt \
        --fp16 \
        --memPoolSize=workspace:1024MiB \
        --builderOptimizationLevel=0 \
        --skipInference \
        """

    if model_name == "model-trained":
        command += \
            """ --profile=0 --minShapes=x:1x1x28x28 --optShapes=x:4x1x28x28 --maxShapes=x:16x1x28x28 \
                --profile=1 --minShapes=x:8x1x28x28 --optShapes=x:32x1x28x28 --maxShapes=x:64x1x28x28 \
            """
    else:
        command += \
            """ --profile=0 \
                --minShapes=input_ids:1x1,attention_mask:1x1 \
                --optShapes=input_ids:1x32,attention_mask:1x32 \
                --maxShapes=input_ids:1x64,attention_mask:1x64 \
            """

    os.system(command)

    # Get engine meta data (engine itself is enough)
    print_engine_information(trt_file=Path(model_name + ".trt"), plugin_file_list=[], device_index=0)

    # Get engine input / output tensor data (engine itself is enough)
    print_engine_io_information(trt_file=Path(model_name + ".trt"), plugin_file_list=[])

    # Convert engine to a ONNX-like file (dumped json file is needed)
    export_engine_as_onnx(engine_json_file=Path(model_name + ".json"), export_onnx_file=Path(model_name + "-network.onnx"))

if __name__ == "__main__":
    # Use a network of MNIST
    case_simple("model-trained")
    # Use large encodernetwork
    case_simple("model-large")

    print("Finish")
