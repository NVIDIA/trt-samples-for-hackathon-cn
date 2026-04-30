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

import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, cookbook_path, load_mnist_network_trt, check_api_coverage, CookbookErrorRecorder
import numpy as np

data_path = cookbook_path("00-Data", "data")
model_path = cookbook_path("00-Data", "model")
data = {"x": np.load(data_path / "InferenceData.npy")}
shape = list(data["x"].shape)

def case_simple():
    tw = TRTWrapperV1()
    tw.builder_config.set_flag(trt.BuilderFlag.FP16)  # add FP16 to get more alternative algorithms
    tw.builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    load_mnist_network_trt(tw)

    tw.build()
    tw.setup(data)

    inspector = tw.engine.create_engine_inspector()

    # [Optional] Bind an execution context so that the inspector can report context-specific information (e.g. chosen tactics for the profile)
    inspector.execution_context = tw.context  # Default value is None (report build-time information only)
    # [Optional] Bind an error recorder
    inspector.error_recorder = CookbookErrorRecorder()  # Default value is None (no error recording)

    print("Engine information (txt format):")  # engine information is equivalent to put all layer information together
    print(inspector.get_engine_information(trt.LayerInformationFormat.ONELINE))  # text format
    #print("Engine information (json format):")
    #print(inspector.get_engine_information(trt.LayerInformationFormat.JSON))  # json format
    #print("Layer information:")  # Part of the information above
    #for i in range(engine.num_layers):
    #    print(inspector.get_layer_information(i, trt.LayerInformationFormat.ONELINE))

    tw.infer()

    check_api_coverage(inspector)

if __name__ == "__main__":
    #
    case_simple()

    print("Finish")
