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
from tensorrt_cookbook import TRTWrapperV1, build_mnist_network_trt, print_context_io_information
from pathlib import Path
import numpy as np

data = {"x": np.load(Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data" / "InferenceData.npy")}

def case_simple():

    tw = TRTWrapperV1()
    output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)
    tw.build(output_tensor_list)

    # Set input shape from input data
    tw.setup(data)

    # Print shape of all input / output tensors in the context
    print_context_io_information(engine=tw.engine, context=tw.context)

if __name__ == "__main__":
    # Use a network of MNIST
    case_simple()

    print("Finish")
