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

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, case_mark, datatype_np_to_trt

data = {"tensor": np.arange(np.prod(60), dtype=np.float32).reshape(3, 4, 5) * 10 - 300}  # [0,59] -> [-300, 290]

@case_mark
def case_int8():
    tw = TRTWrapperV1()
    tw.config.set_flag(trt.BuilderFlag.INT8)
    tensor = tw.network.add_input("tensor", datatype_np_to_trt(data["tensor"].dtype), data["tensor"].shape)
    layer = tw.network.add_cast(tensor, trt.int8)
    layer.get_input(0).dynamic_range = [-300, 300]
    layer.get_output(0).dynamic_range = [-300, 300]
    layer.get_output(0).dtype = trt.DataType.INT8

    tw.build([layer.get_output(0)])
    tw.setup(data)
    tw.infer()

if __name__ == "__main__":
    # A case to use INT8 mode and cast float32 tensor into int8 tensor
    case_int8()  # deprecated

    print("Finish")
