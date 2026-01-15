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

import numpy as np
import tensorrt as trt
from tensorrt_cookbook import CookbookErrorRecorder, TRTWrapperV1, case_mark

data = {"inputT0": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}

@case_mark
def case_buildtime():
    tw = TRTWrapperV1()

    myErrorRecorder = CookbookErrorRecorder()
    tw.builder.error_recorder = myErrorRecorder  # can be assigned to Builder or Network in buildtime
    tw.network.error_recorder = myErrorRecorder

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    tw.network.add_identity(tensor)

    #tw.network.mark_output(layer.get_output(0))  # We build a network without output tensor on purpose

    try:
        print("-------------------------------- Report error during building engine")
        tw.build([])
    except Exception:
        pass

    print("-------------------------------- Report error after building engine")

    n_error = myErrorRecorder.num_errors()
    print(f"There is {n_error} error(s):")
    for i in range(n_error):
        print(f"    Number={i},Code={int(myErrorRecorder.get_error_code(i))},Information=P{myErrorRecorder.get_error_desc(i)}")
    myErrorRecorder.clear()  # clear all error information

@case_mark
def case_runtime():
    tw = TRTWrapperV1()

    tensor = tw.network.add_input("inputT0", trt.float32, data["inputT0"].shape)
    layer = tw.network.add_identity(tensor)
    tensor = layer.get_output(0)
    tensor.name = "outputT0"

    tw.build([tensor])

    #tw.setup(data)  # We skip some prepare work for infernece
    tw.runtime = trt.Runtime(tw.logger)
    tw.engine = tw.runtime.deserialize_cuda_engine(tw.engine_bytes)
    tw.context = tw.engine.create_execution_context()

    myErrorRecorder = CookbookErrorRecorder()
    tw.runtime.error_recorder = myErrorRecorder  # can be assigned to Engine or Runtime or ExecutionContext
    tw.engine.error_recorder = myErrorRecorder
    tw.context.error_recorder = myErrorRecorder

    try:
        print("-------------------------------- Report error during doing inference")
        tw.context.execute_async_v3(0)  # enqueue with nullptr
    except Exception:
        pass

    print("-------------------------------- Report error after doing inference")

    n_error = myErrorRecorder.num_errors()
    print(f"There is {n_error} error(s):")
    for i in range(n_error):
        print(f"    Number={i},Code={int(myErrorRecorder.get_error_code(i))},Information=P{myErrorRecorder.get_error_desc(i)}")
    myErrorRecorder.clear()  # clear all error information

if __name__ == "__main__":
    case_buildtime()
    case_runtime()

    print("Finish")
