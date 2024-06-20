#
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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
#

import sys

import numpy as np

sys.path.append("/trtcookbook/include")
from utils import MyProfiler, TRTWrapperV1, build_mnist_network_trt, case_mark

data = {"x": np.zeros([1, 1, 28, 28], dtype=np.float32)}

@case_mark
def case_normal(b_emit_profile):
    tw = TRTWrapperV1()

    output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)

    tw.build(output_tensor_list)
    tw.setup(data)

    my_profiler = MyProfiler()
    tw.context.profiler = my_profiler  # assign profiler to context

    # When `tw.context.enqueue_emits_profile` is True, all enqueue will be reported by Profiler.
    # Otherwise, only the ONE enqueue after call of `tw.context.report_to_profiler()` will be reported.
    tw.context.enqueue_emits_profile = b_emit_profile  # default value: True

    tw.infer(b_print_io=False)

    if not b_emit_profile:
        tw.context.report_to_profiler()  # We should enqueue once at least before this call

    tw.infer(b_print_io=False)

if __name__ == "__main__":
    case_normal(True)  # We can see the report two times
    case_normal(False)  # We can only see the report one time

    print("Finish")
