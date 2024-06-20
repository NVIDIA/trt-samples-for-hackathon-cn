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

sys.path.append("/trtcookbook/include")
from utils import MyProgressMonitor, TRTWrapperV1, build_mnist_network_trt

tw = TRTWrapperV1()
tw.config.progress_monitor = MyProgressMonitor(False)

output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)

tw.build(output_tensor_list)
