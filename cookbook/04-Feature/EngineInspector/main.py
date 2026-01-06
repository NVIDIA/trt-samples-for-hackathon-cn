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

import tensorrt as trt
from tensorrt_cookbook import TRTWrapperV1, build_mnist_network_trt

tw = TRTWrapperV1()
tw.config.set_flag(trt.BuilderFlag.FP16)  # add FP16 to get more alternative algorithms
tw.config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
output_tensor_list = build_mnist_network_trt(tw.config, tw.network, tw.profile)

tw.build(output_tensor_list)

engine = trt.Runtime(tw.logger).deserialize_cuda_engine(tw.engine_bytes)

print(f"\n---------------------------------------------------------------- Inspector related")
inspector = engine.create_engine_inspector()
print("Engine information (txt format):")  # engine information is equivalent to put all layer information together
print(inspector.get_engine_information(trt.LayerInformationFormat.ONELINE))  # text format
#print("Engine information (json format):")
#print(inspector.get_engine_information(trt.LayerInformationFormat.JSON))  # json format
#print("Layer information:")  # Part of the information above
#for i in range(engine.num_layers):
#    print(inspector.get_layer_information(i, trt.LayerInformationFormat.ONELINE))
