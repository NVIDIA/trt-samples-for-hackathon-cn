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

import tensorrt as trt
from tensorrt_cookbook import (TRTWrapperV1, case_mark, print_engine_io_information)

@case_mark
def case_combination(b_strongly_typed: bool, b_set_output_type_as_fp16: bool):

    tw = TRTWrapperV1()
    tw.config.set_flag(trt.BuilderFlag.FP16)
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)) if b_strongly_typed else 0
    tw.network = tw.builder.create_network(flag)

    x = tw.network.add_input("x", trt.float32, (1, 4))
    identity_layer = tw.network.add_identity(x)
    if b_set_output_type_as_fp16:
        identity_layer.set_output_type(0, trt.float16)

    try:
        tw.build([identity_layer.get_output(0)])
        tw.runtime = trt.Runtime(tw.logger)
        tw.engine = tw.runtime.deserialize_cuda_engine(tw.engine_bytes)
        print_engine_io_information(engine=tw.engine)
    except Exception as e:
        print(f"strongly_typed={b_strongly_typed}, error={e}")

@case_mark
def case_strongly_typed_correct_way():

    tw = TRTWrapperV1()
    tw.network = tw.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    x = tw.network.add_input("x", trt.float32, (1, 4))
    cast_layer = tw.network.add_cast(x, trt.float16)
    identity_layer = tw.network.add_identity(cast_layer.get_output(0))

    tw.build([identity_layer.get_output(0)])
    tw.runtime = trt.Runtime(tw.logger)
    tw.engine = tw.runtime.deserialize_cuda_engine(tw.engine_bytes)
    print_engine_io_information(engine=tw.engine)

if __name__ == "__main__":

    # Old style without enabling strongly-typed.
    case_combination(False, False)
    case_combination(False, True)

    # Enable strongly-typed but not set-output-type. The output dtype is inferred as FLOAT and cannot be changed to FP16.
    # [TRT] [E] IBuilder::buildSerializedNetwork: Error Code 3: API Usage Error (Parameter check failed, condition: !config.getFlag(BuilderFlag::kFP16).  In createNetworkBuildConfig at /_src/optimizer/api/builder.cpp:893)
    case_combination(True, False)

    # Enable strongly-typed and set-output-type
    # [TRT] [E] ILayer::setOutputType: Error Code 3: API Usage Error (Parameter check failed, condition: !mNetwork->usingStronglyTyped(). INetworkLayer::setOutputType cannot be called for a strongly typed network. In setOutputType at /_src/optimizer/api/network.cpp:902)
    # [TRT] [E] IBuilder::buildSerializedNetwork: Error Code 3: API Usage Error (Parameter check failed, condition: !config.getFlag(BuilderFlag::kFP16).  In createNetworkBuildConfig at /_src/optimizer/api/builder.cpp:893)
    case_combination(True, True)

    # Correct way: enable strongly-typed and use a Cast layer to cast to FP16
    case_strongly_typed_correct_way()

    print("Finish")
