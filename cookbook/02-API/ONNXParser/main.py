#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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
#

import os

import numpy as np
import tensorrt as trt

shape = [1, 1, 28, 28]
onnxFile = "./model.onnx"

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()

os.chdir("/w/gitlab/tensorrt-cookbook/02-API/ONNXParser")

parser = trt.OnnxParser(network, logger)

# check whether one certain operator is supported by ONNX parser
print("parser.supports_operator('LayerNormalization') = %s" % parser.supports_operator("LayerNormalization"))

# ?
print("parser.sget_used_vc_plugin_libraries() = %s" % parser.get_used_vc_plugin_libraries())

if True:  # four equivalent methods to parse ONNX file
    res = parser.parse_from_file(onnxFile)  # parse from file
else:
    with open(onnxFile, "rb") as model:
        modelString = model.read()
        # three equivalent methods to parse ONNX byte stream, but supports_model can provide more information
        # both method parse and supports_model have an optional input parameter "path" pointing to the directory of the seperated weights files (used when ONNX file is larger than 2 GiB)
        res = parser.parse(modelString)  
        #res, information = parser.supports_model(modelString)  # In my opinion, supports_model should just tell me the result (True / False) without parsing the model into network
        #print(information)
        #res = parser.parse_with_weight_descriptors(modelString)

if not res:
    print("Failed parsing %s" % onnxFile)   
    for i in range(parser.num_errors):  # get error information
        error = parser.get_error(i)
        print(error)  # print error information 
        print("node=%s" % error.node())
        print("code=%s" % error.code())
        print("desc=%s" % error.desc())
        print("file=%s" % error.file())
        print("func=%s" % error.func())
        print("line=%s" % error.line())
        
    parser.clear_errors()  # clean the error information, not required
    exit()
print("Succeeded parsing %s" % onnxFile)

inputTensor = network.get_input(0)
profile.set_shape(inputTensor.name, [1] + shape[1:], [2] + shape[1:], [4] + shape[1:])
config.add_optimization_profile(profile)

engineString = builder.build_serialized_network(network, config)
print("%s building serialized network" % ("Failed" if engineString is None else "Succeeded")) 