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

import tensorrt as trt

class MyLogger(trt.ILogger):  # customerized Logger

    def __init__(self):
        trt.ILogger.__init__(self)

    def log(self, severity, msg):
        if severity <= self.min_severity:
            # int(trt.ILogger.Severity.VERBOSE) == 4
            # int(trt.ILogger.Severity.INFO) == 3
            # int(trt.ILogger.Severity.WARNING) == 2
            # int(trt.ILogger.Severity.ERROR) == 1
            # int(trt.ILogger.Severity.INTERNAL_ERROR) == 0
            print("My Logger[%s] %s" % (severity, msg))  # customerized log content

logger = MyLogger()  # default severity is VERBOSE

print("Build time --------------------------------------------------------------")
logger.min_severity = trt.ILogger.Severity.INFO  # use severity INFO in build time
builder = trt.Builder(logger)  # assign logger to Builder
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
inputTensor = network.add_input("inputT0", trt.float32, [3, 4, 5])
identityLayer = network.add_identity(inputTensor)
network.mark_output(identityLayer.get_output(0))
engineString = builder.build_serialized_network(network, config)

print("Run time ----------------------------------------------------------------")
logger.min_severity = trt.ILogger.Severity.VERBOSE  # change severity into VERBOSE in run time

engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)  # assign logger to Runtime