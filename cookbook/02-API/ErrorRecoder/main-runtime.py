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

trtFile = "./model.plan"

class MyErrorRecorder(trt.IErrorRecorder):

    def __init__(self):
        super(MyErrorRecorder, self).__init__()
        self.errorList = []
        self.nError = 0
        self.nMaxError = 256

    def clear(self):
        print("[MyErrorRecorder::clear]")
        self.nError = []
        self.nError = 0
        return None

    def get_error_code(self, index):
        print("[MyErrorRecorder::get_error_code]")
        if index < 0 or index >= self.nError:
            print("Error index")
            return trt.ErrorCodeTRT.SUCCESS
        return self.errorList[index][0]

    def get_error_desc(self, index):
        print("[MyErrorRecorder::get_error_desc]")
        if index < 0 or index >= self.nError:
            print("Error index")
            return ""
        # Error number in self.errorList[index][0]:
        # trt.ErrorCodeTRT.SUCCESS  # 0
        # trt.ErrorCodeTRT.UNSPECIFIED_ERROR  # 1
        # trt.ErrorCodeTRT.INTERNAL_ERROR  # 2
        # trt.ErrorCodeTRT.INVALID_ARGUMENT  # 3
        # trt.ErrorCodeTRT.INVALID_CONFIG  # 4
        # trt.ErrorCodeTRT.FAILED_ALLOCATION  # 5
        # trt.ErrorCodeTRT.FAILED_INITIALIZATION  # 6
        # trt.ErrorCodeTRT.FAILED_EXECUTION  # 7
        # trt.ErrorCodeTRT.FAILED_COMPUTATION  # 8
        # trt.ErrorCodeTRT.INVALID_STATE  # 9
        # trt.ErrorCodeTRT.UNSUPPORTED_STATE  # 10
        return self.errorList[index][1]

    def has_overflowed(self):
        print("[MyErrorRecorder::has_overflowed]")
        if self.nError >= self.nMaxError:
            print("Error recorder overflowed!")
            return True
        return False

    def num_errors(self):
        print("[MyErrorRecorder::num_errors]")
        return self.nError

    def report_error(self, errorCode, errorDescription):
        print("[MyErrorRecorder::report_error]\n\tNumber=%d,Code=%d,Information=%s" % (self.nError, int(errorCode), errorDescription))
        self.nError += 1
        self.errorList.append([errorCode, errorDescription])
        if self.has_overflowed():
            print("Error Overflow!")
        return

    def helloWorld(self):  # not required API, just for fun
        return "Hello World!"

myErrorRecorder = MyErrorRecorder()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
inputTensor = network.add_input("inputT0", trt.float32, [-1, -1, -1])
profile.set_shape(inputTensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
config.add_optimization_profile(profile)

identityLayer = network.add_identity(inputTensor)
network.mark_output(identityLayer.get_output(0))
engineString = builder.build_serialized_network(network, config)
runtime = trt.Runtime(logger)
runtime.error_recorder = myErrorRecorder  # ErrorRecorder for runtime, it can be assigned to Runtime or Engine or ExecutionContext
engine = runtime.deserialize_cuda_engine(engineString)
#engine.error_recorder = myErrorRecorder
context = engine.create_execution_context()
#context.error_recorder = myErrorRecorder

print("Runtime.error_recorder:", runtime.error_recorder, runtime.error_recorder.helloWorld())
print("Engine.error_recorder:", engine.error_recorder, engine.error_recorder.helloWorld())
print("Context.error_recorder:", context.error_recorder, context.error_recorder.helloWorld())

context.execute_v2([int(0), int(0)])  # use null pointer to do inference, TensorRT raises a error

print("Failed doing inference!")
print("Report error after all other work ---------------------------------------")
print("There is %d error" % myErrorRecorder.num_errors())
for i in range(myErrorRecorder.num_errors()):
    print("\tNumber=%d,Code=%d,Information=%s" % (i, int(myErrorRecorder.get_error_code(i)), myErrorRecorder.get_error_desc(i)))
myErrorRecorder.clear()  # clear all error information
