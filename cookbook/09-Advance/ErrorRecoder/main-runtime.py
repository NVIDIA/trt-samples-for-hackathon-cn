#
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
        if index < 0 or index >= self.nError:
            print("Error index!")
            return trt.ErrorCodeTRT.SUCCESS
        return self.errorList[index][0]

    def get_error_desc(self, index):
        if index < 0 or index >= self.nError:
            print("Error index!")
            return ""
        return self.errorList[index][1]

    def has_overflowed(self):
        if self.nError >= self.nMaxError:
            print("Error recorder overflowed!")
            return True
        return False

    def num_errors(self):
        return self.nError

    def report_error(self, errorCode, errorDescription):
        print("[MyErrorRecorder::report_error]\n\tNumber=%d,Code=%d,Information=%s" % (self.nError, int(errorCode), errorDescription))
        self.nError += 1
        self.errorList.append([errorCode, errorDescription])
        if self.has_overflowed():
            print("Error Overflow!")
        return

    def helloWorld(self):  # 非必需 API，仅用于本范例展示作用
        return "Hello World!"

myErrorRecorder = MyErrorRecorder()

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
inputTensor = network.add_input("inputT0", trt.float32, [-1, -1, -1])
profile.set_shape(inputTensor.name, [1, 1, 1], [3, 4, 5], [6, 8, 10])
config.add_optimization_profile(profile)

identityLayer = network.add_identity(inputTensor)
network.mark_output(identityLayer.get_output(0))
engineString = builder.build_serialized_network(network, config)
runtime = trt.Runtime(logger)
runtime.error_recorder = myErrorRecorder # 用于运行期的 ErrorRecorder，可以交给 Runtime 或 Engine 或 ExecutionContext
engine = runtime.deserialize_cuda_engine(engineString)
#engine.error_recorder = myErrorRecorder
context = engine.create_execution_context()
#context.error_recorder = myErrorRecorder

print("Runtime.error_recorder:", runtime.error_recorder, runtime.error_recorder.helloWorld())
print("Engine.error_recorder:", engine.error_recorder, engine.error_recorder.helloWorld())
print("Context.error_recorder:", context.error_recorder, context.error_recorder.helloWorld())

context.execute_v2([int(0), int(0)])  # 凭空进行推理，产生一个 binding 相关的运行期错误

print("Failed doing inference!")
print("Report error after all other work ---------------------------------------")
print("There is %d error" % myErrorRecorder.num_errors())
for i in range(myErrorRecorder.num_errors()):
    print("\tNumber=%d,Code=%d,Information=%s" % (i, int(myErrorRecorder.get_error_code(i)), myErrorRecorder.get_error_desc(i)))
myErrorRecorder.clear()  # 清除所有错误记录
