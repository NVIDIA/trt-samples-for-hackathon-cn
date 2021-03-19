#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from functools import reduce
import tensorrt
import pycuda.driver as cuda
import numpy as np

class TrtLite:
    def __init__(self, build_engine_proc = None, build_engine_params = None, engine_file_path = None):
        logger = tensorrt.Logger(tensorrt.Logger.INFO)
        if engine_file_path is None:
            with tensorrt.Builder(logger) as builder:
                if build_engine_params is not None:
                    self.engine = build_engine_proc(builder, *build_engine_params)
                else:
                    self.engine = build_engine_proc(builder)
        else:
            with open(engine_file_path, 'rb') as f, tensorrt.Runtime(logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
    def __del__(self):
        self.engine = None
        self.context = None
    
    def save_to_file(self, engine_file_path):
        with open(engine_file_path, 'wb') as f:
            f.write(self.engine.serialize())
    
    def get_io_info(self, input_desc):
        def to_numpy_dtype(trt_dtype):
            tb = {
                tensorrt.DataType.BOOL: np.dtype('bool'),
                tensorrt.DataType.FLOAT: np.dtype('float32'),
                tensorrt.DataType.HALF: np.dtype('float16'),
                tensorrt.DataType.INT32: np.dtype('int32'),
                tensorrt.DataType.INT8: np.dtype('int8'),
            }
            return tb[trt_dtype]

        if isinstance(input_desc, dict):
            if self.engine.has_implicit_batch_dimension:
                print('Engine was built with static-shaped input so you should provide batch_size instead of i2shape')
                return
            i2shape = input_desc
            for i, shape in i2shape.items():
                self.context.set_binding_shape(i, shape)
            return [(self.engine.get_binding_name(i), self.engine.binding_is_input(i), 
                self.context.get_binding_shape(i), to_numpy_dtype(self.engine.get_binding_dtype(i))) for i in range(self.engine.num_bindings)]
        
        batch_size = input_desc
        return [(self.engine.get_binding_name(i), self.engine.binding_is_input(i), 
            (batch_size,) + tuple(self.context.get_binding_shape(i)), to_numpy_dtype(self.engine.get_binding_dtype(i))) for i in range(self.engine.num_bindings)]
    
    def allocate_io_buffers(self, input_desc, on_gpu):
        io_info = self.get_io_info(input_desc)
        if io_info is None:
            return
        if on_gpu:
            return [cuda.mem_alloc(reduce(lambda x, y: x * y, i[2]) * i[3].itemsize) for i in io_info]
        else:
            return [np.zeros(i[2], i[3]) for i in io_info]

    def execute(self, bindings, input_desc, stream_handle = 0, input_consumed = None):
        if isinstance(input_desc, dict):
            i2shape = input_desc
            for i, shape in i2shape.items():
                self.context.set_binding_shape(i, shape)
            self.context.execute_async_v2(bindings, stream_handle, input_consumed)
            return
        
        batch_size = input_desc
        self.context.execute_async(batch_size, bindings, stream_handle, input_consumed)

    def print_info(self):
        print("Batch dimension is", "implicit" if self.engine.has_implicit_batch_dimension else "explicit")
        for i in range(self.engine.num_bindings):
            print("input" if self.engine.binding_is_input(i) else "output", 
                  self.engine.get_binding_name(i), self.engine.get_binding_dtype(i), 
                  self.engine.get_binding_shape(i), 
                  -1 if -1 in self.engine.get_binding_shape(i) else reduce(
                      lambda x, y: x * y, self.engine.get_binding_shape(i)) * self.engine.get_binding_dtype(i).itemsize)
