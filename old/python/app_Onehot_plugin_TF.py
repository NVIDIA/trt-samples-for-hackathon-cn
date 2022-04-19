#!/usr/bin/python3

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

from datetime import datetime
import numpy as np
import os, sys

i_gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import tensorflow.compat.v1 as tf
import tf2onnx
import ctypes
import onnx_graphsurgeon as gs
import onnx

ctypes.cdll.LoadLibrary('../build/OnehotPlugin.so')
#TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def modify_onehot(graph):
    for node in graph.nodes:
        if node.op == "OneHot":
            depth = node.inputs[1].values
            attrs = {"depth": int(depth)}
            onehot = gs.Node(op="OnehotPlugin", name=node.name, attrs=attrs)
            graph.nodes.append(onehot)

            inp_output_tensor = node.inputs[0]
            inp_output_tensor.outputs = [onehot]
            onehot.outputs = node.outputs
            node.outputs.clear()
            print(onehot)

    # Remove the non-used node from the graph completely
    graph.cleanup()
    return graph

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        # size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size,
                          bindings=bindings,
                          stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def main():
    tf.set_random_seed(1234)
    np.random.seed(0)
    iterations = 100
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        batch_size = 16
        input_data = np.random.rand(batch_size, 256).astype(np.float32)
        input_ph = tf.placeholder(dtype=tf.float32,
                                  shape=[batch_size, 256],
                                  name="input")

        x = tf.layers.dense(input_ph, 256)

        # test one_hot
        depth = 256
        indices = tf.cast(
            tf.clip_by_value(tf.reshape(x, [-1]), 0, depth - 1),
            tf.int32)
        x = tf.one_hot(indices, depth)
        x = tf.reshape(x, [batch_size, -1])
        x = tf.layers.dense(x, 256)

        output = tf.identity(x, name="output")
        sess.run(tf.global_variables_initializer())

        time_sum = 0
        a = datetime.now()
        for i in range(iterations):
            tf_result = sess.run([output], {input_ph: input_data})
        b = datetime.now()
        time_sum = (b - a).total_seconds()
        tf_time = "[INFO] TF  execution time " + str(
            time_sum * 1000 / iterations) + " ms"
        print(tf_time)

        output_name_without_port = ["output"]
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_name_without_port)
        # save frozen model
        with open("test_op.pb", "wb") as ofile:
            ofile.write(frozen_graph.SerializeToString())

    model_file = "test_op.onnx"
    os.system("python3 -m tf2onnx.convert --input test_op.pb --inputs input:0 --outputs output:0 --output test_op.onnx --verbose --opset 11")

    ### use ONNX GraphSurgeon
    # ONNX operator is required to keep aligned (like name, inputs, outputs and attributes) with TensorRT plugin to use Fallback mechanism.
    # ONNX GraphSurgeon is useful for modification and you can install it by the following commands.
    # pip install nvidia-pyindex
    # pip install onnx-graphsurgeon
    graph = gs.import_onnx(onnx.load(model_file))
    graph = modify_onehot(graph)
    model_file = "test_op_onehot.onnx"
    onnx.save(gs.export_onnx(graph), model_file)

    # build trt model by onnx model
    cuda.Device(0).make_context()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = batch_size

        with open(model_file, 'rb') as model:
            # parse onnx model
            parser.parse(model.read())
            for i in range(parser.num_errors):
                print(parser.get_error(i))

        engine = builder.build_engine(network, builder.create_builder_config())
        if engine == None:
            print("[ERROR] engine is None")
            exit(-1)
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        with engine.create_execution_context() as context:
            input_data = input_data.ravel()
            np.copyto(inputs[0].host, input_data)

            time_sum = 0
            a = datetime.now()
            for i in range(iterations):
                np.copyto(inputs[0].host, input_data)
                output = do_inference(context,
                                      bindings=bindings,
                                      inputs=inputs,
                                      outputs=outputs,
                                      stream=stream,
                                      batch_size=batch_size)
            b = datetime.now()
            time_sum = (b - a).total_seconds()
            trt_time = ("TRT execution time " +
                        str(time_sum * 1000 / iterations) + " ms")
            trt_result = output

    for i in range(len(trt_result)):
        print(
            "trt cross_check output_%d " % i +
            str(np.allclose(tf_result[i].flatten(), trt_result[i], atol=1e-5)))
        print("max diff " +
              str(np.fabs(tf_result[i].flatten() - trt_result[i]).max()))
        print("min diff " +
              str(np.fabs(tf_result[i].flatten() - trt_result[i]).min()))

    print(tf_time)
    print(trt_time)

    cuda.Context.pop()


if __name__ == '__main__':
    main()
