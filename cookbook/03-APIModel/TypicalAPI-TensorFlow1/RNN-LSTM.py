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
from datetime import datetime as dt
from cuda import cudart
import tensorrt as trt

os.environ["TF_ENABLE_DEPRECATION_WARNINGS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
np.random.seed(31193)
tf.compat.v1.set_random_seed(97)
epsilon = 1e-6
nBatchSize, nSequenceLength, nInputDim, nHiddenDim = 2, 4, 7, 5
inputX = np.random.rand(nBatchSize, nSequenceLength, nInputDim).astype(np.float32).reshape([nBatchSize, nSequenceLength, nInputDim])
inputH = np.random.rand(nBatchSize, nHiddenDim).astype(np.float32).reshape([nBatchSize, nHiddenDim])
inputC = np.random.rand(nBatchSize, nHiddenDim).astype(np.float32).reshape([nBatchSize, nHiddenDim])

def check(a, b, weak=False, info=""):
    if weak:
        res = np.all(np.abs(a - b) < epsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + epsilon))
    print("check %s:" % info, res, diff0, diff1)

def printArrayInfomation(x, info="", n=5):
    print( "%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f"%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print("\t", x.reshape(-1)[:n], x.reshape(-1)[-n:])

# for debug
def smallTest(x0, h0, c0):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    para = np.load("test?.npz")
    weight = [np.split(i, [nInputDim], axis=0) for i in np.split(para["?/kernel:0"], 4, axis=1)]
    bias = np.split(para["?/bias:0"], 4)
    h, c = h0, c0
    for t in range(nSequenceLength):
        x = x0[:, t, :]
        it = sigmoid(np.matmul(x, weight[0][0]) + np.matmul(h, weight[0][1]) + bias[0])
        ct_ = np.tanh(np.matmul(x, weight[1][0]) + np.matmul(h, weight[1][1]) + bias[1])
        ft = sigmoid(np.matmul(x, weight[2][0]) + np.matmul(h, weight[2][1]) + bias[2])
        ot = sigmoid(np.matmul(x, weight[3][0]) + np.matmul(h, weight[3][1]) + bias[3])
        ct = ft * c0 + it * ct_
        ht = ot * np.tanh(ct)
        print("ht=\n", ht, "\nct=\n", ct)
        h = ht
        c = ct

    print("here")
    return

def test1():
    print("\ntf.keras.layers.LSTM or tf.keras.layers.LSTMCell + tf.keras.layers.RNN")
    # TensorFlow part ----------------------------------------------------------
    x = tf.compat.v1.placeholder(tf.float32, [None, nSequenceLength, nInputDim], name="x")
    h0 = tf.compat.v1.placeholder(tf.float32, [None, nHiddenDim], name="h0")
    c0 = tf.compat.v1.placeholder(tf.float32, [None, nHiddenDim], name="c0")
    # Two equivalent realization
    if True:  # tf.keras.layers.LSTM
        lstm    = tf.compat.v1.keras.layers.LSTM( \
                    nHiddenDim,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    use_bias=True,
                    kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1),
                    recurrent_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1),
                    bias_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1),
                    unit_forget_bias=False,
                    kernel_regularizer=None,
                    recurrent_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    recurrent_constraint=None,
                    bias_constraint=None,
                    dropout=0.0,
                    recurrent_dropout=0.0,
                    implementation=1,
                    return_sequences=True,
                    return_state=True,
                    go_backwards=False,
                    stateful=False,
                    unroll=False,
                    time_major=False
                    )
    else:  # tf.keras.layers.LSTMCell + tf.keras.layers.RNN
        cell    = tf.keras.layers.LSTMCell( \
                    nHiddenDim,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    use_bias=True,
                    kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1),
                    recurrent_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1),
                    bias_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1),
                    unit_forget_bias=False,
                    kernel_regularizer=None,
                    recurrent_regularizer=None,
                    bias_regularizer=None,
                    kernel_constraint=None,
                    recurrent_constraint=None,
                    bias_constraint=None,
                    dropout=0.0,
                    recurrent_dropout=0.0,
                    )
        lstm    = tf.keras.layers.RNN( \
                    cell,
                    return_sequences=True,
                    return_state=True,
                    go_backwards=False,
                    stateful=False,
                    unroll=False,
                    time_major=False
                    )
    y, h1, c1 = lstm(x, initial_state=[h0, c0])

    tfConfig = tf.compat.v1.ConfigProto()
    tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.compat.v1.Session(config=tfConfig)
    sess.run(tf.compat.v1.global_variables_initializer())
    outputTF, outputTFh1, outputTFc1 = sess.run([y, h1, c1], feed_dict={x: inputX, h0: inputH, c0: inputC})

    tfPara = {}
    print("Weight:")
    for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
        name, value = i.name, sess.run(i)
        print(name, value.shape)
        tfPara[name] = value
    np.savez("test1.npz", **tfPara)
    sess.close()

    # TensorRT part ------------------------------------------------------------
    # Two equivalent realization
    if True:  # use Loop Structure, Dynamic Shape mode is supported but Refit is not supported
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()

        inputT0 = network.add_input("inputT0", trt.float32, (-1, -1, nInputDim))
        inputT1 = network.add_input("inputT1", trt.float32, (-1, nHiddenDim))
        inputT2 = network.add_input("inputT2", trt.float32, (-1, nHiddenDim))
        profile.set_shape(inputT0.name, [1, 1, nInputDim], [nBatchSize, nSequenceLength, nInputDim], [nBatchSize * 2, nSequenceLength * 2, nInputDim])
        profile.set_shape(inputT1.name, [1, nHiddenDim], [nBatchSize, nHiddenDim], [nBatchSize * 2, nHiddenDim])
        profile.set_shape(inputT2.name, [1, nHiddenDim], [nBatchSize, nHiddenDim], [nBatchSize * 2, nHiddenDim])
        config.add_optimization_profile(profile)

        para = np.load("test1.npz")
        weightXLayerList = [network.add_constant([nInputDim, nHiddenDim], np.ascontiguousarray(i.reshape(-1))) for i in np.split(para["lstm/kernel:0"], 4, axis=1)]
        weightHLayerList = [network.add_constant([nHiddenDim, nHiddenDim], np.ascontiguousarray(i.reshape(-1))) for i in np.split(para["lstm/recurrent_kernel:0"], 4, axis=1)]
        biasLayerList = [network.add_constant([1, nHiddenDim], np.ascontiguousarray(i.reshape(-1))) for i in np.split(para["lstm/bias:0"], 4)]

        loop = network.add_loop()

        def gate(network, xTensor, wx, hTensor, wh, b, isSigmoid):
            _h0 = network.add_matrix_multiply(xTensor, trt.MatrixOperation.NONE, wx, trt.MatrixOperation.NONE)
            _h1 = network.add_matrix_multiply(hTensor, trt.MatrixOperation.NONE, wh, trt.MatrixOperation.NONE)
            _h2 = network.add_elementwise(_h0.get_output(0), _h1.get_output(0), trt.ElementWiseOperation.SUM)
            _h3 = network.add_elementwise(_h2.get_output(0), b, trt.ElementWiseOperation.SUM)
            _h4 = network.add_activation(_h3.get_output(0), trt.ActivationType.SIGMOID if isSigmoid else trt.ActivationType.TANH)
            return _h4

        _t0 = network.add_shape(inputT0)
        _t1 = network.add_slice(_t0.get_output(0), [1], [1], [1])
        _t2 = network.add_shuffle(_t1.get_output(0))
        _t2.reshape_dims = ()
        loop.add_trip_limit(_t2.get_output(0), trt.TripLimit.COUNT)
        iteratorLayer = loop.add_iterator(inputT0, 1, False)  # iterator throws one piece of inputT0 each time, shape: [nBatchSize, nInputDim]
        hiddenStateLayer = loop.add_recurrence(inputT1)  # initial hidden state and cell state
        cellStateLayer = loop.add_recurrence(inputT2)

        gateI = gate(network, iteratorLayer.get_output(0), weightXLayerList[0].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[0].get_output(0), biasLayerList[0].get_output(0), True)
        gateF = gate(network, iteratorLayer.get_output(0), weightXLayerList[1].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[1].get_output(0), biasLayerList[1].get_output(0), True)
        gateC = gate(network, iteratorLayer.get_output(0), weightXLayerList[2].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[2].get_output(0), biasLayerList[2].get_output(0), False)
        gateO = gate(network, iteratorLayer.get_output(0), weightXLayerList[3].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[3].get_output(0), biasLayerList[3].get_output(0), True)

        _h5 = network.add_elementwise(gateF.get_output(0), cellStateLayer.get_output(0), trt.ElementWiseOperation.PROD)
        _h6 = network.add_elementwise(gateI.get_output(0), gateC.get_output(0), trt.ElementWiseOperation.PROD)
        newCellStateLayer = network.add_elementwise(_h5.get_output(0), _h6.get_output(0), trt.ElementWiseOperation.SUM)
        _h7 = network.add_activation(newCellStateLayer.get_output(0), trt.ActivationType.TANH)
        newHiddenStateLayer = network.add_elementwise(gateO.get_output(0), _h7.get_output(0), trt.ElementWiseOperation.PROD)

        hiddenStateLayer.set_input(1, newHiddenStateLayer.get_output(0))
        cellStateLayer.set_input(1, newCellStateLayer.get_output(0))

        loopOutput0 = loop.add_loop_output(hiddenStateLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # output final hidden state, shape: [nBatchSize,nHiddenSize]
        loopOutput1 = loop.add_loop_output(newHiddenStateLayer.get_output(0), trt.LoopOutput.CONCATENATE, 1)  # output all hidden state, shape: [nBatchSize,nSequenceLength,nHiddenSize]
        loopOutput1.set_input(1, _t2.get_output(0))
        loopOutput2 = loop.add_loop_output(cellStateLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # output final cell state, shape: [nBatchSize,nHiddenSize]

        network.mark_output(loopOutput0.get_output(0))
        network.mark_output(loopOutput1.get_output(0))
        network.mark_output(loopOutput2.get_output(0))
        engineString = builder.build_serialized_network(network, config)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
        context = engine.create_execution_context()
        context.set_input_shape(engine.get_tensor_name(0), [nBatchSize, nSequenceLength, nInputDim])
        context.set_input_shape(engine.get_tensor_name(1), [nBatchSize, nHiddenDim])
        context.set_input_shape(engine.get_tensor_name(2), [nBatchSize, nHiddenDim])
        _, stream = cudart.cudaStreamCreate()

        inputH0 = np.ascontiguousarray(inputX.reshape(-1))
        inputH1 = np.ascontiguousarray(inputH.reshape(-1))
        inputH2 = np.ascontiguousarray(inputC.reshape(-1))
        outputH0 = np.empty(context.get_binding_shape(3), dtype=trt.nptype(engine.get_binding_dtype(3)))
        outputH1 = np.empty(context.get_binding_shape(4), dtype=trt.nptype(engine.get_binding_dtype(4)))
        outputH2 = np.empty(context.get_binding_shape(5), dtype=trt.nptype(engine.get_binding_dtype(5)))
        _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
        _, inputD1 = cudart.cudaMallocAsync(inputH1.nbytes, stream)
        _, inputD2 = cudart.cudaMallocAsync(inputH2.nbytes, stream)
        _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)
        _, outputD1 = cudart.cudaMallocAsync(outputH1.nbytes, stream)
        _, outputD2 = cudart.cudaMallocAsync(outputH2.nbytes, stream)

        cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        cudart.cudaMemcpyAsync(inputD1, inputH1.ctypes.data, inputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        cudart.cudaMemcpyAsync(inputD2, inputH2.ctypes.data, inputH2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        context.execute_async_v2([int(inputD0), int(inputD1), int(inputD2), int(outputD0), int(outputD1), int(outputD2)], stream)

        cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaMemcpyAsync(outputH1.ctypes.data, outputD1, outputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaMemcpyAsync(outputH2.ctypes.data, outputD2, outputH2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaStreamSynchronize(stream)

        #printArrayInfomation(inputX,"x")
        #print(inputX)
        #printArrayInfomation(inputH,"h0")
        #print(inputH)
        #printArrayInfomation(inputC,"c0")
        #print(inputC)
        #printArrayInfomation(outputTFh1,"TF h1")
        #printArrayInfomation(outputH0,"TRT h1")
        #printArrayInfomation(outputTFc1,"TF c1")
        #printArrayInfomation(outputH2,"TRT c1")
        #printArrayInfomation(outputTF,"TF AllOutput")
        #printArrayInfomation(outputH1,"TRT AllOutput")
        check(outputTFh1, outputH0, True, "h1")
        check(outputTFc1, outputH2, True, "c1")
        check(outputTF, outputH1, True, "AllOutput")

        cudart.cudaStreamDestroy(stream)
        cudart.cudaFree(inputD0)
        cudart.cudaFree(inputD1)
        cudart.cudaFree(inputD2)
        cudart.cudaFree(outputD0)
        cudart.cudaFree(outputD1)
        cudart.cudaFree(outputD2)

    else:  # use RNNV2 layer, Dynamic Shape mode is not supported, deprecated since TensorRT 8.5
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
                inputT0 = network.add_input("inputT0", trt.float32, (nBatchSize, nSequenceLength, nInputDim))
        inputT1 = network.add_input("inputT1", trt.float32, (nBatchSize, 1, nHiddenDim))
        inputT2 = network.add_input("inputT2", trt.float32, (nBatchSize, 1, nHiddenDim))

        rnnV2Layer = network.add_rnn_v2(inputT0, 1, nHiddenDim, nSequenceLength, trt.RNNOperation.LSTM)
        rnnV2Layer.direction = trt.RNNDirection.UNIDIRECTION
        rnnV2Layer.input_mode = trt.RNNInputMode.LINEAR
        rnnV2Layer.hidden_state = inputT1
        rnnV2Layer.cell_state = inputT2

        gateList = [trt.RNNGateType.INPUT, trt.RNNGateType.FORGET, trt.RNNGateType.CELL, trt.RNNGateType.OUTPUT]
        para = np.load("test1.npz")
        weightXList = [i.transpose().reshape(-1) for i in np.split(para["lstm/kernel:0"], 4, axis=1)]
        weightHList = [i.transpose().reshape(-1) for i in np.split(para["lstm/recurrent_kernel:0"], 4, axis=1)]
        biasList = [i.reshape(-1) for i in np.split(para["lstm/bias:0"], 4)]
        for gate, weightX, weightH, bias in zip(gateList, weightXList, weightHList, biasList):
            rnnV2Layer.set_weights_for_gate(0, gate, True, weightX)
            rnnV2Layer.set_weights_for_gate(0, gate, False, weightH)
            rnnV2Layer.set_bias_for_gate(0, gate, True, bias)
            rnnV2Layer.set_bias_for_gate(0, gate, False, np.zeros([nHiddenDim], dtype=np.float32))

        network.mark_output(rnnV2Layer.get_output(0))
        network.mark_output(rnnV2Layer.get_output(1))
        network.mark_output(rnnV2Layer.get_output(2))
        engineString = builder.build_serialized_network(network, config)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
        context = engine.create_execution_context()
        _, stream = cudart.cudaStreamCreate()

        inputH0 = np.ascontiguousarray(inputX.reshape(-1))
        inputH1 = np.ascontiguousarray(inputH.reshape(-1))
        inputH2 = np.ascontiguousarray(inputC.reshape(-1))
        outputH0 = np.empty(context.get_binding_shape(3), dtype=trt.nptype(engine.get_binding_dtype(3)))
        outputH1 = np.empty(context.get_binding_shape(4), dtype=trt.nptype(engine.get_binding_dtype(4)))
        outputH2 = np.empty(context.get_binding_shape(5), dtype=trt.nptype(engine.get_binding_dtype(5)))
        _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
        _, inputD1 = cudart.cudaMallocAsync(inputH1.nbytes, stream)
        _, inputD2 = cudart.cudaMallocAsync(inputH2.nbytes, stream)
        _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)
        _, outputD1 = cudart.cudaMallocAsync(outputH1.nbytes, stream)
        _, outputD2 = cudart.cudaMallocAsync(outputH2.nbytes, stream)

        cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        cudart.cudaMemcpyAsync(inputD1, inputH1.ctypes.data, inputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        cudart.cudaMemcpyAsync(inputD2, inputH2.ctypes.data, inputH2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        context.execute_async_v2([int(inputD0), int(inputD1), int(inputD2), int(outputD0), int(outputD1), int(outputD2)], stream)
        cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaMemcpyAsync(outputH1.ctypes.data, outputD1, outputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaMemcpyAsync(outputH2.ctypes.data, outputD2, outputH2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaStreamSynchronize(stream)

        #print("inputH0 :", inputX.shape)
        #print(inputX)
        #printArrayInfomation(outputTF,"TF AllOutput")
        #printArrayInfomation(outputH0,"TRT AllOutput")
        #printArrayInfomation(outputTFh1,"TF h1")
        #printArrayInfomation(outputH1,"TRT h1")
        #printArrayInfomation(outputTFc1,"TF c1")
        #printArrayInfomation(outputH2,"TRT c1")
        check(outputTF, outputH0, True, "AllOutput")
        check(outputTFh1, outputH1.reshape(nBatchSize, nHiddenDim), True, "h1")
        check(outputTFc1, outputH2.reshape(nBatchSize, nHiddenDim), True, "c1")

        cudart.cudaStreamDestroy(stream)
        cudart.cudaFree(inputD0)
        cudart.cudaFree(inputD1)
        cudart.cudaFree(inputD2)
        cudart.cudaFree(outputD0)
        cudart.cudaFree(outputD1)
        cudart.cudaFree(outputD2)

def test2():
    print("\ntf.keras.layers.CuDNNLSTM -----------------------------------------")
    # TensorFlow part ----------------------------------------------------------
    x = tf.compat.v1.placeholder(tf.float32, [None, nSequenceLength, nInputDim], name="x")
    h0 = tf.compat.v1.placeholder(tf.float32, [None, nHiddenDim], name="h0")
    c0 = tf.compat.v1.placeholder(tf.float32, [None, nHiddenDim], name="c0")
    lstm    = tf.compat.v1.keras.layers.CuDNNLSTM( \
                nHiddenDim,
                kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1),
                recurrent_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1),
                bias_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1),
                unit_forget_bias=True,
                kernel_regularizer=None,
                recurrent_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
                return_sequences=True,
                return_state=True,
                go_backwards=False,
                stateful=False,
                time_major=False
                )
    y, h1, c1 = lstm(x, initial_state=[h0, c0])

    tfConfig = tf.compat.v1.ConfigProto()
    tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.compat.v1.Session(config=tfConfig)
    sess.run(tf.compat.v1.global_variables_initializer())
    outputTF, outputTFh1, outputTFc1 = sess.run([y, h1, c1], feed_dict={x: inputX, h0: inputH, c0: inputC})

    tfPara = {}
    print("Weight:")
    for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
        name, value = i.name, sess.run(i)
        print(name, value.shape)
        tfPara[name] = value
    np.savez("test2.npz", **tfPara)
    sess.close()

    # TensorRT part ------------------------------------------------------------
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
        inputT0 = network.add_input("inputT0", trt.float32, (-1, -1, nInputDim))
    inputT1 = network.add_input("inputT1", trt.float32, (-1, nHiddenDim))
    inputT2 = network.add_input("inputT2", trt.float32, (-1, nHiddenDim))
    profile.set_shape(inputT0.name, [1, 1, nInputDim], [nBatchSize, nSequenceLength, nInputDim], [nBatchSize * 2, nSequenceLength * 2, nInputDim])
    profile.set_shape(inputT1.name, [1, nHiddenDim], [nBatchSize, nHiddenDim], [nBatchSize * 2, nHiddenDim])
    profile.set_shape(inputT2.name, [1, nHiddenDim], [nBatchSize, nHiddenDim], [nBatchSize * 2, nHiddenDim])
    config.add_optimization_profile(profile)

    para = np.load("test2.npz")  # the shape and order of the weights is different from keras.layers.LSTM
    weightXLayerList = [network.add_constant([nInputDim, nHiddenDim], i.reshape(nHiddenDim, nInputDim).transpose().reshape(-1)) for i in np.split(para["cu_dnnlstm/kernel:0"], 4, axis=1)]
    weightHLayerList = [network.add_constant([nHiddenDim, nHiddenDim], i.transpose().reshape(-1)) for i in np.split(para["cu_dnnlstm/recurrent_kernel:0"], 4, axis=1)]
    biasLayerList = [network.add_constant([1, nHiddenDim], i.reshape(-1)) for i in np.split(np.sum(para["cu_dnnlstm/bias:0"].reshape(2, -1), axis=0), 4)]

    loop = network.add_loop()

    def gate(network, xTensor, wx, hTensor, wh, b, isSigmoid):
        _h0 = network.add_matrix_multiply(xTensor, trt.MatrixOperation.NONE, wx, trt.MatrixOperation.NONE)
        _h1 = network.add_matrix_multiply(hTensor, trt.MatrixOperation.NONE, wh, trt.MatrixOperation.NONE)
        _h2 = network.add_elementwise(_h0.get_output(0), _h1.get_output(0), trt.ElementWiseOperation.SUM)
        _h3 = network.add_elementwise(_h2.get_output(0), b, trt.ElementWiseOperation.SUM)
        _h4 = network.add_activation(_h3.get_output(0), trt.ActivationType.SIGMOID if isSigmoid else trt.ActivationType.TANH)
        return _h4

    _t0 = network.add_shape(inputT0)
    _t1 = network.add_slice(_t0.get_output(0), [1], [1], [1])
    _t2 = network.add_shuffle(_t1.get_output(0))
    _t2.reshape_dims = ()
    loop.add_trip_limit(_t2.get_output(0), trt.TripLimit.COUNT)
    iteratorLayer = loop.add_iterator(inputT0, 1, False)  # iterator throws one piece of inputT0 each time, shape: [nBatchSize, nInputDim]
    hiddenStateLayer = loop.add_recurrence(inputT1)  # initial hidden state and cell state
    cellStateLayer = loop.add_recurrence(inputT2)

    gateI = gate(network, iteratorLayer.get_output(0), weightXLayerList[0].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[0].get_output(0), biasLayerList[0].get_output(0), True)
    gateF = gate(network, iteratorLayer.get_output(0), weightXLayerList[1].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[1].get_output(0), biasLayerList[1].get_output(0), True)
    gateC = gate(network, iteratorLayer.get_output(0), weightXLayerList[2].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[2].get_output(0), biasLayerList[2].get_output(0), False)
    gateO = gate(network, iteratorLayer.get_output(0), weightXLayerList[3].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[3].get_output(0), biasLayerList[3].get_output(0), True)

    _h5 = network.add_elementwise(gateF.get_output(0), cellStateLayer.get_output(0), trt.ElementWiseOperation.PROD)
    _h6 = network.add_elementwise(gateI.get_output(0), gateC.get_output(0), trt.ElementWiseOperation.PROD)
    newCellStateLayer = network.add_elementwise(_h5.get_output(0), _h6.get_output(0), trt.ElementWiseOperation.SUM)
    _h7 = network.add_activation(newCellStateLayer.get_output(0), trt.ActivationType.TANH)
    newHiddenStateLayer = network.add_elementwise(gateO.get_output(0), _h7.get_output(0), trt.ElementWiseOperation.PROD)

    hiddenStateLayer.set_input(1, newHiddenStateLayer.get_output(0))
    cellStateLayer.set_input(1, newCellStateLayer.get_output(0))

    loopOutput0 = loop.add_loop_output(hiddenStateLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # output final hidden state, shape: [nBatchSize,nHiddenSize]
    loopOutput1 = loop.add_loop_output(newHiddenStateLayer.get_output(0), trt.LoopOutput.CONCATENATE, 1)  # output all hidden state, shape: [nBatchSize,nSequenceLength,nHiddenSize]
    loopOutput1.set_input(1, _t2.get_output(0))
    loopOutput2 = loop.add_loop_output(cellStateLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # output final cell state, shape: [nBatchSize,nHiddenSize]

    network.mark_output(loopOutput0.get_output(0))
    network.mark_output(loopOutput1.get_output(0))
    network.mark_output(loopOutput2.get_output(0))
    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    context = engine.create_execution_context()
    context.set_input_shape(engine.get_tensor_name(0), [nBatchSize, nSequenceLength, nInputDim])
    context.set_input_shape(engine.get_tensor_name(1), [nBatchSize, nHiddenDim])
    context.set_input_shape(engine.get_tensor_name(2), [nBatchSize, nHiddenDim])
    _, stream = cudart.cudaStreamCreate()

    inputH0 = np.ascontiguousarray(inputX.reshape(-1))
    inputH1 = np.ascontiguousarray(inputH.reshape(-1))
    inputH2 = np.ascontiguousarray(inputC.reshape(-1))
    outputH0 = np.empty(context.get_binding_shape(3), dtype=trt.nptype(engine.get_binding_dtype(3)))
    outputH1 = np.empty(context.get_binding_shape(4), dtype=trt.nptype(engine.get_binding_dtype(4)))
    outputH2 = np.empty(context.get_binding_shape(5), dtype=trt.nptype(engine.get_binding_dtype(5)))
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
    _, inputD1 = cudart.cudaMallocAsync(inputH1.nbytes, stream)
    _, inputD2 = cudart.cudaMallocAsync(inputH2.nbytes, stream)
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)
    _, outputD1 = cudart.cudaMallocAsync(outputH1.nbytes, stream)
    _, outputD2 = cudart.cudaMallocAsync(outputH2.nbytes, stream)

    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    cudart.cudaMemcpyAsync(inputD1, inputH1.ctypes.data, inputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    cudart.cudaMemcpyAsync(inputD2, inputH2.ctypes.data, inputH2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    context.execute_async_v2([int(inputD0), int(inputD1), int(inputD2), int(outputD0), int(outputD1), int(outputD2)], stream)

    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaMemcpyAsync(outputH1.ctypes.data, outputD1, outputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaMemcpyAsync(outputH2.ctypes.data, outputD2, outputH2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    #printArrayInfomation(inputX,"input")
    #print(inputX)
    #printArrayInfomation(outputTFh1,"TF h1")
    #printArrayInfomation(outputH0,"TRT h1")
    #printArrayInfomation(outputTFc1,"TF c1")
    #printArrayInfomation(outputH2,"TRT c1")
    #printArrayInfomation(outputTF,"TF AllOutput")
    #printArrayInfomation(outputH1,"TRT AllOutput")
    check(outputTFh1, outputH0, True, "h1")
    check(outputTFc1, outputH2, True, "c1")
    check(outputTF, outputH1, True, "AllOutput")

    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(inputD1)
    cudart.cudaFree(inputD2)
    cudart.cudaFree(outputD0)
    cudart.cudaFree(outputD1)
    cudart.cudaFree(outputD2)

def test3():
    print("\n[tf.nn.rnn_cell.BasicLSTMCell / tf.contrib.rnn.BasicLSTMCell] + [tf.nn.static_rnn / tf.nn.dynamic_rnn]")
    forgetBias = 1.0
    # TensorFlow part ----------------------------------------------------------
    x = tf.compat.v1.placeholder(tf.float32, [None, nSequenceLength, nInputDim], name="x")
    h0 = tf.compat.v1.placeholder(tf.float32, [None, nHiddenDim], name="h0")
    c0 = tf.compat.v1.placeholder(tf.float32, [None, nHiddenDim], name="c0")
    # tf.nn.rnn_cell.BasicLSTMCell and tf.contrib.rnn.BasicLSTMCell are alias
    cell    = tf.nn.rnn_cell.BasicLSTMCell( \
    #cell    = tf.contrib.rnn.BasicLSTMCell( \
                nHiddenDim,
                forget_bias=forgetBias,
                state_is_tuple=True,
                activation=None,
                reuse=None,
                name=None,
                dtype=None
                )
    # Two equivalent realization
    if True:
        y,hc    = tf.nn.static_rnn( \
                    cell,
                    [ x[:,i,:] for i in range(nSequenceLength) ],
                    initial_state=[c0,h0],
                    dtype=None,
                    sequence_length=None,
                    scope=None
                    )
    else:
        y,hc    = tf.nn.dynamic_rnn( \
                    cell,
                    x,
                    sequence_length=None,
                    initial_state=[c0,h0],
                    dtype=None,
                    parallel_iterations=None,
                    swap_memory=False,
                    time_major=False,
                    scope=None
                    )

    tfConfig = tf.compat.v1.ConfigProto()
    tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.compat.v1.Session(config=tfConfig)
    sess.run(tf.compat.v1.global_variables_initializer())
    outputTF, outputTFhc = sess.run([y, hc], feed_dict={x: inputX, h0: inputH, c0: inputC})

    tfPara = {}
    print("Weight:")
    for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
        name, value = i.name, sess.run(i)
        print(name, value.shape)
        tfPara[name] = value
    np.savez("test3.npz", **tfPara)
    sess.close()

    # TensorRT part ------------------------------------------------------------
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
        inputT0 = network.add_input("inputT0", trt.float32, (-1, -1, nInputDim))
    inputT1 = network.add_input("inputT1", trt.float32, (-1, nHiddenDim))
    inputT2 = network.add_input("inputT2", trt.float32, (-1, nHiddenDim))
    profile.set_shape(inputT0.name, [1, 1, nInputDim], [nBatchSize, nSequenceLength, nInputDim], [nBatchSize * 2, nSequenceLength * 2, nInputDim])
    profile.set_shape(inputT1.name, [1, nHiddenDim], [nBatchSize, nHiddenDim], [nBatchSize * 2, nHiddenDim])
    profile.set_shape(inputT2.name, [1, nHiddenDim], [nBatchSize, nHiddenDim], [nBatchSize * 2, nHiddenDim])
    config.add_optimization_profile(profile)

    para = np.load("test3.npz")
    weight = [np.split(i, [nInputDim], axis=0) for i in np.split(para["rnn/basic_lstm_cell/kernel:0"], 4, axis=1)]
    bias = np.split(para["rnn/basic_lstm_cell/bias:0"], 4)
    weightXLayerList = [network.add_constant([nInputDim, nHiddenDim], weight[i][0].reshape(-1)) for i in range(4)]
    weightHLayerList = [network.add_constant([nHiddenDim, nHiddenDim], weight[i][1].reshape(-1)) for i in range(4)]
    biasLayerList = [network.add_constant([1, nHiddenDim], bias[i]) for i in range(4)]

    loop = network.add_loop()

    def gate(network, xTensor, wx, hTensor, wh, b, gateName):
        _h0 = network.add_matrix_multiply(xTensor, trt.MatrixOperation.NONE, wx, trt.MatrixOperation.NONE)
        _h1 = network.add_matrix_multiply(hTensor, trt.MatrixOperation.NONE, wh, trt.MatrixOperation.NONE)
        _h2 = network.add_elementwise(_h0.get_output(0), _h1.get_output(0), trt.ElementWiseOperation.SUM)
        _h3 = network.add_elementwise(_h2.get_output(0), b, trt.ElementWiseOperation.SUM)
        if gateName == "F" and np.abs(forgetBias) > epsilon:
            _constant = network.add_constant([1, 1], np.array(forgetBias, dtype=np.float32))
            _h4 = network.add_elementwise(_h3.get_output(0), _constant.get_output(0), trt.ElementWiseOperation.SUM)
        else:
            _h4 = _h3
        if gateName == "C":
            _h5 = network.add_activation(_h4.get_output(0), trt.ActivationType.TANH)
        else:
            _h5 = network.add_activation(_h4.get_output(0), trt.ActivationType.SIGMOID)
        return _h5

    _t0 = network.add_shape(inputT0)
    _t1 = network.add_slice(_t0.get_output(0), [1], [1], [1])
    _t2 = network.add_shuffle(_t1.get_output(0))
    _t2.reshape_dims = ()
    loop.add_trip_limit(_t2.get_output(0), trt.TripLimit.COUNT)
    iteratorLayer = loop.add_iterator(inputT0, 1, False)  # iterator throws one piece of inputT0 each time, shape: [nBatchSize, nInputDim]
    hiddenStateLayer = loop.add_recurrence(inputT1)  # initial hidden state and cell state
    cellStateLayer = loop.add_recurrence(inputT2)

    # order if weights is ICFO, rather than IFCO
    gateI = gate(network, iteratorLayer.get_output(0), weightXLayerList[0].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[0].get_output(0), biasLayerList[0].get_output(0), "I")
    gateC = gate(network, iteratorLayer.get_output(0), weightXLayerList[1].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[1].get_output(0), biasLayerList[1].get_output(0), "C")
    gateF = gate(network, iteratorLayer.get_output(0), weightXLayerList[2].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[2].get_output(0), biasLayerList[2].get_output(0), "F")
    gateO = gate(network, iteratorLayer.get_output(0), weightXLayerList[3].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[3].get_output(0), biasLayerList[3].get_output(0), "O")

    _h5 = network.add_elementwise(gateF.get_output(0), cellStateLayer.get_output(0), trt.ElementWiseOperation.PROD)
    _h6 = network.add_elementwise(gateI.get_output(0), gateC.get_output(0), trt.ElementWiseOperation.PROD)
    newCellStateLayer = network.add_elementwise(_h5.get_output(0), _h6.get_output(0), trt.ElementWiseOperation.SUM)
    _h7 = network.add_activation(newCellStateLayer.get_output(0), trt.ActivationType.TANH)
    newHiddenStateLayer = network.add_elementwise(gateO.get_output(0), _h7.get_output(0), trt.ElementWiseOperation.PROD)

    hiddenStateLayer.set_input(1, newHiddenStateLayer.get_output(0))
    cellStateLayer.set_input(1, newCellStateLayer.get_output(0))

    loopOutput0 = loop.add_loop_output(hiddenStateLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # output final hidden state, shape: [nBatchSize,nHiddenSize]
    loopOutput1 = loop.add_loop_output(newHiddenStateLayer.get_output(0), trt.LoopOutput.CONCATENATE, 1)  # output all hidden state, shape: [nBatchSize,nSequenceLength,nHiddenSize]
    loopOutput1.set_input(1, _t2.get_output(0))
    loopOutput2 = loop.add_loop_output(cellStateLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # output final cell state, shape: [nBatchSize,nHiddenSize]

    network.mark_output(loopOutput0.get_output(0))
    network.mark_output(loopOutput1.get_output(0))
    network.mark_output(loopOutput2.get_output(0))
    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    context = engine.create_execution_context()
    context.set_input_shape(engine.get_tensor_name(0), [nBatchSize, nSequenceLength, nInputDim])
    context.set_input_shape(engine.get_tensor_name(1), [nBatchSize, nHiddenDim])
    context.set_input_shape(engine.get_tensor_name(2), [nBatchSize, nHiddenDim])
    _, stream = cudart.cudaStreamCreate()

    inputH0 = np.ascontiguousarray(inputX.reshape(-1))
    inputH1 = np.ascontiguousarray(inputH.reshape(-1))
    inputH2 = np.ascontiguousarray(inputC.reshape(-1))
    outputH0 = np.empty(context.get_binding_shape(3), dtype=trt.nptype(engine.get_binding_dtype(3)))
    outputH1 = np.empty(context.get_binding_shape(4), dtype=trt.nptype(engine.get_binding_dtype(4)))
    outputH2 = np.empty(context.get_binding_shape(5), dtype=trt.nptype(engine.get_binding_dtype(5)))
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
    _, inputD1 = cudart.cudaMallocAsync(inputH1.nbytes, stream)
    _, inputD2 = cudart.cudaMallocAsync(inputH2.nbytes, stream)
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)
    _, outputD1 = cudart.cudaMallocAsync(outputH1.nbytes, stream)
    _, outputD2 = cudart.cudaMallocAsync(outputH2.nbytes, stream)

    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    cudart.cudaMemcpyAsync(inputD1, inputH1.ctypes.data, inputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    cudart.cudaMemcpyAsync(inputD2, inputH2.ctypes.data, inputH2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    context.execute_async_v2([int(inputD0), int(inputD1), int(inputD2), int(outputD0), int(outputD1), int(outputD2)], stream)

    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaMemcpyAsync(outputH1.ctypes.data, outputD1, outputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaMemcpyAsync(outputH2.ctypes.data, outputD2, outputH2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    outputTF = np.concatenate([x[:, np.newaxis, :] for x in outputTF], axis=1)

    #printArrayInfomation(inputX,"input")
    #print(inputX)
    #printArrayInfomation(outputTFhc[1],"TF h1")
    #printArrayInfomation(outputH0,"TRT h1")
    #printArrayInfomation(outputTFhc[0],"TF c1")
    #printArrayInfomation(outputH2,"TRT c1")
    #printArrayInfomation(outputTF,"TF AllOutput")
    #printArrayInfomation(outputH1,"TRT AllOutput")
    check(outputTFhc[1], outputH0, True, "h1")
    check(outputTFhc[0], outputH2, True, "c1")
    check(outputTF, outputH1, True, "AllOutput")

    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(inputD1)
    cudart.cudaFree(inputD2)
    cudart.cudaFree(outputD0)
    cudart.cudaFree(outputD1)
    cudart.cudaFree(outputD2)

def test4():
    print("\n[tf.nn.rnn_cell.LSTMCell / tf.contrib.rnn.LSTMCell] + [tf.nn.static_rnn / tf.nn.dynamic_rnn]")
    forgetBias = 1.0
    useProjection = True
    # TensorFlow part ----------------------------------------------------------
    x = tf.compat.v1.placeholder(tf.float32, [None, nSequenceLength, nInputDim], name="x")
    h0 = tf.compat.v1.placeholder(tf.float32, [None, nHiddenDim], name="h0")
    c0 = tf.compat.v1.placeholder(tf.float32, [None, nHiddenDim], name="c0")
    # tf.nn.rnn_cell.LSTMCell and tf.contrib.rnn.LSTMCell are alias
    cell    = tf.nn.rnn_cell.LSTMCell( \
    #cell    = tf.contrib.rnn.LSTMCell( \
                nHiddenDim,
                use_peepholes=False,
                cell_clip=None,
                initializer=None,
                num_proj=(nHiddenDim if useProjection else None),    # only None or nHiddenDim are supported? the order of weights using None is the same as the version which name contains Basic.
                proj_clip=None,
                num_unit_shards=None,
                num_proj_shards=None,
                forget_bias=forgetBias,
                state_is_tuple=True,
                activation=None,
                reuse=None,
                name=None,
                dtype=None,
                )
    # Two equivalent realization
    if True:
        y,hc    = tf.nn.static_rnn( \
                    cell,
                    [ x[:,i,:] for i in range(nSequenceLength) ],
                    initial_state=[c0,h0],
                    dtype=None,
                    sequence_length=None,
                    scope=None
                    )
    else:
        y,hc    = tf.nn.dynamic_rnn( \
                    cell,
                    x,
                    sequence_length=None,
                    initial_state=[c0,h0],
                    dtype=None,
                    parallel_iterations=None,
                    swap_memory=False,
                    time_major=False,
                    scope=None
                    )

    tfConfig = tf.compat.v1.ConfigProto()
    tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.compat.v1.Session(config=tfConfig)
    sess.run(tf.compat.v1.global_variables_initializer())
    outputTF, outputTFhc = sess.run([y, hc], feed_dict={x: inputX, h0: inputH, c0: inputC})

    tfPara = {}
    print("Weight:")
    for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
        name, value = i.name, sess.run(i)
        print(name, value.shape)
        tfPara[name] = value
    np.savez("test4.npz", **tfPara)
    sess.close()

    # TensorRT part ------------------------------------------------------------
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
        inputT0 = network.add_input("inputT0", trt.float32, (-1, -1, nInputDim))
    inputT1 = network.add_input("inputT1", trt.float32, (-1, nHiddenDim))
    inputT2 = network.add_input("inputT2", trt.float32, (-1, nHiddenDim))
    profile.set_shape(inputT0.name, [1, 1, nInputDim], [nBatchSize, nSequenceLength, nInputDim], [nBatchSize * 2, nSequenceLength * 2, nInputDim])
    profile.set_shape(inputT1.name, [1, nHiddenDim], [nBatchSize, nHiddenDim], [nBatchSize * 2, nHiddenDim])
    profile.set_shape(inputT2.name, [1, nHiddenDim], [nBatchSize, nHiddenDim], [nBatchSize * 2, nHiddenDim])
    config.add_optimization_profile(profile)

    para = np.load("test4.npz")
    weight = [np.split(i, [nInputDim], axis=0) for i in np.split(para["rnn/lstm_cell/kernel:0"], 4, axis=1)]
    bias = np.split(para["rnn/lstm_cell/bias:0"], 4)
    weightXLayerList = [network.add_constant([nInputDim, nHiddenDim], weight[i][0].reshape(-1)) for i in range(4)]
    weightHLayerList = [network.add_constant([nHiddenDim, nHiddenDim], weight[i][1].reshape(-1)) for i in range(4)]
    biasLayerList = [network.add_constant([1, nHiddenDim], bias[i]) for i in range(4)]

    loop = network.add_loop()

    def gate(network, xTensor, wx, hTensor, wh, b, gateName):
        _h0 = network.add_matrix_multiply(xTensor, trt.MatrixOperation.NONE, wx, trt.MatrixOperation.NONE)
        _h1 = network.add_matrix_multiply(hTensor, trt.MatrixOperation.NONE, wh, trt.MatrixOperation.NONE)
        _h2 = network.add_elementwise(_h0.get_output(0), _h1.get_output(0), trt.ElementWiseOperation.SUM)
        _h3 = network.add_elementwise(_h2.get_output(0), b, trt.ElementWiseOperation.SUM)
        if gateName == "F" and np.abs(forgetBias) > epsilon:
            _constant = network.add_constant([1, 1], np.array(forgetBias, dtype=np.float32))
            _h4 = network.add_elementwise(_h3.get_output(0), _constant.get_output(0), trt.ElementWiseOperation.SUM)
        else:
            _h4 = _h3
        if gateName == "C":
            _h5 = network.add_activation(_h4.get_output(0), trt.ActivationType.TANH)
        else:
            _h5 = network.add_activation(_h4.get_output(0), trt.ActivationType.SIGMOID)
        return _h5

    _t0 = network.add_shape(inputT0)
    _t1 = network.add_slice(_t0.get_output(0), [1], [1], [1])
    _t2 = network.add_shuffle(_t1.get_output(0))
    _t2.reshape_dims = ()
    loop.add_trip_limit(_t2.get_output(0), trt.TripLimit.COUNT)
    iteratorLayer = loop.add_iterator(inputT0, 1, False)  # iterator throws one piece of inputT0 each time, shape: [nBatchSize, nInputDim]
    hiddenStateLayer = loop.add_recurrence(inputT1)  # initial hidden state and cell state
    cellStateLayer = loop.add_recurrence(inputT2)

    # the order of weights is ICFO rather than IFCO
    gateI = gate(network, iteratorLayer.get_output(0), weightXLayerList[0].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[0].get_output(0), biasLayerList[0].get_output(0), "I")
    gateC = gate(network, iteratorLayer.get_output(0), weightXLayerList[1].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[1].get_output(0), biasLayerList[1].get_output(0), "C")
    gateF = gate(network, iteratorLayer.get_output(0), weightXLayerList[2].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[2].get_output(0), biasLayerList[2].get_output(0), "F")
    gateO = gate(network, iteratorLayer.get_output(0), weightXLayerList[3].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[3].get_output(0), biasLayerList[3].get_output(0), "O")

    _h5 = network.add_elementwise(gateF.get_output(0), cellStateLayer.get_output(0), trt.ElementWiseOperation.PROD)
    _h6 = network.add_elementwise(gateI.get_output(0), gateC.get_output(0), trt.ElementWiseOperation.PROD)
    newCellStateLayer = network.add_elementwise(_h5.get_output(0), _h6.get_output(0), trt.ElementWiseOperation.SUM)
    _h7 = network.add_activation(newCellStateLayer.get_output(0), trt.ActivationType.TANH)
    _h8 = network.add_elementwise(gateO.get_output(0), _h7.get_output(0), trt.ElementWiseOperation.PROD)
    if useProjection:
        matrix2D = network.add_constant([nHiddenDim, nHiddenDim], para["rnn/lstm_cell/projection/kernel:0"].astype(np.float32))
        newHiddenStateLayer = network.add_matrix_multiply(_h8.get_output(0), trt.MatrixOperation.NONE, matrix2D.get_output(0), trt.MatrixOperation.NONE)
    else:
        newHiddenStateLayer = _h8

    hiddenStateLayer.set_input(1, newHiddenStateLayer.get_output(0))
    cellStateLayer.set_input(1, newCellStateLayer.get_output(0))

    loopOutput0 = loop.add_loop_output(hiddenStateLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # output final hidden state, shape: [nBatchSize,nHiddenSize]
    loopOutput1 = loop.add_loop_output(newHiddenStateLayer.get_output(0), trt.LoopOutput.CONCATENATE, 1)  # output all hidden state, shape: [nBatchSize,nSequenceLength,nHiddenSize]
    loopOutput1.set_input(1, _t2.get_output(0))
    loopOutput2 = loop.add_loop_output(cellStateLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # output final cell state, shape: [nBatchSize,nHiddenSize]

    network.mark_output(loopOutput0.get_output(0))
    network.mark_output(loopOutput1.get_output(0))
    network.mark_output(loopOutput2.get_output(0))
    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    context = engine.create_execution_context()
    context.set_input_shape(engine.get_tensor_name(0), [nBatchSize, nSequenceLength, nInputDim])
    context.set_input_shape(engine.get_tensor_name(1), [nBatchSize, nHiddenDim])
    context.set_input_shape(engine.get_tensor_name(2), [nBatchSize, nHiddenDim])
    _, stream = cudart.cudaStreamCreate()

    inputH0 = np.ascontiguousarray(inputX.reshape(-1))
    inputH1 = np.ascontiguousarray(inputH.reshape(-1))
    inputH2 = np.ascontiguousarray(inputC.reshape(-1))
    outputH0 = np.empty(context.get_binding_shape(3), dtype=trt.nptype(engine.get_binding_dtype(3)))
    outputH1 = np.empty(context.get_binding_shape(4), dtype=trt.nptype(engine.get_binding_dtype(4)))
    outputH2 = np.empty(context.get_binding_shape(5), dtype=trt.nptype(engine.get_binding_dtype(5)))
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
    _, inputD1 = cudart.cudaMallocAsync(inputH1.nbytes, stream)
    _, inputD2 = cudart.cudaMallocAsync(inputH2.nbytes, stream)
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)
    _, outputD1 = cudart.cudaMallocAsync(outputH1.nbytes, stream)
    _, outputD2 = cudart.cudaMallocAsync(outputH2.nbytes, stream)

    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    cudart.cudaMemcpyAsync(inputD1, inputH1.ctypes.data, inputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    cudart.cudaMemcpyAsync(inputD2, inputH2.ctypes.data, inputH2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    context.execute_async_v2([int(inputD0), int(inputD1), int(inputD2), int(outputD0), int(outputD1), int(outputD2)], stream)

    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaMemcpyAsync(outputH1.ctypes.data, outputD1, outputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaMemcpyAsync(outputH2.ctypes.data, outputD2, outputH2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    outputTF = np.concatenate([x[:, np.newaxis, :] for x in outputTF], axis=1)

    #printArrayInfomation(inputX,"input")
    #print(inputX)
    #printArrayInfomation(outputTFhc[1],"TF h1")
    #printArrayInfomation(outputH0,"TRT h1")
    #printArrayInfomation(outputTFhc[0],"TF c1")
    #printArrayInfomation(outputH2,"TRT c1")
    #printArrayInfomation(outputTF,"TF AllOutput")
    #printArrayInfomation(outputH1,"TRT AllOutput")
    check(outputTFhc[1], outputH0, True, "h1")
    check(outputTFhc[0], outputH2, True, "c1")
    check(outputTF, outputH1, True, "AllOutput")

    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(inputD1)
    cudart.cudaFree(inputD2)
    cudart.cudaFree(outputD0)
    cudart.cudaFree(outputD1)
    cudart.cudaFree(outputD2)

def test5():
    print("\ntf.contrib.cudnn_rnn.CudnnLSTM ------------------------------------")
    # TensorFlow part ----------------------------------------------------------
    x = tf.compat.v1.placeholder(tf.float32, [None, nSequenceLength, nInputDim], name="x")
    h0 = tf.compat.v1.placeholder(tf.float32, [None, 1, nHiddenDim], name="h0")
    c0 = tf.compat.v1.placeholder(tf.float32, [None, 1, nHiddenDim], name="c0")
    lstm    = tf.contrib.cudnn_rnn.CudnnLSTM( \
                1,
                nHiddenDim,
                input_mode="linear_input",
                direction="unidirectional",
                dropout=0.0,
                seed=None,
                dtype=tf.dtypes.float32,
                kernel_initializer=None,
                bias_initializer=None,
                name=None
                )
    y, hc = lstm(
        x,
        initial_state=(h0, c0),  # [h0,c0] rather than [c0,h0]
        sequence_lengths=None,
        time_major=False,
        training=False
    )

    tfConfig = tf.compat.v1.ConfigProto()
    tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.compat.v1.Session(config=tfConfig)
    sess.run(tf.compat.v1.global_variables_initializer())
    outputTF, outputTFhc = sess.run([y, hc], feed_dict={x: inputX, h0: inputH.reshape([nBatchSize, 1, nHiddenDim]), c0: inputC.reshape([nBatchSize, 1, nHiddenDim])})  # 输入形状与其他几个不同

    tfPara = {}
    print("Weight:")
    for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
        name, value = i.name, sess.run(i)
        print(name, value.shape)
        tfPara[name] = value
    np.savez("test5.npz", **tfPara)
    sess.close()

    # TensorRT part ------------------------------------------------------------
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
        inputT0 = network.add_input("inputT0", trt.float32, (-1, -1, nInputDim))
    inputT1 = network.add_input("inputT1", trt.float32, (-1, nHiddenDim))
    inputT2 = network.add_input("inputT2", trt.float32, (-1, nHiddenDim))
    profile.set_shape(inputT0.name, [1, 1, nInputDim], [nBatchSize, nSequenceLength, nInputDim], [nBatchSize * 2, nSequenceLength * 2, nInputDim])
    profile.set_shape(inputT1.name, [1, nHiddenDim], [nBatchSize, nHiddenDim], [nBatchSize * 2, nHiddenDim])
    profile.set_shape(inputT2.name, [1, nHiddenDim], [nBatchSize, nHiddenDim], [nBatchSize * 2, nHiddenDim])
    config.add_optimization_profile(profile)

    para = np.load("test5.npz")["cudnn_lstm/opaque_kernel:0"]
    # --------------------------------+-------------------------------+
    # | weight                        | bias                          |
    # --------------------------------+-------------------------------+
    # |wIX|wCX|wFX|wOX|wIH|wCH|wFH|wOH|bIX|bCX|bFX|bOX|bIH|bCH|bFH|bOH|
    # --------------------------------+-------------------------------+
    weightX, weightH, bias = np.split(para, [nInputDim * nHiddenDim * 4, (nInputDim + nHiddenDim) * nHiddenDim * 4])
    weightXLayerList = [network.add_constant([nInputDim, nHiddenDim], i.reshape(nHiddenDim, nInputDim).transpose().reshape(-1)) for i in np.split(weightX, 4)]
    weightHLayerList = [network.add_constant([nHiddenDim, nHiddenDim], i.reshape(nHiddenDim, nHiddenDim).transpose().reshape(-1)) for i in np.split(weightH, 4)]
    biasLayerList = [network.add_constant([1, nHiddenDim], i.reshape(-1)) for i in np.split(np.sum(bias.reshape(2, -1), axis=0), 4)]

    def gate(network, xTensor, wx, hTensor, wh, b, isSigmoid):
        _h0 = network.add_matrix_multiply(xTensor, trt.MatrixOperation.NONE, wx, trt.MatrixOperation.NONE)
        _h1 = network.add_matrix_multiply(hTensor, trt.MatrixOperation.NONE, wh, trt.MatrixOperation.NONE)
        _h2 = network.add_elementwise(_h0.get_output(0), _h1.get_output(0), trt.ElementWiseOperation.SUM)
        _h3 = network.add_elementwise(_h2.get_output(0), b, trt.ElementWiseOperation.SUM)
        _h4 = network.add_activation(_h3.get_output(0), trt.ActivationType.SIGMOID if isSigmoid else trt.ActivationType.TANH)
        return _h4

    loop = network.add_loop()
    _t0 = network.add_shape(inputT0)
    _t1 = network.add_slice(_t0.get_output(0), [1], [1], [1])
    _t2 = network.add_shuffle(_t1.get_output(0))
    _t2.reshape_dims = ()
    loop.add_trip_limit(_t2.get_output(0), trt.TripLimit.COUNT)
    iteratorLayer = loop.add_iterator(inputT0, 1, False)  # iterator throws one piece of inputT0 each time, shape: [nBatchSize, nInputDim]
    hiddenStateLayer = loop.add_recurrence(inputT1)  # initial hidden state and cell state
    cellStateLayer = loop.add_recurrence(inputT2)

    gateI = gate(network, iteratorLayer.get_output(0), weightXLayerList[0].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[0].get_output(0), biasLayerList[0].get_output(0), True)
    gateF = gate(network, iteratorLayer.get_output(0), weightXLayerList[1].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[1].get_output(0), biasLayerList[1].get_output(0), True)
    gateC = gate(network, iteratorLayer.get_output(0), weightXLayerList[2].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[2].get_output(0), biasLayerList[2].get_output(0), False)
    gateO = gate(network, iteratorLayer.get_output(0), weightXLayerList[3].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[3].get_output(0), biasLayerList[3].get_output(0), True)

    _h5 = network.add_elementwise(gateF.get_output(0), cellStateLayer.get_output(0), trt.ElementWiseOperation.PROD)
    _h6 = network.add_elementwise(gateI.get_output(0), gateC.get_output(0), trt.ElementWiseOperation.PROD)
    newCellStateLayer = network.add_elementwise(_h5.get_output(0), _h6.get_output(0), trt.ElementWiseOperation.SUM)
    _h7 = network.add_activation(newCellStateLayer.get_output(0), trt.ActivationType.TANH)
    newHiddenStateLayer = network.add_elementwise(gateO.get_output(0), _h7.get_output(0), trt.ElementWiseOperation.PROD)

    hiddenStateLayer.set_input(1, newHiddenStateLayer.get_output(0))
    cellStateLayer.set_input(1, newCellStateLayer.get_output(0))

    loopOutput0 = loop.add_loop_output(hiddenStateLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # output final hidden state, shape: [nBatchSize,nHiddenSize]
    loopOutput1 = loop.add_loop_output(newHiddenStateLayer.get_output(0), trt.LoopOutput.CONCATENATE, 1)  # output all hidden state, shape: [nBatchSize,nSequenceLength,nHiddenSize]
    loopOutput1.set_input(1, _t2.get_output(0))
    loopOutput2 = loop.add_loop_output(cellStateLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # output final cell state, shape: [nBatchSize,nHiddenSize]

    network.mark_output(loopOutput0.get_output(0))
    network.mark_output(loopOutput1.get_output(0))
    network.mark_output(loopOutput2.get_output(0))
    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    context = engine.create_execution_context()
    context.set_input_shape(engine.get_tensor_name(0), [nBatchSize, nSequenceLength, nInputDim])
    context.set_input_shape(engine.get_tensor_name(1), [nBatchSize, nHiddenDim])
    context.set_input_shape(engine.get_tensor_name(2), [nBatchSize, nHiddenDim])
    _, stream = cudart.cudaStreamCreate()

    inputH0 = np.ascontiguousarray(inputX.reshape(-1))
    inputH1 = np.ascontiguousarray(inputH.reshape(-1))
    inputH2 = np.ascontiguousarray(inputC.reshape(-1))
    outputH0 = np.empty(context.get_binding_shape(3), dtype=trt.nptype(engine.get_binding_dtype(3)))
    outputH1 = np.empty(context.get_binding_shape(4), dtype=trt.nptype(engine.get_binding_dtype(4)))
    outputH2 = np.empty(context.get_binding_shape(5), dtype=trt.nptype(engine.get_binding_dtype(5)))
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
    _, inputD1 = cudart.cudaMallocAsync(inputH1.nbytes, stream)
    _, inputD2 = cudart.cudaMallocAsync(inputH2.nbytes, stream)
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)
    _, outputD1 = cudart.cudaMallocAsync(outputH1.nbytes, stream)
    _, outputD2 = cudart.cudaMallocAsync(outputH2.nbytes, stream)

    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    cudart.cudaMemcpyAsync(inputD1, inputH1.ctypes.data, inputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    cudart.cudaMemcpyAsync(inputD2, inputH2.ctypes.data, inputH2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    context.execute_async_v2([int(inputD0), int(inputD1), int(inputD2), int(outputD0), int(outputD1), int(outputD2)], stream)

    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaMemcpyAsync(outputH1.ctypes.data, outputD1, outputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaMemcpyAsync(outputH2.ctypes.data, outputD2, outputH2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    outputTFhc = [i[:, 0, :] for i in outputTFhc]

    #printArrayInfomation(inputX,"input")
    #print(inputX)
    #printArrayInfomation(outputTFhc[0],"TF h1")
    #printArrayInfomation(outputH0,"TRT h1")
    #printArrayInfomation(outputTFhc[1],"TF c1")
    #printArrayInfomation(outputH2,"TRT c1")
    #printArrayInfomation(outputTF,"TF AllOutput")
    #printArrayInfomation(outputH1,"TRT AllOutput")
    check(outputTFhc[0], outputH0, True, "h1")
    check(outputTFhc[1], outputH2, True, "c1")
    check(outputTF, outputH1, True, "AllOutput")

    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(inputD1)
    cudart.cudaFree(inputD2)
    cudart.cudaFree(outputD0)
    cudart.cudaFree(outputD1)
    cudart.cudaFree(outputD2)

def test6():
    print("\n[tf.contrib.rnn.LSTMBlockCell / tf.contrib.rnn.LSTMBlockFusedCell] + [tf.nn.static_rnn / tf.nn.dynamic_rnn]")
    forgetBias = 1.0
    # TensorFlow part ----------------------------------------------------------
    x = tf.compat.v1.placeholder(tf.float32, [None, nSequenceLength, nInputDim], name="x")
    h0 = tf.compat.v1.placeholder(tf.float32, [None, nHiddenDim], name="h0")
    c0 = tf.compat.v1.placeholder(tf.float32, [None, nHiddenDim], name="c0")
    # tf.contrib.rnn.LSTMBlockCell and tf.contrib.rnn.LSTMBlockFusedCell shares the same API and the order of weights
    cell    = tf.contrib.rnn.LSTMBlockCell( \
    #cell    = tf.contrib.rnn.LSTMBlockFusedCell( \
               nHiddenDim,
               forget_bias=forgetBias,
               cell_clip=None,
               use_peephole=False,
               dtype=None,
               reuse=None,
               name="tf-contrib-rnn-LSTMBlockCell-LSTM"
    )
    # Two equivalent realization
    if True:
        y,hc    = tf.nn.static_rnn( \
                    cell,
                    [ x[:,i,:] for i in range(nSequenceLength) ],
                    initial_state=[c0,h0],
                    dtype=None,
                    sequence_length=None,
                    scope=None
                    )
    else:
        y,hc    = tf.nn.dynamic_rnn( \
                    cell,
                    x,
                    sequence_length=None,
                    initial_state=[c0,h0],
                    dtype=None,
                    parallel_iterations=None,
                    swap_memory=False,
                    time_major=False,
                    scope=None
                    )

    tfConfig = tf.compat.v1.ConfigProto()
    tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.compat.v1.Session(config=tfConfig)
    sess.run(tf.compat.v1.global_variables_initializer())
    outputTF, outputTFhc = sess.run([y, hc], feed_dict={x: inputX, h0: inputH, c0: inputC})

    tfPara = {}
    print("Weight:")
    for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
        name, value = i.name, sess.run(i)
        print(name, value.shape)
        tfPara[name] = value
    np.savez("test6.npz", **tfPara)
    sess.close()

    # TensorRT part ------------------------------------------------------------
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
        inputT0 = network.add_input("inputT0", trt.float32, (-1, -1, nInputDim))
    inputT1 = network.add_input("inputT1", trt.float32, (-1, nHiddenDim))
    inputT2 = network.add_input("inputT2", trt.float32, (-1, nHiddenDim))
    profile.set_shape(inputT0.name, [1, 1, nInputDim], [nBatchSize, nSequenceLength, nInputDim], [nBatchSize * 2, nSequenceLength * 2, nInputDim])
    profile.set_shape(inputT1.name, [1, nHiddenDim], [nBatchSize, nHiddenDim], [nBatchSize * 2, nHiddenDim])
    profile.set_shape(inputT2.name, [1, nHiddenDim], [nBatchSize, nHiddenDim], [nBatchSize * 2, nHiddenDim])
    config.add_optimization_profile(profile)

    para = np.load("test6.npz")
    weight = [np.split(i, [nInputDim], axis=0) for i in np.split(para["rnn/tf-contrib-rnn-LSTMBlockCell-LSTM/kernel:0"], 4, axis=1)]
    bias = np.split(para["rnn/tf-contrib-rnn-LSTMBlockCell-LSTM/bias:0"], 4)
    weightXLayerList = [network.add_constant([nInputDim, nHiddenDim], weight[i][0].reshape(-1)) for i in range(4)]
    weightHLayerList = [network.add_constant([nHiddenDim, nHiddenDim], weight[i][1].reshape(-1)) for i in range(4)]
    biasLayerList = [network.add_constant([1, nHiddenDim], bias[i]) for i in range(4)]

    loop = network.add_loop()

    def gate(network, xTensor, wx, hTensor, wh, b, gateName):
        _h0 = network.add_matrix_multiply(xTensor, trt.MatrixOperation.NONE, wx, trt.MatrixOperation.NONE)
        _h1 = network.add_matrix_multiply(hTensor, trt.MatrixOperation.NONE, wh, trt.MatrixOperation.NONE)
        _h2 = network.add_elementwise(_h0.get_output(0), _h1.get_output(0), trt.ElementWiseOperation.SUM)
        _h3 = network.add_elementwise(_h2.get_output(0), b, trt.ElementWiseOperation.SUM)
        if gateName == "F" and np.abs(forgetBias) > epsilon:
            _constant = network.add_constant([1, 1], np.array(forgetBias, dtype=np.float32))
            _h4 = network.add_elementwise(_h3.get_output(0), _constant.get_output(0), trt.ElementWiseOperation.SUM)
        else:
            _h4 = _h3
        if gateName == "C":
            _h5 = network.add_activation(_h4.get_output(0), trt.ActivationType.TANH)
        else:
            _h5 = network.add_activation(_h4.get_output(0), trt.ActivationType.SIGMOID)
        return _h5

    _t0 = network.add_shape(inputT0)
    _t1 = network.add_slice(_t0.get_output(0), [1], [1], [1])
    _t2 = network.add_shuffle(_t1.get_output(0))
    _t2.reshape_dims = ()
    loop.add_trip_limit(_t2.get_output(0), trt.TripLimit.COUNT)
    iteratorLayer = loop.add_iterator(inputT0, 1, False)  # iterator throws one piece of inputT0 each time, shape: [nBatchSize, nInputDim]
    hiddenStateLayer = loop.add_recurrence(inputT1)  # initial hidden state and cell state
    cellStateLayer = loop.add_recurrence(inputT2)

    # 权重顺序是 ICFO 而不是 IFCO
    gateI = gate(network, iteratorLayer.get_output(0), weightXLayerList[0].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[0].get_output(0), biasLayerList[0].get_output(0), "I")
    gateC = gate(network, iteratorLayer.get_output(0), weightXLayerList[1].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[1].get_output(0), biasLayerList[1].get_output(0), "C")
    gateF = gate(network, iteratorLayer.get_output(0), weightXLayerList[2].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[2].get_output(0), biasLayerList[2].get_output(0), "F")
    gateO = gate(network, iteratorLayer.get_output(0), weightXLayerList[3].get_output(0), hiddenStateLayer.get_output(0), weightHLayerList[3].get_output(0), biasLayerList[3].get_output(0), "O")

    _h5 = network.add_elementwise(gateF.get_output(0), cellStateLayer.get_output(0), trt.ElementWiseOperation.PROD)
    _h6 = network.add_elementwise(gateI.get_output(0), gateC.get_output(0), trt.ElementWiseOperation.PROD)
    newCellStateLayer = network.add_elementwise(_h5.get_output(0), _h6.get_output(0), trt.ElementWiseOperation.SUM)
    _h7 = network.add_activation(newCellStateLayer.get_output(0), trt.ActivationType.TANH)
    newHiddenStateLayer = network.add_elementwise(gateO.get_output(0), _h7.get_output(0), trt.ElementWiseOperation.PROD)

    hiddenStateLayer.set_input(1, newHiddenStateLayer.get_output(0))
    cellStateLayer.set_input(1, newCellStateLayer.get_output(0))

    loopOutput0 = loop.add_loop_output(hiddenStateLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # output final hidden state, shape: [nBatchSize,nHiddenSize]
    loopOutput1 = loop.add_loop_output(newHiddenStateLayer.get_output(0), trt.LoopOutput.CONCATENATE, 1)  # output all hidden state, shape: [nBatchSize,nSequenceLength,nHiddenSize]
    loopOutput1.set_input(1, _t2.get_output(0))
    loopOutput2 = loop.add_loop_output(cellStateLayer.get_output(0), trt.LoopOutput.LAST_VALUE, 0)  # output final cell state, shape: [nBatchSize,nHiddenSize]

    network.mark_output(loopOutput0.get_output(0))
    network.mark_output(loopOutput1.get_output(0))
    network.mark_output(loopOutput2.get_output(0))
    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    context = engine.create_execution_context()
    context.set_input_shape(engine.get_tensor_name(0), [nBatchSize, nSequenceLength, nInputDim])
    context.set_input_shape(engine.get_tensor_name(1), [nBatchSize, nHiddenDim])
    context.set_input_shape(engine.get_tensor_name(2), [nBatchSize, nHiddenDim])
    _, stream = cudart.cudaStreamCreate()

    inputH0 = np.ascontiguousarray(inputX.reshape(-1))
    inputH1 = np.ascontiguousarray(inputH.reshape(-1))
    inputH2 = np.ascontiguousarray(inputC.reshape(-1))
    outputH0 = np.empty(context.get_binding_shape(3), dtype=trt.nptype(engine.get_binding_dtype(3)))
    outputH1 = np.empty(context.get_binding_shape(4), dtype=trt.nptype(engine.get_binding_dtype(4)))
    outputH2 = np.empty(context.get_binding_shape(5), dtype=trt.nptype(engine.get_binding_dtype(5)))
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
    _, inputD1 = cudart.cudaMallocAsync(inputH1.nbytes, stream)
    _, inputD2 = cudart.cudaMallocAsync(inputH2.nbytes, stream)
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)
    _, outputD1 = cudart.cudaMallocAsync(outputH1.nbytes, stream)
    _, outputD2 = cudart.cudaMallocAsync(outputH2.nbytes, stream)

    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    cudart.cudaMemcpyAsync(inputD1, inputH1.ctypes.data, inputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    cudart.cudaMemcpyAsync(inputD2, inputH2.ctypes.data, inputH2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    context.execute_async_v2([int(inputD0), int(inputD1), int(inputD2), int(outputD0), int(outputD1), int(outputD2)], stream)

    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaMemcpyAsync(outputH1.ctypes.data, outputD1, outputH1.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaMemcpyAsync(outputH2.ctypes.data, outputD2, outputH2.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
    cudart.cudaStreamSynchronize(stream)

    outputTF = np.concatenate([x[:, np.newaxis, :] for x in outputTF], axis=1)

    #printArrayInfomation(inputX,"input")
    #print(inputX)
    #printArrayInfomation(outputTFhc[1],"TF h1")
    #printArrayInfomation(outputH0,"TRT h1")
    #printArrayInfomation(outputTFhc[0],"TF c1")
    #printArrayInfomation(outputH2,"TRT c1")
    #printArrayInfomation(outputTF,"TF AllOutput")
    #printArrayInfomation(outputH1,"TRT AllOutput")
    check(outputTFhc[1], outputH0, True, "h1")
    check(outputTFhc[0], outputH2, True, "c1")
    check(outputTF, outputH1, True, "AllOutput")

    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(inputD1)
    cudart.cudaFree(inputD2)
    cudart.cudaFree(outputD0)
    cudart.cudaFree(outputD1)
    cudart.cudaFree(outputD2)

if __name__ == "__main__":
    cudart.cudaDeviceSynchronize()
    np.set_printoptions(precision=3, linewidth=100, suppress=True)

    test1()  # tf.keras.layers.LSTM or tf.keras.layers.LSTMCell + tf.keras.layers.RNN
    test2()  # tf.keras.layers.CuDNNLSTM
    test3()  # tf.nn.rnn_cell.BasicLSTMCell / tf.contrib.rnn.BasicLSTMCell + tf.nn.static_rnn / tf.nn.dynamic_rnn
    test4()  # tf.nn.rnn_cell.LSTMCell / tf.contrib.rnn.LSTMCell + tf.nn.static_rnn / tf.nn.dynamic_rnn
    test5()  # tf.contrib.cudnn_rnn.CudnnLSTM
    test6()  # tf.contrib.rnn.LSTMBlockCell / tf.contrib.rnn.LSTMBlockFusedCell + tf.nn.static_rnn / tf.nn.dynamic_rnn

    print("\ntest finish!")
