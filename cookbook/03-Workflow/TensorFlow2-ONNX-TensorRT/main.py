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

import os
from datetime import datetime as dt
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf2
import tensorrt as trt
from tensorflow.python.framework.convert_to_constants import \
    convert_variables_to_constants_v2
from tensorrt_cookbook import MyCalibratorMNIST, TRTWrapperV1, case_mark

np.random.seed(31193)
tf2.random.set_seed(97)
batch_size, height, width = 128, 28, 28
n_epoch = 10
data_path = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data"
train_data_file = data_path / "TrainData.npz"
test_data_file = data_path / "TestData.npz"
pbfile_path = "./pbModel/"
pbFile = "model.pb"
onnx_file_trained = "./model.onnx"

data = np.load(data_path / "InferenceData.npy")
calibration_data_file = data_path / "CalibrationData.npy"
trt_file = Path("model.trt")
int8_cache_file = Path("model.Int8Cache")

tf2.config.experimental.set_memory_growth(tf2.config.list_physical_devices("GPU")[0], True)

def case_get_onnx(b_single_pbfile: bool):

    class MyData:

        def __init__(self, b_train: bool = True):
            data = np.load(train_data_file if b_train else test_data_file)
            self.data = data["data"].reshape(-1, height, width, 1)
            self.label = data["label"]
            return

        def get_data(self):
            return self.data, self.label

    train_data_loader = MyData(True).get_data()
    test_data_loader = MyData(False).get_data()

    modelInput = tf2.keras.Input(shape=[height, width, 1], dtype=tf2.dtypes.float32, name="x")

    layerConv1 = tf2.keras.layers.Conv2D(32, [5, 5], strides=[1, 1], padding="same", data_format=None, dilation_rate=[1, 1], groups=1, activation="relu", use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="conv1")
    x = layerConv1(modelInput)

    layerPool1 = tf2.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding="same", data_format=None, name="pool1")
    x = layerPool1(x)

    layerConv2 = tf2.keras.layers.Conv2D(64, [5, 5], strides=[1, 1], padding="same", data_format=None, dilation_rate=[1, 1], groups=1, activation="relu", use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="conv2")
    x = layerConv2(x)

    laerPool2 = tf2.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding="same", data_format=None, name="pool2")
    x = laerPool2(x)

    layerReshape = tf2.keras.layers.Reshape([-1], name="reshape")
    x = layerReshape(x)

    layerDense1 = tf2.keras.layers.Dense(1024, activation="relu", use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="dense1")
    x = layerDense1(x)

    layerDense2 = tf2.keras.layers.Dense(10, activation=None, use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name="dense2")
    x = layerDense2(x)

    layerSoftmax = tf2.keras.layers.Softmax(axis=1, name="softmax")
    y = layerSoftmax(x)
    y.name = "y"

    model = tf2.keras.Model(inputs=modelInput, outputs=y, name="MNISTExample")

    model.summary()

    model.compile(
        loss=tf2.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf2.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    history = model.fit(*train_data_loader, batch_size=128, epochs=n_epoch, validation_split=0.1)

    testScore = model.evaluate(*test_data_loader, verbose=2)
    print(f"[{dt.now()}], loss = {testScore[0]}, accuracy = {testScore[1]}")

    tf2.saved_model.save(model, pbfile_path)

    if b_single_pbfile:
        modelFunction = tf2.function(lambda Input: model(Input)).get_concrete_function(tf2.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
        frozen_func = convert_variables_to_constants_v2(modelFunction)
        frozen_func.graph.as_graph_def()
        print("_________________________________________________________________")
        print("Frozen model inputs:\n", frozen_func.inputs)
        print("Frozen model outputs:\n", frozen_func.outputs)
        print("Frozen model layers:")
        for op in frozen_func.graph.get_operations():
            print(op.name)
        tf2.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=pbfile_path, name=pbFile, as_text=False)

    print("Succeed building model in TensorFlow2")

    # Export model to ONNX file
    # Remove `--inputs-as-nchw` to use NHWC format mandatorily
    command = f"python3 -m tf2onnx.convert --opset=13 --inputs-as-nchw 'Input:0' --output {onnx_file_trained} "
    if b_single_pbfile:
        command += f"--input {pbfile_path + pbFile} --inputs 'Input:0' --outputs 'Identity:0'"
    else:
        command += f"--saved-model {pbfile_path}"
    os.system(command)
    print("Succeed converting model into ONNX")

@case_mark
def case_normal(is_fp16: bool = False, is_int8_ptq: bool = False, b_single_pbfile: bool = True):
    data_local = {"x": data.reshape(-1, 1, height, width) if b_single_pbfile else data.reshape(-1, height, width, 1)}
    shape = list(data_local["x"].shape)

    tw = TRTWrapperV1()

    parser = trt.OnnxParser(tw.network, tw.logger)
    with open(onnx_file_trained, "rb") as model:
        parser.parse(model.read())

    x = tw.network.get_input(0)
    x.name = "x"
    tw.profile.set_shape(x.name, shape, [1] + shape[1:], [4] + shape[1:])
    tw.config.add_optimization_profile(tw.profile)

    y = tw.network.get_output(0)
    y.name = "y"
    layer_topk = tw.network.add_topk(y, trt.TopKOperation.MAX, 1, 1 << 1)
    layer_topk.name = "TopK"
    z = layer_topk.get_output(1)
    z.name = "z"
    tw.network.mark_output(z)

    suffix = ""
    if is_fp16:  # FP16 and INT8 can be used at the same time
        print("Using FP16")
        tw.config.set_flag(trt.BuilderFlag.FP16)
        suffix += "-fp16"
    if is_int8_ptq:
        print("Using INT8-PTQ")
        tw.config.set_flag(trt.BuilderFlag.INT8)
        input_info = {"x": [data_local["x"].dtype, data_local["x"].shape]}
        tw.config.int8_calibrator = MyCalibratorMNIST(input_info, calibration_data_file, int8_cache_file)
        suffix += "-int8ptq"

    tw.build()
    tw.serialize_engine(Path(str(trt_file) + suffix))

    tw.setup(data_local)
    tw.infer()
    return

if __name__ == "__main__":
    os.system("rm -rf *.trt* *.Int8Cache")

    case_get_onnx(True)  # Save pb file in one file
    #case_get_onnx(False)  # Save pb file in separated files
    case_normal()
    case_normal(is_fp16=True, b_single_pbfile=True)
    case_normal(is_int8_ptq=True, b_single_pbfile=True)

    print("Finish")
