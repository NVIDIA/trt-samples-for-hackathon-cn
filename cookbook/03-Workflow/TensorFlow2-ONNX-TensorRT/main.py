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
from tensorrt_cookbook import CookbookCalibratorMNIST, TRTWrapperV1, case_mark

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
            self.data = data["data"].reshape(-1, height, width, 1).astype(np.float32)
            self.label = data["label"]

        def get_data(self):
            return self.data, self.label

    train_data_loader = MyData(True).get_data()
    test_data_loader = MyData(False).get_data()

    # 使用 Sequential API 或 Functional API 构建模型,简化参数
    model_input = tf2.keras.Input(shape=(height, width, 1), dtype=tf2.float32, name="x")

    # 简化 Conv2D 层参数,只保留必要的
    x = tf2.keras.layers.Conv2D(32, (5, 5), padding="same", activation="relu", name="conv1")(model_input)
    x = tf2.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    x = tf2.keras.layers.Conv2D(64, (5, 5), padding="same", activation="relu", name="conv2")(x)
    x = tf2.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # 使用 Flatten 代替 Reshape 更加清晰
    x = tf2.keras.layers.Flatten(name="flatten")(x)

    x = tf2.keras.layers.Dense(1024, activation="relu", name="dense1")(x)
    x = tf2.keras.layers.Dense(10, name="dense2")(x)

    y = tf2.keras.layers.Softmax(name="softmax")(x)

    model = tf2.keras.Model(inputs=model_input, outputs=y, name="MNISTExample")

    model.summary()

    # 使用最新的优化器 API
    model.compile(
        loss=tf2.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf2.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    history = model.fit(*train_data_loader, batch_size=128, epochs=n_epoch, validation_split=0.1, verbose=1)

    test_score = model.evaluate(*test_data_loader, verbose=2)
    print(f"[{dt.now()}], loss = {test_score[0]}, accuracy = {test_score[1]}")

    # 保存模型到 SavedModel 格式
    tf2.saved_model.save(model, pbfile_path)

    if b_single_pbfile:
        # 使用 tf.function 和 get_concrete_function 冻结图
        full_model = tf2.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(tf2.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

        # 转换为常量图
        frozen_func = convert_variables_to_constants_v2(full_model)

        print("_________________________________________________________________")
        print("Frozen model inputs:\n", frozen_func.inputs)
        print("Frozen model outputs:\n", frozen_func.outputs)
        print("Frozen model layers:")
        for op in frozen_func.graph.get_operations():
            print(op.name)

        # 写入冻结的图
        tf2.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=pbfile_path, name=pbFile, as_text=False)

    print("Succeed building model in TensorFlow2")

    # 导出模型到 ONNX 文件
    # 使用更新的 opset 版本
    command = f"python3 -m tf2onnx.convert --opset=17 --inputs-as-nchw 'x:0' --output {onnx_file_trained} "
    if b_single_pbfile:
        command += f"--input {pbfile_path + pbFile} --inputs 'x:0' --outputs 'Identity:0'"
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
        tw.config.int8_calibrator = CookbookCalibratorMNIST(input_info, calibration_data_file, int8_cache_file)
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
