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
import shutil
import subprocess
from datetime import datetime as dt
from pathlib import Path
import onnx.helper as onnx_helper
import numpy as np
import onnx

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf2
import tensorrt as trt
from tensorflow.python.framework.convert_to_constants import \
    convert_variables_to_constants_v2

if not hasattr(onnx_helper, "float32_to_bfloat16"):

    def _float32_to_bfloat16(value):
        float_array = np.asarray(value, dtype=np.float32)
        uint32_array = float_array.view(np.uint32)
        bfloat16_array = (uint32_array >> 16).astype(np.uint16)
        if bfloat16_array.ndim == 0:
            return int(bfloat16_array)
        return bfloat16_array

    onnx_helper.float32_to_bfloat16 = _float32_to_bfloat16

from tensorrt_cookbook import CookbookCalibratorMNIST, TRTWrapperV1, case_mark

np.random.seed(31193)
tf2.random.set_seed(97)
batch_size, height, width = 128, 28, 28
n_epoch = 10
data_path = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data"
train_data_file = data_path / "TrainData.npz"
test_data_file = data_path / "TestData.npz"
pbfile_path = Path("pbModel")
pbFile = pbfile_path / "model.pb"
onnx_file_trained = Path("model.onnx")

data = np.load(data_path / "InferenceData.npy")
calibration_data_file = data_path / "CalibrationData.npy"
trt_file = Path("model.trt")
int8_cache_file = Path("model.Int8Cache")

gpus = tf2.config.list_physical_devices("GPU")
for gpu in gpus:
    tf2.config.experimental.set_memory_growth(gpu, True)

class Net(tf2.keras.Model):

    def __init__(self):
        super(Net, self).__init__(name="MNISTExample")
        self.conv1 = tf2.keras.layers.Conv2D(32, (5, 5), padding="same", activation="relu", name="conv1")
        self.pool1 = tf2.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")
        self.conv2 = tf2.keras.layers.Conv2D(64, (5, 5), padding="same", activation="relu", name="conv2")
        self.pool2 = tf2.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")
        self.flatten = tf2.keras.layers.Flatten(name="flatten")
        self.dense1 = tf2.keras.layers.Dense(1024, activation="relu", name="dense1")
        self.dense2 = tf2.keras.layers.Dense(10, name="dense2")
        self.softmax = tf2.keras.layers.Softmax(name="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        y = self.softmax(x)
        return y

def case_get_onnx():

    class MyData:

        def __init__(self, b_train: bool = True):
            data = np.load(train_data_file if b_train else test_data_file)
            self.data = data["data"].reshape(-1, height, width, 1).astype(np.float32)
            self.label = data["label"]

        def get_data(self):
            return self.data, self.label

    train_data_loader = MyData(True).get_data()
    test_data_loader = MyData(False).get_data()

    model = Net()
    model.build((None, height, width, 1))

    model.summary()

    # Use the latest optimizer API
    model.compile(
        loss=tf2.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf2.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    model.fit(*train_data_loader, batch_size=128, epochs=n_epoch, validation_split=0.1, verbose=1)

    test_score = model.evaluate(*test_data_loader, verbose=2)
    print(f"[{dt.now()}], loss = {test_score[0]}, accuracy = {test_score[1]}")

    # Freeze the graph using tf.function and get_concrete_function
    full_model = tf2.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(tf2.TensorSpec([None, height, width, 1], tf2.float32, name="x"))

    # Convert to a constant graph
    frozen_func = convert_variables_to_constants_v2(full_model)

    print("_________________________________________________________________")
    print("Frozen model inputs:\n", frozen_func.inputs)
    print("Frozen model outputs:\n", frozen_func.outputs)
    print("Frozen model layers:")
    for op in frozen_func.graph.get_operations():
        print(op.name)

    # Write the frozen graph
    pbfile_path.mkdir(parents=True, exist_ok=True)
    tf2.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(pbfile_path), name=pbFile.name, as_text=False)

    print("Succeed building model in TensorFlow2")

    # Export the model to an ONNX file
    # Use a newer opset version
    command = ["python3", "-m", "tf2onnx.convert", "--opset", "17", "--inputs-as-nchw", "x:0", "--output", str(onnx_file_trained), "--input", str(pbFile), "--inputs", "x:0", "--outputs", "Identity:0"]
    subprocess.run(command, check=True)

    model_onnx = onnx.load(str(onnx_file_trained))
    for value in list(model_onnx.graph.input) + list(model_onnx.graph.output):
        dims = value.type.tensor_type.shape.dim
        if len(dims) > 0:
            dims[0].dim_param = "N"
            if dims[0].HasField("dim_value"):
                dims[0].ClearField("dim_value")
    onnx.save(model_onnx, str(onnx_file_trained))

    print("Succeed converting model into ONNX")

@case_mark
def case_normal(is_fp16: bool = False, is_int8_ptq: bool = False):
    data_local = {"x": data.reshape(-1, 1, height, width)}
    shape = list(data_local["x"].shape)

    tw = TRTWrapperV1()

    parser = trt.OnnxParser(tw.network, tw.logger)
    with open(onnx_file_trained, "rb") as model:
        if not parser.parse(model.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

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
    for pattern in ("*.trt*", "*.Int8Cache", "*.onnx"):
        for target_path in Path(".").glob(pattern):
            target_path.unlink(missing_ok=True)
    shutil.rmtree(pbfile_path, ignore_errors=True)

    case_get_onnx()  # Get model
    case_normal()
    case_normal(is_fp16=True)
    case_normal(is_int8_ptq=True)

    print("Finish")
