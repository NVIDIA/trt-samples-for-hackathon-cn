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

import os
from datetime import datetime as dt
from pathlib import Path

import numpy as np
import paddle
import paddle.nn.functional as F
import tensorrt as trt
from tensorrt_cookbook import MyCalibratorMNIST, TRTWrapperV1, case_mark

np.random.seed(31193)
paddle.seed(97)
batch_size, height, width = 128, 28, 28
n_epoch = 100
data_path = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data"
train_data_file = data_path / "TrainData.npz"
test_data_file = data_path / "TestData.npz"
onnx_file_trained = "./model.onnx"

data = {"x": np.load(data_path / "InferenceData.npy")}
calibration_data_file = data_path / "CalibrationData.npy"
shape = list(data["x"].shape)
trt_file = Path("model.trt")
int8_cache_file = Path("model.Int8Cache")

def case_get_onnx():

    class MyData:

        def __init__(self, b_train: bool = True, batch_size: int = 1):
            data = np.load(train_data_file if b_train else test_data_file)
            self.data = data["data"]
            self.label = data["label"]
            self.offset = 0
            self.batch_size = batch_size
            return

        def __getitem__(self, index: int):
            return self.data[index], self.label[index]

        def __len__(self):
            return len(self.data)

        def get_batch(self, batch_size: int = 0):
            if batch_size == 0:
                batch_size = self.batch_size
            index = np.arange(self.offset, self.offset + batch_size, dtype=np.int32) % len(self)
            self.offset += batch_size
            return self.data[index], self.label[index]

    train_data_loader = MyData(True, batch_size)
    test_data_loader = MyData(False, batch_size)

    class Net(paddle.nn.Layer):

        def __init__(self, num_classes=1):
            super(Net, self).__init__()
            self.conv1 = paddle.nn.Conv2D(1, 32, [5, 5], 1, 2)
            self.pool1 = paddle.nn.MaxPool2D(2, 2)
            self.conv2 = paddle.nn.Conv2D(32, 64, [5, 5], 1, 2)
            self.pool2 = paddle.nn.MaxPool2D(2, 2)
            self.flatten = paddle.nn.Flatten(1)
            self.fc1 = paddle.nn.Linear(64 * 7 * 7, 1024)
            self.fc2 = paddle.nn.Linear(1024, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool1(x)

            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool2(x)

            x = self.flatten(x)
            x = self.fc1(x)
            x = F.relu(x)
            y = self.fc2(x)
            z = F.softmax(y, 1)
            z = paddle.argmax(z, 1)
            return y, z

    model = Net()

    opt = paddle.optimizer.Adam(0.001, parameters=model.parameters())
    for epoch in range(n_epoch):
        model.eval()
        x_train, y_train = train_data_loader.get_batch()
        x_train, y_train = paddle.to_tensor(x_train), paddle.to_tensor(y_train)
        y, z = model(x_train)
        loss = F.cross_entropy(y, paddle.argmax(y_train, 1, keepdim=True))
        loss.backward()
        opt.step()
        opt.clear_grad()

        model.eval()
        acc = 0
        n = 0
        for i in range(len(test_data_loader) // batch_size):
            x_text, y_text = test_data_loader.get_batch()
            x_text, y_text = paddle.to_tensor(x_train), paddle.to_tensor(y_train)
            y, z = model(x_text)
            acc += paddle.sum(z - paddle.argmax(y_text, 1) == 0).numpy().item()
            n += batch_size
        print(f"[{dt.now()}]Epoch {epoch:2d}, loss = {float(loss.data)}, test acc = {acc / n}")

    print("Succeed building model in Paddlepaddle")

    # Export model to ONNX file
    inputDescList = []
    inputDescList.append(paddle.static.InputSpec(shape=[None, 1, height, width], dtype='float32', name='x'))
    paddle.onnx.export(model, onnx_file_trained[:-5], input_spec=inputDescList, opset_version=11)
    print("Succeed converting model into ONNX")

@case_mark
def case_normal(is_fp16: bool = False, is_int8_ptq: bool = False):
    tw = TRTWrapperV1()

    parser = trt.OnnxParser(tw.network, tw.logger)
    with open(onnx_file_trained, "rb") as model:
        parser.parse(model.read())

    input_tensor = tw.network.get_input(0)
    tw.profile.set_shape(input_tensor.name, shape, [1] + shape[1:], [4] + shape[1:])
    tw.config.add_optimization_profile(tw.profile)

    suffix = ""
    if is_fp16:  # FP16 and INT8 can be used at the same time
        print("Using FP16")
        tw.config.set_flag(trt.BuilderFlag.FP16)
        suffix += "-fp16"
    if is_int8_ptq:
        print("Using INT8-PTQ")
        tw.config.set_flag(trt.BuilderFlag.INT8)
        input_info = {"x": [data["x"].dtype, data["x"].shape]}
        tw.config.int8_calibrator = MyCalibratorMNIST(input_info, calibration_data_file, int8_cache_file)
        suffix += "-int8ptq"

    tw.build()
    tw.serialize_engine(Path(str(trt_file) + suffix))

    tw.setup(data)
    tw.infer()
    return

if __name__ == "__main__":
    os.system("rm -rf *.trt* *.Int8Cache")

    case_get_onnx()
    case_normal()
    case_normal(is_fp16=True)
    case_normal(is_int8_ptq=True)

    print("Finish")
