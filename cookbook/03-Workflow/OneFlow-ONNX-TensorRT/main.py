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
import oneflow as flow
import oneflow.nn as nn
import tensorrt as trt
# from tensorrt_cookbook import CookbookCalibratorMNIST, TRTWrapperV1, case_mark

np.random.seed(31193)
flow.manual_seed(97)
batch_size, height, width = 128, 28, 28
n_epoch = 10
data_path = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data"
train_data_file = data_path / "TrainData.npz"
test_data_file = data_path / "TestData.npz"
onnx_file_trained = Path("model.onnx")

data = np.load(data_path / "InferenceData.npy")
calibration_data_file = data_path / "CalibrationData.npy"
trt_file = Path("model.trt")
int8_cache_file = Path("model.Int8Cache")

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = flow.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = flow.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = flow.relu(x)
        y = self.fc2(x)
        return y

def case_get_onnx():

    class MyData:

        def __init__(self, b_train: bool = True):
            data = np.load(train_data_file if b_train else test_data_file)
            self.data = data["data"].reshape(-1, 1, height, width).astype(np.float32)
            self.label = np.argmax(data["label"], axis=1).astype(np.int64)

        def get_data(self):
            return self.data, self.label

    x_train, y_train = MyData(True).get_data()
    x_test, y_test = MyData(False).get_data()

    device = "cuda" if flow.cuda.is_available() else "cpu"
    model = Net().to(device)
    optimizer = flow.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(n_epoch):
        model.train()
        index = np.random.choice(len(x_train), size=batch_size, replace=False)
        x_batch = flow.tensor(x_train[index], dtype=flow.float32, device=device)
        y_batch = flow.tensor(y_train[index], dtype=flow.int64, device=device)

        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        model.eval()
        with flow.no_grad():
            y_pred = model(flow.tensor(x_test[:batch_size], dtype=flow.float32, device=device)).cpu().numpy()
        acc = float(np.mean(np.argmax(y_pred, axis=1) == y_test[:batch_size]))
        print(f"[{dt.now()}]Epoch {epoch:2d}, loss = {float(loss.numpy()):.6f}, test acc = {acc:.4f}")

    print("Succeed building model in OneFlow")

    model.eval()
    dummy_input = flow.randn(1, 1, height, width, device=device)
    flow.onnx.export(
        model,
        dummy_input,
        str(onnx_file_trained),
        input_names=["x"],
        output_names=["y"],
        opset_version=17,
        dynamic_axes={
            "x": {
                0: "batch_size"
            },
            "y": {
                0: "batch_size"
            }
        },
    )
    print("Succeed converting model into ONNX")

@case_mark
def case_normal(is_fp16: bool = False, is_int8_ptq: bool = False):
    data_local = {"x": data.reshape(-1, 1, height, width)}
    shape = list(data_local["x"].shape)

    tw = TRTWrapperV1()

    parser = trt.OnnxParser(tw.network, tw.logger)
    with open(onnx_file_trained, "rb") as model_file:
        if not parser.parse(model_file.read()):
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
    for pattern in ("*.trt*", "*.Int8Cache", "*.onnx", "*.log"):
        for target_path in Path(".").glob(pattern):
            target_path.unlink(missing_ok=True)

    case_get_onnx()
    # case_normal()
    # case_normal(is_fp16=True)
    # case_normal(is_int8_ptq=True)

    print("Finish")
