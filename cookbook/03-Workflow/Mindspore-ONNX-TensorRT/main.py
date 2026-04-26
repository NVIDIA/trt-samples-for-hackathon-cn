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

import mindspore as ms
import mindspore.nn as nn
import numpy as np
import onnx
import onnx.helper as onnx_helper
import tensorrt as trt
from mindspore import Tensor

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

ms.set_seed(97)
ms.set_context(mode=ms.PYNATIVE_MODE)

class Net(nn.Cell):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, pad_mode="pad", padding=2)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, pad_mode="pad", padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(64 * 7 * 7, 1024)
        self.fc2 = nn.Dense(1024, 10)

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        y = self.fc2(x)
        return y

def case_get_onnx():

    class MyData:

        def __init__(self, b_train: bool = True):
            data = np.load(train_data_file if b_train else test_data_file)
            image = data["data"].reshape(-1, 1, height, width).astype(np.float32)
            if image.max() > 1:
                image /= 255.0
            label = data["label"]
            if label.ndim > 1:
                label = np.argmax(label, axis=1)
            self.data = image
            self.label = label.astype(np.int32)

        def get_data(self):
            return self.data, self.label

    x_train, y_train = MyData(True).get_data()
    x_test, y_test = MyData(False).get_data()

    model = Net()
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Adam(model.trainable_params(), learning_rate=1e-3)
    net_with_loss = nn.WithLossCell(model, loss_fn)
    train_net = nn.TrainOneStepCell(net_with_loss, optimizer)
    train_net.set_train()

    n_train = len(x_train)
    for epoch in range(n_epoch):
        permutation = np.random.permutation(n_train)
        loss_sum = 0.0
        n_step = 0

        for i in range(0, n_train, batch_size):
            index = permutation[i:i + batch_size]
            x_batch = Tensor(x_train[index], ms.float32)
            y_batch = Tensor(y_train[index], ms.int32)
            loss = train_net(x_batch, y_batch)
            loss_sum += float(loss.asnumpy())
            n_step += 1

        mean_loss = loss_sum / max(n_step, 1)

        model.set_train(False)
        logits = model(Tensor(x_test, ms.float32)).asnumpy()
        acc = float(np.mean(np.argmax(logits, axis=1) == y_test))
        print(f"[{dt.now()}]Epoch {epoch:2d}, loss = {mean_loss:.6f}, test acc = {acc:.4f}")
        model.set_train(True)

    print("Succeed building model in Mindspore")

    model.set_train(False)
    ms.export(model, Tensor(np.random.randn(1, 1, height, width).astype(np.float32)), file_name=onnx_file_trained.stem, file_format="ONNX")

    # Convert ONNX batch dimension into dynamic dim for TensorRT profile [1, 4]
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
    with open(onnx_file_trained, "rb") as model_file:
        if not parser.parse(model_file.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    x = tw.network.get_input(0)
    x.name = "x"
    tw.profile.set_shape(x.name, [1] + shape[1:], [1] + shape[1:], [4] + shape[1:])
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

    case_get_onnx()
    case_normal()
    case_normal(is_fp16=True)
    case_normal(is_int8_ptq=True)

    print("Finish")
