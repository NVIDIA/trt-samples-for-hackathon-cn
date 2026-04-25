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

if hasattr(np, "dtypes") and hasattr(np.dtypes, "StrDType") and not hasattr(np.dtypes, "StringDType"):
    np.dtypes.StringDType = np.dtypes.StrDType

import jax
import jax.numpy as jnp
import jax2onnx
import onnx.helper as onnx_helper
import optax
import tensorrt as trt
from flax import linen as nn

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
cookbook_root = Path(os.getenv("TRT_COOKBOOK_PATH") or Path(__file__).resolve().parents[2])
data_path = cookbook_root / "00-Data" / "data"
train_data_file = data_path / "TrainData.npz"
test_data_file = data_path / "TestData.npz"
onnx_file_trained = Path("model.onnx")

data = np.load(data_path / "InferenceData.npy")
calibration_data_file = data_path / "CalibrationData.npy"
trt_file = Path("model.trt")
int8_cache_file = Path("model.Int8Cache")

class Net(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(5, 5), padding="SAME", name="conv1")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = nn.Conv(features=64, kernel_size=(5, 5), padding="SAME", name="conv2")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=1024, name="dense1")(x)
        x = nn.relu(x)
        y = nn.Dense(features=10, name="dense2")(x)
        return y

def export_onnx_from_params(model: Net, params, output_path: Path):

    if output_path.exists():
        output_path.unlink()
    external_data_path = output_path.with_suffix(output_path.suffix + ".data")
    if external_data_path.exists():
        external_data_path.unlink()

    def infer_fn(x):
        logits = model.apply(params, x)
        return nn.softmax(logits, axis=1)

    jax2onnx.to_onnx(
        infer_fn,
        inputs=[("B", height, width, 1)],
        model_name="JAXMNIST",
        opset=17,
        return_mode="file",
        output_path=str(output_path),
        inputs_as_nchw=[0],
        input_names=["x"],
        output_names=["y"],
    )

def case_get_onnx():

    class MyData:

        def __init__(self, b_train: bool = True):
            data = np.load(train_data_file if b_train else test_data_file)
            self.data = data["data"].reshape(-1, height, width, 1).astype(np.float32)
            self.label = data["label"].astype(np.float32)

        def get_data(self):
            return self.data, self.label

    x_train, y_train = MyData(True).get_data()
    x_test, y_test = MyData(False).get_data()

    model = Net()
    params = model.init(jax.random.PRNGKey(97), jnp.ones((1, height, width, 1), dtype=jnp.float32))
    tx = optax.adam(1e-3)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(current_params, current_opt_state, x_batch, y_batch):

        def loss_fn(p):
            logits = model.apply(p, x_batch)
            loss = jnp.mean(optax.softmax_cross_entropy(logits, y_batch))
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(current_params)
        updates, current_opt_state = tx.update(grads, current_opt_state)
        current_params = optax.apply_updates(current_params, updates)
        return current_params, current_opt_state, loss

    @jax.jit
    def eval_step(current_params, x_batch):
        logits = model.apply(current_params, x_batch)
        return nn.softmax(logits, axis=1)

    n_train = len(x_train)
    n_batch = n_train // batch_size

    for epoch in range(n_epoch):
        index = np.random.permutation(n_train)
        epoch_loss = 0.0

        for i in range(n_batch):
            batch_index = index[i * batch_size:(i + 1) * batch_size]
            x_batch = jnp.asarray(x_train[batch_index])
            y_batch = jnp.asarray(y_train[batch_index])
            params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
            epoch_loss += float(loss)

        y_pred = np.asarray(eval_step(params, jnp.asarray(x_test)))
        acc = float(np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)))
        avg_loss = epoch_loss / max(n_batch, 1)
        print(f"[{dt.now()}]Epoch {epoch:2d}, loss = {avg_loss:.6f}, test acc = {acc:.4f}")

    print("Succeed building model in JAX")

    export_onnx_from_params(model, params, onnx_file_trained)
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
    for pattern in ("*.trt*", "*.Int8Cache", "*.onnx"):
        for target_path in Path(".").glob(pattern):
            target_path.unlink(missing_ok=True)

    case_get_onnx()
    case_normal()
    case_normal(is_fp16=True)
    case_normal(is_int8_ptq=True)

    print("Finish")
