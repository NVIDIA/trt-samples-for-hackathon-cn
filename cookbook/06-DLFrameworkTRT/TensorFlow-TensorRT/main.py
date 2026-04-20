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
import time
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

np.random.seed(31193)
tf.random.set_seed(97)

BATCH_SIZE = 128
HEIGHT = 28
WIDTH = 28
EPOCHS = 10
WARMUP = 50
STEPS = 200

DATA_PATH = Path(os.getenv("TRT_COOKBOOK_PATH")) / "00-Data" / "data"
TRAIN_DATA_FILE = DATA_PATH / "TrainData.npz"
TEST_DATA_FILE = DATA_PATH / "TestData.npz"
INFER_DATA_FILE = DATA_PATH / "InferenceData.npy"

SAVED_MODEL_DIR = Path("saved_model") / "fp32"
TFTRT_MODEL_DIR = Path("saved_model") / "tftrt_fp32"

def setup_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def to_nhwc(data: np.ndarray) -> np.ndarray:
    if data.ndim != 4:
        raise ValueError(f"Expect rank-4 input, got shape={data.shape}")
    if data.shape[-1] == 1:
        return data.astype(np.float32)
    if data.shape[1] == 1:
        return np.transpose(data, (0, 2, 3, 1)).astype(np.float32)
    raise ValueError(f"Can not infer channel axis for input shape={data.shape}")

class Net(tf.keras.Model):

    def __init__(self):
        super().__init__(name="MNISTExample")
        self.conv1 = tf.keras.layers.Conv2D(32, (5, 5), padding="same", activation="relu", name="conv1")
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")
        self.conv2 = tf.keras.layers.Conv2D(64, (5, 5), padding="same", activation="relu", name="conv2")
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.dense1 = tf.keras.layers.Dense(1024, activation="relu", name="dense1")
        self.dense2 = tf.keras.layers.Dense(10, name="dense2")
        self.softmax = tf.keras.layers.Softmax(name="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        logits = self.dense2(self.dense1(x))
        probs = self.softmax(logits)
        pred = tf.argmax(probs, axis=1, output_type=tf.int32)
        return logits, pred

def load_train_test_data() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    train_data = np.load(TRAIN_DATA_FILE)
    test_data = np.load(TEST_DATA_FILE)

    x_train = train_data["data"].reshape(-1, HEIGHT, WIDTH, 1).astype(np.float32)
    y_train = train_data["label"].astype(np.float32)

    x_test = test_data["data"].reshape(-1, HEIGHT, WIDTH, 1).astype(np.float32)
    y_test = test_data["label"].astype(np.float32)
    return (x_train, y_train), (x_test, y_test)

def train_and_export_saved_model() -> Net:
    (x_train, y_train), (x_test, y_test) = load_train_test_data()

    model = Net()
    model.build((None, HEIGHT, WIDTH, 1))
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=1)
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"TensorFlow test result: loss={loss:.6f}, accuracy={acc:.6f}")

    if SAVED_MODEL_DIR.exists():
        shutil.rmtree(SAVED_MODEL_DIR)

    @tf.function(input_signature=[tf.TensorSpec([None, HEIGHT, WIDTH, 1], tf.float32, name="x")])
    def serving_fn(x):
        logits, pred = model(x, training=False)
        return {"logits": logits, "pred": pred}

    tf.saved_model.save(model, str(SAVED_MODEL_DIR), signatures={"serving_default": serving_fn})
    return model

def build_tftrt_saved_model() -> tf.types.experimental.ConcreteFunction:
    if TFTRT_MODEL_DIR.exists():
        shutil.rmtree(TFTRT_MODEL_DIR)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=str(SAVED_MODEL_DIR),
        conversion_params=trt.TrtConversionParams(
            precision_mode=trt.TrtPrecisionMode.FP32,
            max_workspace_size_bytes=1 << 30,
            maximum_cached_engines=1,
        ),
    )
    converter.convert()

    def input_fn():
        sample = np.load(INFER_DATA_FILE)
        sample = to_nhwc(sample)
        for _ in range(4):
            yield (tf.constant(sample, dtype=tf.float32), )

    converter.build(input_fn=input_fn)
    converter.save(str(TFTRT_MODEL_DIR))

    tftrt_model = tf.saved_model.load(str(TFTRT_MODEL_DIR))
    return tftrt_model.signatures["serving_default"]

def build_xla_fn(model: Net) -> tf.types.experimental.ConcreteFunction:

    @tf.function(jit_compile=True)
    def xla_fn(x):
        logits, pred = model(x, training=False)
        return logits, pred

    return xla_fn

def benchmark_tftrt(fn, x: tf.Tensor, warmup: int = WARMUP, steps: int = STEPS) -> float:
    for _ in range(warmup):
        _ = fn(x)

    t0 = time.perf_counter()
    for _ in range(steps):
        _ = fn(x)
    outputs = fn(x)
    _ = outputs["logits"].numpy()
    _ = outputs["pred"].numpy()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / steps

def benchmark_xla(fn, x: tf.Tensor, warmup: int = WARMUP, steps: int = STEPS) -> float:
    for _ in range(warmup):
        _ = fn(x)

    t0 = time.perf_counter()
    for _ in range(steps):
        _ = fn(x)
    logits, pred = fn(x)
    _ = logits.numpy()
    _ = pred.numpy()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / steps

def main() -> None:
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("This sample requires CUDA GPU.")

    setup_gpu()
    model = train_and_export_saved_model()

    x_infer = tf.constant(to_nhwc(np.load(INFER_DATA_FILE)), dtype=tf.float32)

    tftrt_fn = build_tftrt_saved_model()
    xla_fn = build_xla_fn(model)

    tftrt_outputs = tftrt_fn(x_infer)
    xla_logits, xla_pred = xla_fn(x_infer)

    tf.debugging.assert_near(tftrt_outputs["logits"], xla_logits, rtol=1e-2, atol=1e-2)
    tf.debugging.assert_equal(tftrt_outputs["pred"], xla_pred)

    tftrt_latency_ms = benchmark_tftrt(tftrt_fn, x_infer)
    xla_latency_ms = benchmark_xla(xla_fn, x_infer)

    print(f"TensorFlow-TRT latency: {tftrt_latency_ms:.3f} ms")
    print(f"TensorFlow XLA(jit_compile) latency: {xla_latency_ms:.3f} ms")
    faster = "TensorFlow-TRT" if tftrt_latency_ms < xla_latency_ms else "TensorFlow XLA(jit_compile)"
    print(f"Lower latency: {faster}")
    print("Finish")

if __name__ == "__main__":
    main()
