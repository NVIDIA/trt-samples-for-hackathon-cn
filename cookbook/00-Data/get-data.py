# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
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

from pathlib import Path

import cv2
import numpy as np

if __name__ == "__main__":
    mnist_data_path = Path("./data-gz")
    data_path = Path("./data")
    n_train = 6000  # <= 60,000
    n_test = 500  # <= 10,000
    n_calibration = 100  # <= n_train

    train_picture_path = data_path / "train"
    test_picture_path = data_path / "test"

    # Train data
    data = np.zeros([n_train, 1, 28, 28], dtype=np.float32)
    label = np.zeros([n_train, 10], dtype=np.float32)
    for i, file in enumerate(sorted(train_picture_path.glob("*.jpg"))[:n_train]):
        data[i] = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        label[i, int(file.stem.split("-")[-1])] = 1
    np.savez(data_path / "TrainData.npz", **{"data": data, "label": label})

    # Calibration data
    data = np.zeros([n_calibration, 1, 28, 28], dtype=np.float32)
    for i, file in enumerate(sorted(train_picture_path.glob("*.jpg"))[:n_calibration]):
        data[i] = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    np.save(data_path / "CalibrationData.npy", data)

    # Test data
    data = np.zeros([n_test, 1, 28, 28], dtype=np.float32)
    label = np.zeros([n_test, 10], dtype=np.float32)
    for i, file in enumerate(sorted(test_picture_path.glob("*.jpg"))[:n_test]):
        data[i] = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        label[i, int(file.stem.split("-")[-1])] = 1
    np.savez(data_path / "TestData.npz", **{"data": data, "label": label})

    # Inference data
    inference_data = cv2.imread(str(test_picture_path / "0110-8.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    np.save(data_path / "InferenceData.npy", inference_data.reshape(1, 1, 28, 28))
    np.savez(data_path / "InferenceData.npz", x=inference_data.reshape(1, 1, 28, 28))

    print("Finish")
