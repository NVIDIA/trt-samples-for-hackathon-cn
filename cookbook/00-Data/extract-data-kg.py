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
from zipfile import ZipFile

import cv2
import numpy as np

def extract_mnist_data(idx_path, is_image):
    dt = np.dtype(np.uint32).newbyteorder(">")
    with idx_path.open("rb") as byte_stream:
        magic = np.frombuffer(byte_stream.read(4), dtype=dt)[0]

        if is_image:
            if magic != 2051:
                raise ValueError(f"Fail reading image file: {idx_path}")
            n_image = np.frombuffer(byte_stream.read(4), dtype=dt)[0]
            rows = np.frombuffer(byte_stream.read(4), dtype=dt)[0]
            cols = np.frombuffer(byte_stream.read(4), dtype=dt)[0]
            buf = byte_stream.read(rows * cols * n_image)
            return np.frombuffer(buf, dtype=np.uint8).astype(np.float32).reshape(n_image, rows, cols, 1) / 255

        if magic != 2049:
            raise ValueError(f"Fail reading label file: {idx_path}")
        n_label = np.frombuffer(byte_stream.read(4), dtype=dt)[0]
        buf = byte_stream.read(n_label)
        return np.frombuffer(buf, dtype=np.uint8)

def save_images(images, labels, count, output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        label = int(labels[i])
        filename = f"{i:04d}-{label}.jpg"
        cv2.imwrite(str(output_path / filename), (images[i] * 255).astype(np.uint8))

if __name__ == "__main__":
    zip_path = Path("./data-kg/archive.zip")
    extracted_path = Path("./data-kg/extracted")
    data_path = Path("./data")
    n_train = 6000  # <= 60,000
    n_test = 500  # <= 10,000

    extracted_path.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(extracted_path)

    train_image = extract_mnist_data(extracted_path / "train-images.idx3-ubyte", is_image=True)
    train_label = extract_mnist_data(extracted_path / "train-labels.idx1-ubyte", is_image=False).astype(np.float32)
    test_image = extract_mnist_data(extracted_path / "t10k-images.idx3-ubyte", is_image=True)
    test_label = extract_mnist_data(extracted_path / "t10k-labels.idx1-ubyte", is_image=False).astype(np.float32)

    save_images(train_image, train_label, n_train, data_path / "train")
    save_images(test_image, test_label, n_test, data_path / "test")

    print("Finish")
