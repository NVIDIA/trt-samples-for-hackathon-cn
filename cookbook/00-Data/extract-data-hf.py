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

import pyarrow.parquet as pq

def extract_split(parquet_path, output_dir, max_count):
    print("Extracting", parquet_path)
    table = pq.read_table(parquet_path, columns=["image", "label"])
    image_column = table.column("image")
    label_column = table.column("label")

    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(max_count, table.num_rows)):
        image_bytes = image_column[i].as_py()["bytes"]
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        label = int(label_column[i].as_py())
        filename = f"{i:04d}-{label}.jpg"
        cv2.imwrite(str(output_dir / filename), image)

if __name__ == "__main__":
    mnist_data_path = Path("./data-hf/mnist/mnist")
    data_path = Path("./data")
    n_train = 6000  # <= 60,000
    n_test = 500  # <= 10,000

    train_picture_path = data_path / "train"
    test_picture_path = data_path / "test"

    extract_split(mnist_data_path / "train-00000-of-00001.parquet", train_picture_path, n_train)
    extract_split(mnist_data_path / "test-00000-of-00001.parquet", test_picture_path, n_test)

    print("Finish")
