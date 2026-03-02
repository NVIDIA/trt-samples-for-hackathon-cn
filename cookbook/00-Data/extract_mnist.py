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

# reference: http://yann.lecun.com/exdb/mnist/, https://storage.googleapis.com/cvdf-datasets/mnist/

import argparse
import gzip
from pathlib import Path

import cv2
import numpy as np

class MnistData():

    def __init__(self, dataPath, b_onehot=False, randomSeed=97):
        with open(dataPath + "train-images-idx3-ubyte.gz", "rb") as f:
            self.trainImage = self.extractImage(f)
        with open(dataPath + "train-labels-idx1-ubyte.gz", "rb") as f:
            self.trainLabel = self.extractLabel(f)
        with open(dataPath + "t10k-images-idx3-ubyte.gz", "rb") as f:
            self.testImage = self.extractImage(f)
        with open(dataPath + "t10k-labels-idx1-ubyte.gz", "rb") as f:
            self.testLabel = self.extractLabel(f, b_onehot=b_onehot)

        self.b_onehot = b_onehot
        if self.b_onehot:
            self.trainLabel = self.convertToOneHot(self.trainLabel)
            self.testLabel = self.convertToOneHot(self.testLabel)
        else:
            self.trainLabel = self.trainLabel.astype(np.float32)
            self.testLabel = self.testLabel.astype(np.float32)

        np.random.seed(randomSeed)

    def getBatch(self, batchSize, b_train):
        if b_train:
            index = np.random.choice(len(self.trainImage), batchSize, True)
            return self.trainImage[index], self.trainLabel[index]
        else:
            index = np.random.choice(len(self.testImage), batchSize, True)
            return self.testImage[index], self.testLabel[index]

    def read4Byte(self, byteStream):
        dt = np.dtype(np.uint32).newbyteorder(">")
        return np.frombuffer(byteStream.read(4), dtype=dt)[0]

    def extractImage(self, f):
        print("Extracting", f.name)
        with gzip.GzipFile(fileobj=f) as byteStream:
            if self.read4Byte(byteStream) != 2051:
                raise ValueError("Fail reading file")
            nImage = self.read4Byte(byteStream)
            rows = self.read4Byte(byteStream)
            cols = self.read4Byte(byteStream)
            buf = byteStream.read(rows * cols * nImage)
            return np.frombuffer(buf, dtype=np.uint8).astype(np.float32).reshape(nImage, rows, cols, 1) / 255

    def extractLabel(self, f, b_onehot=False, nClass=10):
        print("Extracting", f.name)
        with gzip.GzipFile(fileobj=f) as byteStream:
            if self.read4Byte(byteStream) != 2049:
                raise ValueError("Fail reading file")
            nLabel = self.read4Byte(byteStream)
            buf = byteStream.read(nLabel)
            return np.frombuffer(buf, dtype=np.uint8)

    def convertToOneHot(self, labelIndex, nClass=10):
        nLabel = labelIndex.shape[0]
        res = np.zeros((nLabel, nClass), dtype=np.float32)
        offset = np.arange(nLabel) * nClass
        res.flat[offset + labelIndex] = 1
        return res

    def saveImage(self, count, output_path, b_train):
        image, label = ([self.testImage, self.testLabel], [self.trainImage, self.trainLabel])[b_train]
        for i in range(count):
            cv2.imwrite(str(output_path) + "/" + str(i).zfill(4) + "-" + str(label[i])[0] + ".jpg", (image[i] * 255).astype(np.uint8))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=Path, default=Path("./data"), help="Path of output train data")
    parser.add_argument('--n_train', type=int, default=6000, choices=range(1, 60001), help="Count of pictures for training, <=60000")
    parser.add_argument('--n_test', type=int, default=500, choices=range(1, 10001), help="Count of pictures for testing <=10000")
    parser.add_argument('--n_calibration', type=int, default=100, choices=range(1, 10001), help="Count of pictures for calibration")
    args = parser.parse_args()

    # Extract .gz into .jpg
    train_picture_path = args.data_path / "train"
    test_picture_path = args.data_path / "test"
    mnist = MnistData("./", b_onehot=False)
    train_picture_path.mkdir(parents=True, exist_ok=True)
    test_picture_path.mkdir(parents=True, exist_ok=True)
    mnist.saveImage(args.n_train, train_picture_path, True)
    mnist.saveImage(args.n_test, test_picture_path, False)

    # Export .jpg into .npz / .npy
    # Train data
    data = np.zeros([args.n_train, 1, 28, 28], dtype=np.float32)
    label = np.zeros([args.n_train, 10], dtype=np.float32)  # onehot
    for i, file in enumerate(sorted(train_picture_path.glob("*.jpg"))[:args.n_train]):
        data[i] = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        label[i, int(str(file).split(".")[-2][-1])] = 1
    np.savez(args.data_path / "TrainData.npz", **{"data": data, "label": label})
    # Calibration data
    data = np.zeros([args.n_calibration, 1, 28, 28], dtype=np.float32)
    for i, file in enumerate(sorted(train_picture_path.glob("*.jpg"))[:args.n_calibration]):
        data[i] = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    np.save(args.data_path / "CalibrationData.npy", data)
    # Test data
    data = np.zeros([args.n_test, 1, 28, 28], dtype=np.float32)
    label = np.zeros([args.n_test, 10], dtype=np.float32)
    for i, file in enumerate(sorted(test_picture_path.glob("*.jpg"))[:args.n_test]):
        data[i] = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        label[i, int(str(file).split(".")[-2][-1])] = 1
    np.savez(args.data_path / "TestData.npz", **{"data": data, "label": label})
    # Inference data
    inference_data = cv2.imread(str(test_picture_path / "0110-8.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    np.save(args.data_path / "InferenceData.npy", inference_data.reshape(1, 1, 28, 28))

    print("Finish")
