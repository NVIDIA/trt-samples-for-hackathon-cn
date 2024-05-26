#
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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
#

# reference: http://yann.lecun.com/exdb/mnist/, https://storage.googleapis.com/cvdf-datasets/mnist/

import argparse
import gzip
from pathlib import Path

import cv2
import numpy as np

class MnistData():

    def __init__(self, dataPath, isOneHot=False, randomSeed=97):
        with open(dataPath + "train-images-idx3-ubyte.gz", "rb") as f:
            self.trainImage = self.extractImage(f)
        with open(dataPath + "train-labels-idx1-ubyte.gz", "rb") as f:
            self.trainLabel = self.extractLabel(f)
        with open(dataPath + "t10k-images-idx3-ubyte.gz", "rb") as f:
            self.testImage = self.extractImage(f)
        with open(dataPath + "t10k-labels-idx1-ubyte.gz", "rb") as f:
            self.testLabel = self.extractLabel(f, isOneHot=isOneHot)

        self.isOneHot = isOneHot
        if self.isOneHot:
            self.trainLabel = self.convertToOneHot(self.trainLabel)
            self.testLabel = self.convertToOneHot(self.testLabel)
        else:
            self.trainLabel = self.trainLabel.astype(np.float32)
            self.testLabel = self.testLabel.astype(np.float32)

        np.random.seed(randomSeed)

    def getBatch(self, batchSize, isTrain):
        if isTrain:
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

    def extractLabel(self, f, isOneHot=False, nClass=10):
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

    def saveImage(self, count, outputPath, isTrain):
        if self.isOneHot:
            return
        image, label = ([self.testImage, self.testLabel], [self.trainImage, self.trainLabel])[isTrain]
        for i in range(count):
            cv2.imwrite(str(outputPath) + "/" + str(i).zfill(5) + "-" + str(label[i]) + ".jpg", (image[i] * 255).astype(np.uint8))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_train', type=Path, default=Path("./train"), help="Path of output train data")
    parser.add_argument('--path_test', type=Path, default=Path("./test"), help="Path of output test data")
    parser.add_argument('--n_train', type=int, default=3000, choices=range(1, 60001), help="Count of pictures for training, <=60000")
    parser.add_argument('--n_test', type=int, default=500, choices=range(1, 10001), help="Count of pictures for testing <=10000")
    parser.add_argument('--n_calibration', type=int, default=100, choices=range(1, 10001), help="Count of pictures for calibration")
    args = parser.parse_args()

    mnist = MnistData("./", isOneHot=False)
    args.path_train.mkdir(parents=True, exist_ok=True)
    args.path_test.mkdir(parents=True, exist_ok=True)
    mnist.saveImage(args.n_train, args.path_train, True)
    mnist.saveImage(args.n_test, args.path_test, False)

    inferenceData = cv2.imread(str(args.path_test / "00110-8.0.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)

    nCalibrationData = args.n_calibration
    nHeight, nWidth = inferenceData.shape
    calibrationDataFileList = sorted(args.path_train.glob("*.jpg"))[:nCalibrationData]
    calibrationData = np.empty([nCalibrationData, 1, nHeight, nWidth])
    for i in range(nCalibrationData):
        calibrationData[i, 0] = cv2.imread(str(calibrationDataFileList[i]), cv2.IMREAD_GRAYSCALE).astype(np.float32)

    dataDictionary = {}
    dataDictionary["inferenceData"] = inferenceData
    dataDictionary["calibrationData"] = calibrationData
    np.savez("data.npz", **dataDictionary)

    print("Finish!")
