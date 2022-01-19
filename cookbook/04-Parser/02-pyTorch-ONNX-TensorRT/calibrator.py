#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#

import os
import cv2
import numpy as np
from glob import glob
from cuda import cuda
import tensorrt as trt

class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibrationDataPath, calibrationCount, inputShape, cacheFile):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.imageList          = glob(calibrationDataPath + "*.jpg")[:100]
        self.calibrationCount   = calibrationCount
        self.shape              = inputShape                    # (N,C,H,W)
        self.buffeSize          = trt.volume(inputShape) * trt.float32.itemsize
        self.cacheFile          = cacheFile
        _,self.dIn              = cuda.cuMemAlloc(self.buffeSize)
        self.oneBatch           = self.batchGenerator()

        print(int(self.dIn))

    def __del__(self):
        cuda.cuMemFree(self.dIn)

    def batchGenerator(self):
        for i in range(self.calibrationCount):
            print("> calibration %d"%i)
            subImageList = np.random.choice(self.imageList, self.shape[0], replace=False)
            yield np.ascontiguousarray(self.loadImageList(subImageList))

    def loadImageList(self, imageList):
        res = np.empty(self.shape, dtype = np.float32)
        for i in range(self.shape[0]):
            res[i,0] = cv2.imread(imageList[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        return res

    def get_batch_size(self):                                                                       # do NOT change name
        return self.shape[0]

    def get_batch(self, nameList = None, inputNodeName = None):                                     # do NOT change name
        try:
            data = next(self.oneBatch)
            cuda.cuMemcpyHtoD(self.dIn, data.ctypes.data, self.buffeSize)
            return [int(self.dIn)]
        except StopIteration:
            return None

    def read_calibration_cache(self):                                                               # do NOT change name
        if os.path.exists(self.cacheFile):
            print( "Succeed finding cahce file: %s" %(self.cacheFile) )
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return

    def write_calibration_cache(self, cache):                                                       # do NOT change name
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache!")

if __name__ == "__main__":
    cuda.cuInit(0)
    cuda.cuDeviceGet(0)
    m = MyCalibrator("../mnistData/test/", 5, (1,1,28,28), "./int8.cache")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
    m.get_batch("FakeNameList")
