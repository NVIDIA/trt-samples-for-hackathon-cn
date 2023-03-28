#
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

import numpy as np
import tensorrt as trt
from cuda import cudart

class MyInt8EntropyCalibrator2(trt.IInt8EntropyCalibrator2):

    def __init__(self, nCalibration, nBatchSize, tensorDictionary, cacheFile):
        super().__init__()

        # some parameters can be passed from TensorRT build time
        self.nCalibration = nCalibration
        self.nBatchSize = nBatchSize
        self.td = tensorDictionary
        self.cacheFile = cacheFile

        self.oneBatch = self.generator()
        for name in self.td.keys():  # add size and device buffer for each input tensor
            item = self.td[name]
            item["size"] = trt.volume((self.nBatchSize, ) + tuple(item["shape"])) * item["dataType"].itemsize
            item["buffer"] = int(cudart.cudaMalloc(item["size"])[1])

    def __del__(self):
        for name in self.td.keys():
            cudart.cudaFree(self.td[name]["buffer"])

    def generator(self):
        for i in range(self.nCalibration):
            print("> calibration %d" % i)  # for debug
            dataDictionary = {}
            for name in self.td.keys():  # create calibration data by name
                data = np.random.rand(np.prod(self.td[name]["shape"])).astype(trt.nptype(self.td[name]["dataType"])).reshape(self.td[name]["shape"])
                dataDictionary[name] = np.ascontiguousarray(data)
            yield dataDictionary

    def get_batch_size(self):  # necessary API
        #print("[MyCalibrator::get_batch_size]")  # for debug
        return self.nBatchSize

    def get_batch(self, nameList=None, inputNodeName=None):  # necessary API
        #print("[MyCalibrator::get_batch]")  # for debug
        assert (set(nameList) == set(self.td.keys()))
        try:
            dataDictionary = next(self.oneBatch)
            bufferD = [None for i in range(len(self.td))]
            for i, name in enumerate(nameList):
                bufferD[i] = self.td[name]["buffer"]
                cudart.cudaMemcpy(bufferD[i], dataDictionary[name].ctypes.data, self.td[name]["size"], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            return bufferD
        except StopIteration:
            return None

    def read_calibration_cache(self):  # necessary API
        #print("[MyCalibrator::read_calibration_cache]")  # for debug
        if os.path.exists(self.cacheFile):
            print("Succeed finding %s" % self.cacheFile)
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding %s" % self.cacheFile)
            return

    def write_calibration_cache(self, cache):  # necessary API
        #print("[MyCalibrator::write_calibration_cache]")  # for debug
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving %s" % self.cacheFile)
        return

# Other calibrators but not recommended to use
class MyInt8Calibrator(trt.IInt8Calibrator):

    def __init__(self, nCalibration, nBatchSize, tensorDictionary, cacheFile):
        super().__init__()

    def __del__(self):
        pass

    def get_batch_size(self):  # necessary API
        return 1

    def get_batch(self, nameList=None, inputNodeName=None):  # necessary API
        return None

    def read_calibration_cache(self):  # necessary API
        return

    def write_calibration_cache(self, cache):  # necessary API
        return

class MyInt8LegacyCalibrator(trt.IInt8LegacyCalibrator):

    def __init__(self, nCalibration, nBatchSize, tensorDictionary, cacheFile):
        super().__init__()

    def __del__(self):
        pass

    def get_batch_size(self):  # necessary API
        return 1

    def get_batch(self, nameList=None, inputNodeName=None):  # necessary API
        return None

    def read_calibration_cache(self):  # necessary API
        return

    def write_calibration_cache(self, cache):  # necessary API
        return

class MyInt8EntropyCalibrator(trt.IInt8EntropyCalibrator):

    def __init__(self, nCalibration, nBatchSize, tensorDictionary, cacheFile):
        super().__init__()

    def __del__(self):
        pass

    def get_batch_size(self):  # necessary API
        return 1

    def get_batch(self, nameList=None, inputNodeName=None):  # necessary API
        return None

    def read_calibration_cache(self):  # necessary API
        return

    def write_calibration_cache(self, cache):  # necessary API
        return

class MyInt8MinMaxCalibrator(trt.IInt8MinMaxCalibrator):

    def __init__(self, nCalibration, nBatchSize, tensorDictionary, cacheFile):
        super().__init__()

    def __del__(self):
        pass

    def get_batch_size(self):  # necessary API
        return 1

    def get_batch(self, nameList=None, inputNodeName=None):  # necessary API
        return None

    def read_calibration_cache(self):  # necessary API
        return

    def write_calibration_cache(self, cache):  # necessary API
        return

if __name__ == "__main__":
    cudart.cudaDeviceSynchronize()
    m = MyInt8EntropyCalibrator2(10, 1, [[1, 1, 28, 28]], [trt.float32], "./model.INT8Cache")
    m.get_batch(["inputT0"])