import os
import numpy as np
from cuda import cudart
import tensorrt as trt

class EncoderCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, npzFile, cacheFile, nBS, nSL):  # BatchSize,SequenceLength,Calibration
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.data = np.load(npzFile)
        self.cacheFile = cacheFile
        self.nBS = nBS
        self.nSL = nSL
        self.iB = 0

        self.keyName0 = "speech-" + str(nSL)
        self.keyName1 = "speech_lengths-" + str(nSL)
        self.bufferSize0 = np.zeros([nBS, nSL, 80], dtype=np.float32).nbytes
        self.bufferSize1 = np.zeros([nBS], dtype=np.int32).nbytes
        self.bufferD = []
        self.bufferD.append(cudart.cudaMalloc(self.bufferSize0)[1])
        self.bufferD.append(cudart.cudaMalloc(self.bufferSize1)[1])
        print("> Encoder calibrator constructor")

    def __del__(self):
        cudart.cudaFree(self.bufferD[0])
        cudart.cudaFree(self.bufferD[1])

    def get_batch_size(self):  # do NOT change name
        return self.nBS

    def get_batch(self, nameList=None, inputNodeName=None):  # do NOT change name
        print("> Calibration: %d/%d" % (self.iB, 100 // self.nBS))
        if (self.iB + 1) * self.nBS > 100:
            return None

        batchData0 = self.data[self.keyName0][self.iB * self.nBS:(self.iB + 1) * self.nBS]
        batchData1 = self.data[self.keyName1][self.iB * self.nBS:(self.iB + 1) * self.nBS]
        cudart.cudaMemcpy(self.bufferD[0], batchData0.ctypes.data, self.bufferSize0, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        cudart.cudaMemcpy(self.bufferD[1], batchData1.ctypes.data, self.bufferSize1, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        self.iB += 1

        return self.bufferD

    def read_calibration_cache(self):  # do NOT change name
        if os.path.exists(self.cacheFile):
            print("Succeed finding int8 cahce: %s" % (self.cacheFile))
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return None

    def write_calibration_cache(self, cache):  # do NOT change name
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache: %s" % (self.cacheFile))

class DecoderCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, npzFile, cacheFile, nBS, nSL):  # BatchSize,SequenceLength,Calibration
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.data = np.load(npzFile)
        self.cacheFile = cacheFile
        self.nBS = nBS
        self.nSL = nSL
        self.iB = 0

        self.keyName0 = "encoder_out-" + str(nSL)
        self.keyName1 = "encoder_out_lens-" + str(nSL)
        self.keyName2 = "hyps_pad_sos_eos-" + str(nSL)
        self.keyName3 = "hyps_lens_sos-" + str(nSL)
        self.keyName4 = "ctc_score-" + str(nSL)
        self.bufferSize0 = np.zeros([nBS, nSL, 256], dtype=np.float32).nbytes
        self.bufferSize1 = np.zeros([nBS], dtype=np.int32).nbytes
        self.bufferSize2 = np.zeros([nBS, 10, 64], dtype=np.int32).nbytes
        self.bufferSize3 = np.zeros([nBS, 10], dtype=np.int32).nbytes
        self.bufferSize4 = np.zeros([nBS, 10], dtype=np.float32).nbytes
        self.bufferD = []
        self.bufferD.append(cudart.cudaMalloc(self.bufferSize0)[1])
        self.bufferD.append(cudart.cudaMalloc(self.bufferSize1)[1])
        self.bufferD.append(cudart.cudaMalloc(self.bufferSize2)[1])
        self.bufferD.append(cudart.cudaMalloc(self.bufferSize3)[1])
        self.bufferD.append(cudart.cudaMalloc(self.bufferSize4)[1])
        print("> Decoder calibrator constructor")

    def __del__(self):
        for i in range(5):
            cudart.cudaFree(self.bufferD[i])

    def get_batch_size(self):  # do NOT change name
        return self.nBS

    def get_batch(self, nameList=None, inputNodeName=None):  # do NOT change name
        print("> Calibration: %d/%d" % (self.iB, 100 // self.nBS))
        if (self.iB + 1) * self.nBS > 100:
            return None

        batchData0 = self.data[self.keyName0][self.iB * self.nBS:(self.iB + 1) * self.nBS]
        batchData1 = self.data[self.keyName1][self.iB * self.nBS:(self.iB + 1) * self.nBS]
        batchData2 = self.data[self.keyName2][self.iB * self.nBS:(self.iB + 1) * self.nBS]
        batchData3 = self.data[self.keyName3][self.iB * self.nBS:(self.iB + 1) * self.nBS]
        batchData4 = self.data[self.keyName4][self.iB * self.nBS:(self.iB + 1) * self.nBS]
        cudart.cudaMemcpy(self.bufferD[0], batchData0.ctypes.data, self.bufferSize0, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        cudart.cudaMemcpy(self.bufferD[1], batchData1.ctypes.data, self.bufferSize1, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        cudart.cudaMemcpy(self.bufferD[2], batchData2.ctypes.data, self.bufferSize2, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        cudart.cudaMemcpy(self.bufferD[3], batchData3.ctypes.data, self.bufferSize3, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        cudart.cudaMemcpy(self.bufferD[4], batchData4.ctypes.data, self.bufferSize4, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        self.iB += 1

        return self.bufferD

    def read_calibration_cache(self):  # do NOT change name
        if os.path.exists(self.cacheFile):
            print("Succeed finding int8 cahce: %s" % (self.cacheFile))
            with open(self.cacheFile, "rb") as f:
                cache = f.read()
                return cache
        else:
            print("Failed finding int8 cache!")
            return None

    def write_calibration_cache(self, cache):  # do NOT change name
        with open(self.cacheFile, "wb") as f:
            f.write(cache)
        print("Succeed saving int8 cache: %s" % (self.cacheFile))

if __name__ == "__main__":
    cudart.cudaDeviceSynchronize()
    nBS = 4
    nSL = 64
    m = EncoderCalibrator("/workspace/data/calibration.npz", "./testCalibrator.cache", nBS, nSL)
    for i in range((100 + nBS - 1) // nBS + 1):
        print("%2d->" % i, m.get_batch("FakeNameList"))

    m = DecoderCalibrator("/workspace/data/calibration.npz", "./testCalibrator.cache", nBS, nSL)
    for i in range((100 + nBS - 1) // nBS + 1):
        print("%2d->" % i, m.get_batch("FakeNameList"))
